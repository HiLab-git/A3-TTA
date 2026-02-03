import logging
import os
import csv
import json
import numpy as np
import torch
from tqdm import tqdm
from robustbench.data import get_dataset, convert_2d
from robustbench.utils import load_model, setup_source
from robustbench.tta import setup_norm, setup_tent, setup_a3_tta
from utils.evaluate import get_multi_class_evaluation_score
import SimpleITK as sitk
from conf import cfg, load_cfg_fom_args

logger = logging.getLogger(__name__)


def save_logs(logs, jsonl_path="bank_logs.jsonl", csv_path="bank_logs.csv"):
    """
    logs: list of dict rows (e.g., {"step": int, "BRI": float, "inserted": bool}).
    """
    os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in logs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = set()
    for row in logs:
        fieldnames.update(row.keys())
    fieldnames = sorted(fieldnames)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in logs:
            writer.writerow(row)

    print(f"[+] Saved {len(logs)} rows to:\n    JSONL: {jsonl_path}\n    CSV:   {csv_path}")


def evaluate(description, adaptation_target=True, infer_test_data=True, save=False):
    load_cfg_fom_args(description)
    # configure model
    base_model = load_model(cfg.MODEL.NETWORK, cfg.MODEL.CKPT_DIR,
                       cfg.MODEL.DATASET,cfg.MODEL.METHOD).cuda()
    if cfg.MODEL.METHOD == "source_test":
        logger.info("test-time adaptation: source")
        model = setup_source(base_model)
    elif cfg.MODEL.METHOD == "norm":
        logger.info("test-time adaptation: NORM")
        model = setup_norm(base_model)
        # model = base_model.eval()
    elif cfg.MODEL.METHOD == "tent":
        logger.info("test-time adaptation: TENT")
        config_model, model = setup_tent(base_model)
    elif cfg.MODEL.METHOD == "a3-tta":
        logger.info("test-time adaptation: a3-tta")
        model = setup_a3_tta(base_model)
    else:
        raise "no specific method of {}".format(cfg.MODEL.METHOD)
    # metric = ['dice','assd']
    metric = ['dice','dice']
    save_model_dir = os.path.join('save_model',cfg.MODEL.DATASET+'_'+cfg.MODEL.NETWORK)
    os.makedirs(save_model_dir, exist_ok=True)
    if adaptation_target:
        for epoch_num in tqdm(range(cfg.ADAPTATION.EPOCH), ncols=70):
            if epoch_num == 0:
                try:
                    model.reset()
                    logger.info("resetting model")
                except:
                    logger.warning("not resetting model")
            score_all_data_1 = []
            name_score_list_1= [] 
            score_all_data_11 = []
            name_score_list_11= [] 
            for target_domain in cfg.ADAPTATION.TARGET_DOMAIN:
                # model_source = deepcopy(model)
                model_source = model
                score_all_data_0 = []
                name_score_list_0= []
                score_all_data_01 = []
                name_score_list_01= []
                results = '{}-{}/{}-{}-{}-I-{}-M-{}'.format('results',cfg.MODEL.DATASET,cfg.MODEL.METHOD,cfg.MODEL.DATASET,cfg.MODEL.EXPNAME,target_domain,cfg.SOURCE.SOURCE_DOMAIN)
                os.makedirs(os.path.join(results, 'mask'), exist_ok=True)
                db_all,_,_ = get_dataset(dataset=cfg.MODEL.DATASET, domain=target_domain, online = True)
                train_loader = torch.utils.data.DataLoader(db_all, batch_size = cfg.ADAPTATION.BATCH_SIZE, shuffle=False, drop_last=False, num_workers= 25)
                for i_batch, sampled_batch in enumerate(train_loader):
                    volume_batch, label_batch, names = sampled_batch['image'].cuda(), sampled_batch['label'].cuda(), sampled_batch['names']
                    volume_batch, label_batch = convert_2d(volume_batch, label_batch)
                    with torch.no_grad():
                        if cfg.MODEL.METHOD in ("a3-tta", "meant"):
                            output_soft = model_source(volume_batch,label_batch,names)
                            output_soft = output_soft.softmax(1)
                        else:
                            output_soft = model_source(volume_batch).softmax(1)
                    if epoch_num == (cfg.ADAPTATION.EPOCH - 1):
                        output = output_soft.argmax(1).cpu().numpy()
                        label = label_batch.cpu().numpy().squeeze(1)
                        assert output.shape[0] == len(names)
                        for i in range(len(names)):
                            name = names[i].split('/')[-1]
                            predict_dir  = os.path.join(results, 'mask',name)
                            # # # print(predict_dir)
                            out_lab_obj = sitk.GetImageFromArray(output[i]/1.0)
                            sitk.WriteImage(out_lab_obj, predict_dir)
                            ####
                            score_vector_0 = get_multi_class_evaluation_score(output[i], label[i], cfg.MODEL.NUMBER_CLASS, metric[0] )
                            score_vector_01 = get_multi_class_evaluation_score(output[i], label[i], cfg.MODEL.NUMBER_CLASS, metric[1] )
                            if(cfg.MODEL.NUMBER_CLASS > 2):
                                score_vector_0.append(np.asarray(score_vector_0).mean())
                                score_vector_01.append(np.asarray(score_vector_01).mean())
                                # print(np.asarray(score_vector_0).mean())
                            score_all_data_0.append(score_vector_0)
                            name_score_list_0.append([name] + score_vector_0)
                            score_all_data_1.append(score_vector_0)
                            name_score_list_1.append([name] + score_vector_0)

                            score_all_data_01.append(score_vector_01)
                            name_score_list_01.append([name] + score_vector_01)
                            score_all_data_11.append(score_vector_01)
                            name_score_list_11.append([name] + score_vector_01)
                score_all_data_0 = np.asarray(score_all_data_0)
                score_mean0 = score_all_data_0.mean(axis = 0)
                score_std0  = score_all_data_0.std(axis = 0)

                score_all_data_01 = np.asarray(score_all_data_01)
                score_mean01 = score_all_data_01.mean(axis = 0)
                score_std01  = score_all_data_01.std(axis = 0)
                name_score_list_0.append(['mean'] + list(score_mean0))
                name_score_list_0.append(['std'] + list(score_std0))
                name_score_list_01.append(['mean'] + list(score_mean01))
                name_score_list_01.append(['std'] + list(score_std01))
                score_csv0 = "{0:}/test_{1:}_all.csv".format(results, metric[0])
                score_csv01 = "{0:}/test_{1:}_all.csv".format(results, metric[1])
                with open(score_csv0, mode='w') as csv_file:
                    csv_writer = csv.writer(csv_file, delimiter=',', 
                                    quotechar='"',quoting=csv.QUOTE_MINIMAL)
                    head = ['image'] + ["class_{0:}".format(i) for i in range(1,cfg.MODEL.NUMBER_CLASS)]
                    if(cfg.MODEL.NUMBER_CLASS > 2):
                        head = head + ["average"]
                    csv_writer.writerow(head)
                    for item in name_score_list_0:
                        csv_writer.writerow(item)
                    print('**********',target_domain,'**********')
                    print("Test dice: {0:} mean ".format(metric[0]), score_mean0)
                    print("Test dice: {0:} std  ".format(metric[0]), score_std0) 
                with open(score_csv01, mode='w') as csv_file:
                    csv_writer = csv.writer(csv_file, delimiter=',', 
                                    quotechar='"',quoting=csv.QUOTE_MINIMAL)
                    head = ['image'] + ["class_{0:}".format(i) for i in range(1,cfg.MODEL.NUMBER_CLASS)]
                    if(cfg.MODEL.NUMBER_CLASS > 2):
                        head = head + ["average"]
                    csv_writer.writerow(head)
                    for item in name_score_list_01:
                        csv_writer.writerow(item)
                    print('**********',target_domain,'**********')
                    print("Test dice: {0:} mean ".format(metric[1]), score_mean01)
                    print("Test dice: {0:} std  ".format(metric[1]), score_std01) 
                                    
                    torch.save(model.state_dict(),'{}/{}-{}-{}-model-latest.pth'.format(save_model_dir,cfg.MODEL.METHOD,cfg.SOURCE.SOURCE_DOMAIN,cfg.MODEL.EXPNAME))
            score_all_data_1 = np.asarray(score_all_data_1)
            score_mean1 = score_all_data_1.mean(axis = 0)
            score_std1  = score_all_data_1.std(axis = 0)
            print('**********','average','**********')
            print("Test dice: {0:} mean ".format(metric[0]), score_mean1)
            print("Test dice: {0:} std  ".format(metric[0]), score_std1) 
            score_all_data_11 = np.asarray(score_all_data_11)
            score_mean11 = score_all_data_11.mean(axis = 0)
            score_std11  = score_all_data_11.std(axis = 0)
            print('**********','average','**********')
            print("Test dice: {0:} mean ".format(metric[1]), score_mean11)
            print("Test dice: {0:} std  ".format(metric[1]), score_std11) 

if __name__ == "__main__":
    evaluate('mms train source.',adaptation_target = True, infer_test_data = True,save = True)

