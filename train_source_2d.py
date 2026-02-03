import logging
import os
import csv
import numpy as np
import torch
from tqdm import tqdm
from robustbench.data import get_dataset, convert_2d
from robustbench.utils import load_model, setup_source
from utils.evaluate import get_multi_class_evaluation_score
import SimpleITK as sitk
from conf import cfg, load_cfg_fom_args

logger = logging.getLogger(__name__)


def train_source(description, train_source=True, infer_test_data=True, save=False):
    logger.info(f"max_epochs: %s", cfg.SOURCE.MAX_EPOCHES)
    load_cfg_fom_args(description)
    # configure model
    base_model = load_model(cfg.MODEL.NETWORK, cfg.MODEL.CKPT_DIR,
                       cfg.MODEL.DATASET,cfg.MODEL.METHOD).cuda()
    logger.info("train source model: NONE")
    model = setup_source(base_model)
    
    db_train,db_valid,db_test = get_dataset(dataset=cfg.MODEL.DATASET, domain=cfg.SOURCE.SOURCE_DOMAIN,online=True)
    train_loader = torch.utils.data.DataLoader(db_train, 
                    batch_size = cfg.SOURCE.BATCH_SIZE, shuffle=True, num_workers= 50)
    valid_loader = torch.utils.data.DataLoader(db_valid, 
                    batch_size = 1, shuffle=False, num_workers= 16)
    test_loader = torch.utils.data.DataLoader(db_test, 
                    batch_size = 1, shuffle=False, num_workers= 16)
    iter_num = 0
    valid_best = 0
    save_model_dir = os.path.join('save_model', f"{cfg.MODEL.DATASET}_{cfg.MODEL.NETWORK}")
    os.makedirs(save_model_dir, exist_ok=True)
    if train_source:
        model.train()
        for epoch_num in tqdm(range(cfg.SOURCE.MAX_EPOCHES), ncols=cfg.SOURCE.MAX_EPOCHES):
            for i_batch, sampled_batch in enumerate(train_loader):
                iter_num += 1*cfg.SOURCE.BATCH_SIZE
                volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
                volume_batch, label_batch = convert_2d(volume_batch, label_batch)
                model.train_source(volume_batch, label_batch)
            torch.save(model.state_dict(),'{}/{}-{}-{}-model-latest.pth'.format(save_model_dir,cfg.MODEL.METHOD,cfg.SOURCE.SOURCE_DOMAIN,cfg.MODEL.EXPNAME))
        torch.save(model.state_dict(),'{}/{}-{}-{}-model-latest.pth'.format(save_model_dir,cfg.MODEL.METHOD,cfg.SOURCE.SOURCE_DOMAIN,cfg.MODEL.EXPNAME))
    if infer_test_data: 
        best_pth = '{}/{}-{}-model-latest.pth'.format(save_model_dir,cfg.MODEL.METHOD,cfg.SOURCE.SOURCE_DOMAIN)
        best_pth = '/data2/jianghao/TTA-MT/TTA-MT/save_model/prostate2d_unet/source-A-model-latest.pth'
        model.load_state_dict(torch.load(best_pth,map_location='cpu'))
        model.eval()
        for test_domain in cfg.SOURCE.ALL_DOMAIN:
            model.eval()
            db_test,_,_ = get_dataset(dataset=cfg.MODEL.DATASET, domain=test_domain, online=True)
            test_loader = torch.utils.data.DataLoader(db_test, batch_size = 1, shuffle=False, num_workers= 10)
            with torch.no_grad():
                score_all_data_0 = []
                name_score_list_0= []
                score_all_data_1 = []
                name_score_list_1= [] 
                for i, sampled_batch in enumerate(test_loader):
                    volume_batch, label_batch, names, spacing = sampled_batch['image'], sampled_batch['label'], sampled_batch['names'], sampled_batch['spacing']
                    volume_batch, label_batch = convert_2d(volume_batch, label_batch)
                    output_soft = model(volume_batch.cuda()).softmax(1)
                    output = output_soft.argmax(1).cpu().numpy()
                    name = names[0].split('/')[-1]
                    results_root = f"results-{cfg.MODEL.DATASET}"
                    os.makedirs(results_root, exist_ok=True)
                    results = f"{results_root}/{cfg.MODEL.METHOD}-{cfg.MODEL.DATASET}-I-{test_domain}-M-{cfg.SOURCE.SOURCE_DOMAIN}"
                    os.makedirs(os.path.join(results, 'mask'), exist_ok=True)
                    if save:
                        predict_dir  = os.path.join(results, 'mask', name)
                        out_lab_obj = sitk.GetImageFromArray(output/1.0)
                        sitk.WriteImage(out_lab_obj, predict_dir)
                    label = label_batch.cpu().numpy().squeeze(1)
                    metric = ['dice','dice']      
                    score_vector_0 = get_multi_class_evaluation_score(output, label, cfg.MODEL.NUMBER_CLASS, metric[0] )
                    score_vector_1 = get_multi_class_evaluation_score(output, label, cfg.MODEL.NUMBER_CLASS, metric[1] )
                    if(cfg.MODEL.NUMBER_CLASS > 2):
                        score_vector_0.append(np.asarray(score_vector_0).mean())
                        score_vector_1.append(np.asarray(score_vector_1).mean())
                    score_all_data_0.append(score_vector_0)
                    score_all_data_1.append(score_vector_1)
                    name_score_list_0.append([name] + score_vector_0)
                    name_score_list_1.append([name] + score_vector_1)
                score_all_data_0 = np.asarray(score_all_data_0)
                score_mean0 = score_all_data_0.mean(axis = 0)
                score_std0  = score_all_data_0.std(axis = 0)
                name_score_list_0.append(['mean'] + list(score_mean0))
                name_score_list_0.append(['std'] + list(score_std0))
                score_all_data_1 = np.asarray(score_all_data_1)
                score_mean1 = score_all_data_1.mean(axis = 0)
                score_std1  = score_all_data_1.std(axis = 0)
                name_score_list_1.append(['mean'] + list(score_mean1))
                name_score_list_1.append(['std'] + list(score_std1))
                # save the result as csv 
                score_csv0 = "{0:}/test_{1:}_all.csv".format(results, metric[0])
                score_csv1 = "{0:}/test_{1:}_all.csv".format(results, metric[1])
                with open(score_csv0, mode='w') as csv_file:
                    csv_writer = csv.writer(csv_file, delimiter=',', 
                                    quotechar='"',quoting=csv.QUOTE_MINIMAL)
                    head = ['image'] + ["class_{0:}".format(i) for i in range(1,cfg.MODEL.NUMBER_CLASS)]
                    if(cfg.MODEL.NUMBER_CLASS > 2):
                        head = head + ["average"]
                    csv_writer.writerow(head)
                    for item in name_score_list_0:
                        csv_writer.writerow(item)
                with open(score_csv1, mode='w') as csv_file:
                    csv_writer = csv.writer(csv_file, delimiter=',', 
                                    quotechar='"',quoting=csv.QUOTE_MINIMAL)
                    head = ['image'] + ["class_{0:}".format(i) for i in range(1,cfg.MODEL.NUMBER_CLASS)]
                    if(cfg.MODEL.NUMBER_CLASS > 2):
                        head = head + ["average"]
                    csv_writer.writerow(head)
                    for item in name_score_list_1:
                        csv_writer.writerow(item)
                print('****************',test_domain,'****************')
                print("Test data: {0:} mean ".format(metric[0]), score_mean0)
                print("Test data: {0:} std  ".format(metric[0]), score_std0) 
                print("Test data: {0:} mean ".format(metric[1]), score_mean1)
                print("Test data: {0:} std  ".format(metric[1]), score_std1) 




if __name__ == '__main__':
    train_source('mms train source.',train_source = True, infer_test_data = True,save = False)

