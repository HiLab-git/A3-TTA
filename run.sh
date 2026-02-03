python 01_train_source_2d.py --cfg cfgs/prostate/source.yaml
python test-time-adaptation.py --cfg cfgs/prostate/source_test.yaml
python test-time-adaptation.py --cfg cfgs/prostate/norm_prostate.yaml
python test-time-adaptation.py --cfg cfgs/prostate/tent_prostate.yaml
python test-time-adaptation.py --cfg cfgs/prostate/sar_prostate.yaml
python test-time-adaptation.py --cfg cfgs/prostate/a3-tta_prostate.yaml
