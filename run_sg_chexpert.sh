
python3 run_sglayer_chexpert.py --neibor relation -e alex  --gpu 5 --batch-size 16 --train-percent 0.7 -r all --pps totally >./gcn_chexpert_alex_aps.log 2>&1 &
python3 run_sglayer_chexpert.py --neibor relation -e res50  --gpu 0 --batch-size 16 --train-percent 0.7 -r all --pps totally >./gcn_chexpert_res50_aps.log 2>&1  &
python3 run_sglayer_chexpert.py --neibor relation -e vgg16bn  --gpu 4 --batch-size 16 --train-percent 0.7 -r all --pps totally >./gcn_chexpert_vgg16bn_aps.log 2>&1  &


python3 run_sglayer_chexpert.py --neibor relation -e alex  --gpu 3 --batch-size 16 --train-percent 0.7 -r all >./gcn_chexpert_alex.log 2>&1 &
python3 run_sglayer_chexpert.py --neibor relation -e res50  --gpu 1 --batch-size 16 --train-percent 0.7 -r all >./gcn_chexpert_res50.log 2>&1  &
python3 run_sglayer_chexpert.py --neibor relation -e vgg16bn  --gpu 2 --batch-size 16 --train-percent 0.7 -r all >./gcn_chexpert_vgg16bn.log 2>&1  &
