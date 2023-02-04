
python3 run_sglayer_mimiccxr.py --neibor relation -e alex  --gpu 5 --batch-size 16 --train-percent 0.7 -r all --pps totally >./gcn_mimiccxr_alex_aps.log 2>&1 &
python3 run_sglayer_mimiccxr.py --neibor relation -e res50  --gpu 8 --batch-size 16 --train-percent 0.7 -r all --pps totally >./gcn_mimiccxr_res50_aps.log 2>&1  &
python3 run_sglayer_mimiccxr.py --neibor relation -e vgg16bn  --gpu 9 --batch-size 16 --train-percent 0.7 -r all --pps totally >./gcn_mimiccxr_vgg16bn_aps.log 2>&1  &


python3 run_sglayer_mimiccxr.py --neibor relation -e alex  --gpu 3 --batch-size 16 --train-percent 0.7 -r all >./gcn_mimiccxr_alex.log 2>&1 &
python3 run_sglayer_mimiccxr.py --neibor relation -e res50  --gpu 6 --batch-size 16 --train-percent 0.7 -r all >./gcn_mimiccxr_res50.log 2>&1  &
python3 run_sglayer_mimiccxr.py --neibor relation -e vgg16bn  --gpu 7 --batch-size 16 --train-percent 0.7 -r all >./gcn_mimiccxr_vgg16bn.log 2>&1  &
