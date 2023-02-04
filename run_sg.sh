
source ~/torchv100/bin/activate

python run_sglayer.py --neibor relation -e alex  --gpu 8 --batch-size 16 --train-percent 0.7 -r all --pps totally >./gcn_CXR_alex_aps.log 2>&1 &
python run_sglayer.py --neibor relation -e res50  --gpu 6 --batch-size 16 --train-percent 0.7 -r all --pps totally >./gcn_CXR_res50_aps.log 2>&1 &
python run_sglayer.py --neibor relation -e res50  --gpu 7 --batch-size 16 --train-percent 0.7 -r all  >./gcn_CXR_res50.log 2>&1 &

# python run_sglayer.py --neibor relation -e res50  --gpu 3 --batch-size 16 --train-percent 0.3 

# python run_sglayer.py --neibor relation -e res50  --gpu 3 --batch-size 16 --train-percent 0.5 

# python run_sglayer.py --neibor relation -e vgg16bn  --gpu 0 --batch-size 16   --train-percent 0.3 
# python run_sglayer.py --neibor relation -e res50  --gpu 0 --batch-size 16   --train-percent 0.7 --pps totally & 

# python run_sglayer.py --neibor relation -e alex  --gpu 0 --batch-size 16 --train-percent 0.7 -r all &
# python run_sglayer.py --neibor relation -e res50  --gpu 1 --batch-size 16 --train-percent 0.7 -r all &
# python run_sglayer.py --neibor relation -e vgg16bn  --gpu 2 --batch-size 16 --train-percent 0.7 -r all &
# python run_sglayer.py --neibor relation -e dens121  --gpu 3 --batch-size 16 --train-percent 0.7 -r all &

# python run_sglayer.py --neibor relation -e vgg16bn  --gpu 1 --batch-size 16 --train-percent 0.7 -r all --weight_decay 0.01 --epochs 30 &
# python run_sglayer.py --neibor relation -e vgg16bn  --gpu 1 --batch-size 16 --train-percent 0.7 -r pid --weight_decay 0.01 --epochs 30 &
# python run_sglayer.py --neibor relation -e vgg16bn  --gpu 3 --batch-size 16 --train-percent 0.7 -r age --weight_decay 0.01 --epochs 30 &
# python run_sglayer.py --neibor relation -e vgg16bn  --gpu 2 --batch-size 16 --train-percent 0.7 -r gender --weight_decay 0.01 --epochs 30 &
# python run_sglayer.py --neibor relation -e vgg16bn  --gpu 0 --batch-size 16 --train-percent 0.7 -r view --weight_decay 0.01 --epochs 30 &


# python run_sglayer.py --neibor relation -e res50  --gpu 0 --batch-size 16 --train-percent 0.7 -r pid &
# python run_sglayer.py --neibor relation -e res50  --gpu 1 --batch-size 16 --train-percent 0.7 -r age &
# python run_sglayer.py --neibor relation -e res50  --gpu 2 --batch-size 16 --train-percent 0.7 -r gender &
# python run_sglayer.py --neibor relation -e res50  --gpu 3 --batch-size 16 --train-percent 0.7 -r view &

# python run_sglayer.py -e res50  --gpu 1 --use test & 
# python run_sglayer.py -e vgg16bn  --gpu 1   --use test & 
# python run_sglayer.py -e alex  --gpu 1   --use test & 