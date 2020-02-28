

# python run_sglayer.py --neibor relation -e res50  --gpu 3 --batch-size 16 --train-percent 0.1 

# python run_sglayer.py --neibor relation -e res50  --gpu 3 --batch-size 16 --train-percent 0.3 

# python run_sglayer.py --neibor relation -e res50  --gpu 3 --batch-size 16 --train-percent 0.5 

# python run_sglayer.py --neibor relation -e vgg16bn  --gpu 0 --batch-size 16   --train-percent 0.3 
# python run_sglayer.py --neibor relation -e res50  --gpu 0 --batch-size 16   --train-percent 0.7 --pps totally & 




python run_sglayer.py --neibor relation -e res50  --gpu 0 --batch-size 16 --train-percent 0.7 -r pid &
python run_sglayer.py --neibor relation -e res50  --gpu 1 --batch-size 16 --train-percent 0.7 -r age &
python run_sglayer.py --neibor relation -e res50  --gpu 2 --batch-size 16 --train-percent 0.7 -r gender &
python run_sglayer.py --neibor relation -e res50  --gpu 3 --batch-size 16 --train-percent 0.7 -r view &

# python run_sglayer.py -e res50  --gpu 1 --use test & 
# python run_sglayer.py -e vgg16bn  --gpu 1   --use test & 
# python run_sglayer.py -e alex  --gpu 1   --use test & 