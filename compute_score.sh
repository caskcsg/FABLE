gpu=0
model_path=../models/all-MiniLM-L6-v2

post_path=$1
python compute_score.py \
    --pre_path $pre_path \
    --post_path $post_path \
    --model_path $model_path \
    --device $gpu

