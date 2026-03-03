gpu=$1

ds_name=$2
ds_size=$3
model_name=$4
save_weights=None

question_template=1
total_machine=$5
this_machine=$6

log_dir=logs/${ds_name}/${model_name}
mkdir -p $log_dir
log_path=${log_dir}/pre_total${total_machine}_this${this_machine}.log
nohup python -u evaluate_baseline.py \
    --ds_name $ds_name \
    --model_name $model_name \
    --ds_size $ds_size \
    --device $gpu \
    --save_weights $save_weights \
    --question_template $question_template \
    --total_machine $total_machine \
    --this_machine $this_machine \
    > $log_path 2>&1&
