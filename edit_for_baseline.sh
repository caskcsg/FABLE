gpu=$1

ds_name=$2
ds_size=$3
model_name=$4
method_name=$5

question_template=1
answer_template=1

total_machine=$6
this_machine=$7

log_dir=logs/${ds_name}/${model_name}/${method_name}
mkdir -p $log_dir
experiment_name=ds${ds_size}_qt${question_template}_at${answer_template}_total${total_machine}_this${this_machine}
log_path=${log_dir}/${experiment_name}.log
nohup python -u edit_for_baseline.py \
    --ds_name $ds_name \
    --model_name $model_name \
    --method_name $method_name \
    --ds_size $ds_size \
    --device $gpu \
    --save_dir ./results \
    --question_template $question_template \
    --answer_template $answer_template \
    --total_machine $total_machine \
    --this_machine $this_machine \
    > $log_path 2>&1&