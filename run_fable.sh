for ds_name in unke
do
    for model_name in Llama3-8B-Instruct
    do
        if [[ $ds_name == unke ]];then
            ds_size=1000
        elif [[ $ds_name == cf ]];then
            ds_size=975
        elif [[ $ds_name == mquake ]];then
            ds_size=354
        fi

        for method_name in FABLE
        do
            echo $model_name $ds_name $ds_size $method_name

            total_machine=1

            for ((gpu=0; gpu<total_machine; gpu++))
            do
                this_machine=$gpu
                bash edit_for_fable.sh $gpu $ds_name $ds_size $model_name $method_name $total_machine $this_machine
            done
        done
    done
done