for ds_name in unke cf mquake
do
    for model_name in Llama3-8B-Instruct Qwen2.5-7B-Instruct
    do
        if [[ $ds_name == unke ]];then
            ds_size=1000
        elif [[ $ds_name == cf ]];then
            ds_size=975
        elif [[ $ds_name == mquake ]];then
            ds_size=354
        fi

        for method_name in Pre-edited FT-L ROME MEMIT MEMIT_ARE UnKE
        do
            echo $model_name $ds_name $ds_size $method_name

            total_machine=1

            if [[ $method_name == Pre-edited ]];then
                for ((gpu=0; gpu<total_machine; gpu++))
                do
                    this_machine=$gpu
                    # For Pre-edited Evaluation
                    bash evaluate_baseline.sh $gpu $ds_name $ds_size $model_name $total_machine $this_machine
                done
            else
                for ((gpu=0; gpu<total_machine; gpu++))
                do
                    this_machine=$gpu
                    bash edit_for_baseline.sh $gpu $ds_name $ds_size $model_name $method_name $total_machine $this_machine
                done
            fi
        done
    done
done