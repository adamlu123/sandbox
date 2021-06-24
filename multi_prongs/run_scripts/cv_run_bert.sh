#!/usr/bin/env bash
cd /extra/yadongl10/git_project/sandbox/multi_prongs/transformer
source activate pytorch


# hyperparemeters
model_type='bert'
epochs=5000
stage='train' # [eval, train]
multip_fldr='/extra/yadongl10/git_project/sandbox/multi_prongs'
exp_path='/baldig/physicsprojects2/N_tagger/exp/cross_valid'
#exp_name='20200412_correct_etacenter_search_large_bert'
#exp_name='20200403_search_tiny_bert'
exp_name='pl'
exp_dir=${exp_path}/${exp_name}

count=0
for fold_id in 8 9 # 4 5 6 7  #6 7 #0 1 2 3
    do
    for lr in 1e-4
        do
        for batch_size in 256
            do
            for inter_dim in 128
                do
                for num_hidden in 4  # [4, 6]
                    do
                    for hidden_size in 256 # embedding size [256, 512]
                    do
                    GPU=${count}
                    ((count++))
                    mkdir -p ${exp_dir}
                    cp ${multip_fldr}/run_scripts/run_bert.sh ${exp_dir}
                    result_dir=${exp_dir}/fold${fold_id}_${model_type}_embed${hidden_size}_inter_dim${inter_dim}_num_hidden${num_hidden}_lr${lr}_batch_size${batch_size}
                    mkdir -p ${result_dir}
                    echo ${result_dir}
                    python bert_net.py --inter_dim ${inter_dim} --num_hidden ${num_hidden} --hidden_size ${hidden_size} --model_type ${model_type}\
                    --stage ${stage} --lr ${lr}  --batch_size ${batch_size} --result_dir ${result_dir} --GPU ${GPU} \
                    --epochs ${epochs} --fold_id ${fold_id} &
                    done
                done
            done
        done
    done
done
