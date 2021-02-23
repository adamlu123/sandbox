#!/usr/bin/env bash
# It is recommend to run the bash file in a screen session
cd /extra/yadongl10/git_project/sandbox/multi_prongs
source activate pytorch

# hyperparemeters
epochs=1000
stage='eval'
model_type='HLNet'
multip_fldr='/extra/yadongl10/git_project/sandbox/multi_prongs'
exp_path='/baldig/physicsprojects2/N_tagger/exp/efps'
exp_name='2020214_search_HLNet_hl3/efp566_strength0_inter_dim800_num_hidden5_lr1e-3_batch_size128' # '/20200209_HLNet_inter_dim800_num_hidden5'
exp_dir=${exp_path}/${exp_name}

# start running
count=3
for epochs in 1000 #950 900 850
    do
    GPU=${count}
    ((count++))
    for batch_size in 128
        do
        for inter_dim in 800
            do
            for num_hidden in 5 #7
                do
                for lr in 1e-3
                    do
                    result_dir=${exp_dir}
                    echo ${result_dir}
                    python efp_exp.py --inter_dim ${inter_dim} --num_hidden ${num_hidden} --model_type ${model_type}\
                    --stage ${stage} --lr ${lr}  --batch_size ${batch_size} --result_dir ${result_dir} --GPU ${GPU} --epochs ${epochs} --load_pretrained
                done
            done
        done
    done
done

#wait

