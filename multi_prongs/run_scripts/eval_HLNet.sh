#!/usr/bin/env bash
# It is recommend to run the bash file in a screen session
cd /extra/yadongl10/git_project/sandbox/multi_prongs
source activate pytorch

# hyperparemeters
epochs=1500
stage='eval'
model_type='HLNet'
multip_fldr='/extra/yadongl10/git_project/sandbox/multi_prongs'
exp_path='/baldig/physicsprojects2/N_tagger/exp/exp_ptcut'
#exp_name='2020214_search_HLNet_hl3/efp566_strength0_inter_dim800_num_hidden5_lr1e-3_batch_size128' #
exp_name='2020308_search_HLnet'
exp_dir=${exp_path}/${exp_name}

# start running
count=0
for do_rate in 4e-1
    do
    GPU=${count}
    ((count++))
    for batch_size in 256
        do
        for inter_dim in 800
            do
            for num_hidden in 5 #7
                do
                for lr in 1e-4
                    do
                    result_dir=${exp_dir}/${model_type}_inter_dim${inter_dim}_num_hidden${num_hidden}_lr${lr}_batch_size${batch_size}_do${do_rate}
                    echo ${result_dir}
                    python efp_exp.py --inter_dim ${inter_dim} --num_hidden ${num_hidden} --model_type ${model_type}\
                    --stage ${stage} --lr ${lr}  --batch_size ${batch_size} --result_dir ${result_dir} --GPU ${GPU} --epochs ${epochs} --load_pretrained
                done
            done
        done
    done
done

#wait

