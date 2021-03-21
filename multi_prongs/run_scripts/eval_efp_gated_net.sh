#!/usr/bin/env bash
# It is recommend to run the bash file in a screen session
cd /extra/yadongl10/git_project/sandbox/multi_prongs
source activate pytorch

# hyperparemeters
epochs=1500
stage='eval'
model_type='GatedHLefpNet'
multip_fldr='/extra/yadongl10/git_project/sandbox/multi_prongs'
exp_path='/baldig/physicsprojects2/N_tagger/exp/exp_ptcut'
exp_name='2020310_search_efp_gated_net'
exp_dir=${exp_path}/${exp_name}

# start running
count=0

for do_rate in 4e-1 #1000 950 900 850
    do
    GPU=${count}
    ((count++))
    for strength in 6 #1 2 3 4 5 6 7 10
        do
        for batch_size in 256
            do
            for inter_dim in 800
                do
                for num_hidden in 5 #7
                    do
                    for lr in 1e-4
                        do
                        result_dir=${exp_path}/${exp_name}/model${model_type}_strength${strength}_inter_dim${inter_dim}_num_hidden${num_hidden}_lr${lr}_batch_size${batch_size}_do${do_rate}
                        echo ${result_dir}
                        python efp_exp.py --inter_dim ${inter_dim} --num_hidden ${num_hidden} --strength ${strength} --model_type ${model_type}\
                        --stage ${stage} --lr ${lr}  --batch_size ${batch_size} --result_dir ${result_dir} --GPU ${GPU} --epochs ${epochs} --load_pretrained
                    done
                done
            done
        done
    done
done

#wait

