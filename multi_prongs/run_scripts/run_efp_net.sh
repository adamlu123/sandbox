#!/usr/bin/env bash
# It is recommend to run the bash file in a screen session
cd /extra/yadongl10/git_project/sandbox/multi_prongs
source activate pytorch

# hyperparemeters
model_type='HLefpNet'  # HLefpNet GatedHLefpNet HLNet
epochs=1500
stage='train'
multip_fldr='/extra/yadongl10/git_project/sandbox/multi_prongs'
exp_path='/baldig/physicsprojects2/N_tagger/exp/tune_HLefpNet'
exp_name='2020321_circularcenter_BN_search_efp_net_rep2'
exp_dir=${exp_path}/${exp_name}
out_dim=128

# start running
count=0
for batch_size in 256
    do
    for inter_dim in 800 # 200 400 600
        do
        for do_rate in 1e-1 2e-1 3e-1 4e-1
            do
            GPU=${count}
            ((count++))
            for num_hidden in 5
                do
                for lr in 1e-4
                    do
                    mkdir -p ${exp_dir}
                    cp ${multip_fldr}/run_scripts/run_efp_net.sh ${exp_dir}
                    result_dir=${exp_path}/${exp_name}/${model_type}_inter_dim${inter_dim}_num_hidden${num_hidden}_out_dim${out_dim}_lr${lr}_batch_size${batch_size}_do${do_rate}
                    mkdir -p ${result_dir}
                    echo ${subsets}
                    echo ${result_dir}
                    python efp_exp.py --inter_dim ${inter_dim} --num_hidden ${num_hidden} --model_type ${model_type}\
                    --stage ${stage} --lr ${lr}  --batch_size ${batch_size} --result_dir ${result_dir} --GPU ${GPU} --epochs ${epochs} &
                done
            done
        done
    done
done

wait



