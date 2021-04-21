#!/usr/bin/env bash
# It is recommend to run the bash file in a screen session
cd /extra/yadongl10/git_project/sandbox/multi_prongs
source activate pytorch

# hyperparemeters
epochs=1500
stage='eval' # [train, eval]
model_type='HLNet'
multip_fldr='/extra/yadongl10/git_project/sandbox/multi_prongs'
exp_path='/baldig/physicsprojects2/N_tagger/exp/cross_valid'
#exp_name='2020308_search_HLnet'
exp_name='2020419_small_HLnet'
exp_dir=${exp_path}/${exp_name}
mkdir -p ${exp_dir}
cp ${multip_fldr}/run_scripts/cross_validation/small_hlnet.sh ${exp_dir}

del='pt'
# start running
count=0
echo ${stage}
for fold_id in 8 #4 5 6 7 # 9
    do
    for do_rate in 2e-1
        do
        for batch_size in 256
            do
            for inter_dim in 400 # 50 100 200 400 800 #
                do
                for num_hidden in 5  # 3 5
                    do
                    GPU=${count}
                    ((count++))
                    for lr in 1e-4
                        do
                        result_dir=${exp_dir}/fold${fold_id}_${model_type}_inter_dim${inter_dim}_num_hidden${num_hidden}_lr${lr}_batch_size${batch_size}_do${do_rate}_del${del}
                        mkdir -p ${result_dir}
                        echo ${result_dir}

                        python cv_efp_exp.py --inter_dim ${inter_dim} --num_hidden ${num_hidden} \
                        --model_type ${model_type} --delete ${del}\
                        --stage ${stage} --lr ${lr}  --batch_size ${batch_size} --result_dir ${result_dir} \
                        --GPU ${GPU} --epochs ${epochs} --fold_id ${fold_id} &
                    done
                done
            done
        done
    done
done
#wait

