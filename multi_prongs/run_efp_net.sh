#!/usr/bin/env bash
#!/usr/bin/env bash
# It is recommend to run the bash file in a screen session
cd /extra/yadongl10/git_project/sandbox/multi_prongs
source activate pytorch

# hyperparemeters
epochs=300
stage='train'
#sarm_fldr='/the_path_to_sarm_dir'
#save_fldr='/the_path_to_save_result'
multip_fldr='/extra/yadongl10/git_project/sandbox/multi_prongs'
exp_path='/baldig/physicsprojects2/N_tagger/exp/efps'
exp_name='2020207_search_lr_1e-3_decay0.5_nowc_weighted_sample_corrected_image_noramed_efp_d5_hl_original'
exp_dir=${exp_path}/${exp_name}

result_dir=${exp_path}/${exp_name}/${inter_dim}_${latent}_lr${lr}_batch_size${batch_size}
mkdir -p ${result_dir}
echo ${result_dir}

# start running
count=2
for batch_size in 128
    do
    for inter_dim in 50 100
        do
        GPU=${count}
        ((count++))
        for num_hidden in 3 5
            do
            for lr in 1e-4
                do
                mkdir -p ${exp_dir}
                cp ${multip_fldr}/run_efp_net.sh ${exp_dir}
                result_dir=${exp_path}/${exp_name}/inter_dim${inter_dim}_num_hidden${num_hidden}_lr${lr}_batch_size${batch_size}
                mkdir -p ${result_fldr}
                echo ${subsets}
                echo ${result_dir}
                python efp_exp.py --inter_dim ${inter_dim} --num_hidden ${num_hidden} \
                --stage ${stage} --lr ${lr}  --batch_size ${batch_size} --result_dir ${result_dir} --GPU ${GPU} --epochs ${epochs} &
            done
        done
    done
done

wait

