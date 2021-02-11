#!/usr/bin/env bash
#!/usr/bin/env bash
# It is recommend to run the bash file in a screen session
cd /extra/yadongl10/git_project/sandbox/multi_prongs
source activate pytorch

# hyperparemeters
epochs=1000
stage='eval'
model_type='GatedHLefpNet'
multip_fldr='/extra/yadongl10/git_project/sandbox/multi_prongs'
exp_path='/baldig/physicsprojects2/N_tagger/exp/efps'
exp_name='2020208_search_lr_1e-3_decay0.5_nowc_weighted_sample_corrected_image_noramed_efp_d5_hl_original'
#exp_name='2020209_search_corrected_image_normed_efp_566_hl_original_gate'
exp_dir=${exp_path}/${exp_name}

#result_dir=${exp_path}/${exp_name}/efp566_${inter_dim}_${latent}_lr${lr}_batch_size${batch_size}
#mkdir -p ${result_dir}
#echo ${result_dir}

# start running
count=0
for strength in 2 3 4 6 #1 5 7 10
    do
    for batch_size in 128
        do
        for inter_dim in 800
            do
            for num_hidden in 5 #7
                do
                GPU=${count}
                ((count++))
                for lr in 1e-3
                    do
                    result_dir=${exp_path}/${exp_name}/efp566_strength${strength}_inter_dim${inter_dim}_num_hidden${num_hidden}_lr${lr}_batch_size${batch_size}
                    echo ${result_dir}
                    python efp_exp.py --inter_dim ${inter_dim} --num_hidden ${num_hidden} --strength ${strength} --model_type ${model_type}\
                    --stage ${stage} --lr ${lr}  --batch_size ${batch_size} --result_dir ${result_dir} --GPU ${GPU} --epochs ${epochs} &
                done
            done
        done
    done
done

wait

