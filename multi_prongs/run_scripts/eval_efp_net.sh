#!/usr/bin/env bash
#!/usr/bin/env bash
# It is recommend to run the bash file in a screen session
cd /extra/yadongl10/git_project/sandbox/multi_prongs
source activate pytorch

# hyperparemeters
epochs=1000
stage='eval'
model_type='HLefpNet'
multip_fldr='/extra/yadongl10/git_project/sandbox/multi_prongs'
exp_path='/baldig/physicsprojects2/N_tagger/exp/exp_ptcut'
exp_name='2020309_search_efp_net'

exp_dir=${exp_path}/${exp_name}

# start running
count=0
for batch_size in 256
    do
    for inter_dim in 800 # 200 400 600
        do
        for do_rate in 4e-1 #2e-1 3e-1 #4e-1
            do
            GPU=${count}
            ((count++))
            for num_hidden in 5
                do
                for lr in 1e-4
                    do
                    result_dir=${exp_path}/${exp_name}/${model_type}_inter_dim${inter_dim}_num_hidden${num_hidden}_lr${lr}_batch_size${batch_size}_do${do_rate}
                    echo ${subsets}
                    echo ${result_dir}
                    python efp_exp.py --inter_dim ${inter_dim} --num_hidden ${num_hidden} --model_type ${model_type}\
                    --stage ${stage} --lr ${lr}  --batch_size ${batch_size} --result_dir ${result_dir} --GPU ${GPU} --epochs ${epochs} --load_pretrained &
                done
            done
        done
    done
done

wait


#
#for epochs in 0 #1000 950 900 850
#    do
#    GPU=${count}
#    ((count++)) &
#    for strength in 1 #2 3 4 5 6 7 10
#        do
#        for batch_size in 256
#            do
#            for inter_dim in 500
#                do
#                for num_hidden in 7 #7
#                    do
#                    for lr in 1e-3
#                        do
##                        result_dir=${exp_path}/${exp_name}/efp566_strength${strength}_inter_dim${inter_dim}_num_hidden${num_hidden}_lr${lr}_batch_size${batch_size}
#                        result_dir=${exp_path}/${exp_name}/efp566_strength0_inter_dim${inter_dim}_num_hidden${num_hidden}_lr${lr}_batch_size${batch_size}
#                        echo ${result_dir}
#                        python efp_exp.py --inter_dim ${inter_dim} --num_hidden ${num_hidden} --strength ${strength} --model_type ${model_type}\
#                        --stage ${stage} --lr ${lr}  --batch_size ${batch_size} --result_dir ${result_dir} --GPU ${GPU} --epochs ${epochs} --load_pretrained
#                    done
#                done
#            done
#        done
#    done
#done
#
##wait
#
