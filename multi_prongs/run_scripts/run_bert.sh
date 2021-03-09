cd /extra/yadongl10/git_project/sandbox/multi_prongs/transformer
source activate pytorch


# hyperparemeters
model_type='bert'
epochs=1000
stage='train'
multip_fldr='/extra/yadongl10/git_project/sandbox/multi_prongs'
exp_path='/baldig/physicsprojects2/N_tagger/exp/exp_ptcut/'
exp_name='2020308_search_tiny_bert' #'2020214_search_HLNet_hl3'
exp_dir=${exp_path}/${exp_name}

count=0

for lr in 1e-3
    do
    for batch_size in 256
        do
        for inter_dim in 128
            do
            for num_hidden in 2 4
                do
                for hidden_size in 128 256 # embedding size
                do
                GPU=${count}
                ((count++))
                mkdir -p ${exp_dir}
                cp ${multip_fldr}/run_scripts/run_bert.sh ${exp_dir}
                result_dir=${exp_path}/${exp_name}/${model_type}_embed${hidden_size}_inter_dim${inter_dim}_num_hidden${num_hidden}_lr${lr}_batch_size${batch_size}
                mkdir -p ${result_dir}
                echo ${subsets}
                echo ${result_dir}
                python bert_net.py --inter_dim ${inter_dim} --num_hidden ${num_hidden} --hidden_size ${hidden_size} --model_type ${model_type}\
                --stage ${stage} --lr ${lr}  --batch_size ${batch_size} --result_dir ${result_dir} --GPU ${GPU} --epochs ${epochs} &
                done
            done
        done
    done
done
