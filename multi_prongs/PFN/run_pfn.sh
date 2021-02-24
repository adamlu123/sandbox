#!/usr/bin/env bash
# It is recommend to run the bash file in a screen session
cd /extra/yadongl10/git_project/sandbox/multi_prongs/PFN
source activate tf180

# hyperparemeters
epochs=1500
stage='train'
multip_fldr='/extra/yadongl10/git_project/sandbox/multi_prongs'
exp_path='/baldig/physicsprojects2/N_tagger/exp'
exp_name='20210223_PFN_search'
exp_dir=${exp_path}/${exp_name}

# start running
count=0
for psize in 100 #200 300
    do
    fsize=${psize}
    for num_hidden in 3 #5
        do
        GPU=${count}
        ((count++))
        for batch_size in 128
            do
            for lr in 1e-3
                do
                mkdir -p ${exp_dir}
                cp ${multip_fldr}/PFN/run_pfn.sh ${exp_dir}
                result_dir=${exp_path}/${exp_name}/num_hidden${num_hidden}_psize${psize}_fsize${fsize}_batchsize${batch_size}_ep${epochs}
                mkdir -p ${result_dir}
                echo ${result_dir}
                python PFN.py --num_hidden ${num_hidden} --psize ${psize} --fsize ${fsize} \
                --stage ${stage} --lr ${lr}  --batch_size ${batch_size} --result_dir ${result_dir} --GPU ${GPU} --epochs ${epochs} &
            done
        done
    done
done



#result_dir=${exp_path}/${exp_name}/efp566_${inter_dim}_${latent}_lr${lr}_batch_size${batch_size}
#mkdir -p ${result_dir}
#echo ${result_dir}