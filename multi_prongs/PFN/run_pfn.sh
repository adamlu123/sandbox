#!/usr/bin/env bash
# It is recommend to run the bash file in a screen session
source ~/.bash_profile
cd /extra/yadongl10/git_project/sandbox/multi_prongs/PFN
source activate tf180

# hyperparemeters
epochs=1000
stage='train'
multip_fldr='/extra/yadongl10/git_project/sandbox/multi_prongs'
exp_path='/baldig/physicsprojects2/N_tagger/exp/exp_ptcut/'
exp_name='20210307_PFN_search_batch256'
exp_dir=${exp_path}/${exp_name}
mkdir -p exp_dir
lr=1e-3

# start running
count=1
for num_hidden in 4 # 3 7 9
    do
    for psize in 512 256 128 #140 160 170  #128 #50 100 150 200
        do
        fsize=${psize}
        for batch_size in 512
            do
            for dropout in 25e-2 # 1e-1 2e-1 25e-2 3e-1 #5e-1 7e-1
                do
                GPU=${count}
                ((count++))
                mkdir -p ${exp_dir}
                cp ${multip_fldr}/PFN/run_pfn.sh ${exp_dir}
                cp ${multip_fldr}/PFN/PFN.py ${exp_dir}
                result_dir=${exp_path}/${exp_name}/do${dropout}_num_hidden${num_hidden}_psize${psize}_fsize${fsize}_batchsize${batch_size}_ep${epochs}
                mkdir -p ${result_dir}
                echo ${GPU}
                python PFN.py --num_hidden ${num_hidden} --psize ${psize} --fsize ${fsize} --dropout ${dropout}\
                --stage ${stage} --lr ${lr}  --batch_size ${batch_size} --result_dir ${result_dir} --GPU ${GPU} --epochs ${epochs} &
            done
        done
    done
done

# cd /baldig/physicsprojects2/N_tagger/exp/20210223_PFN_search_v2

#result_dir=${exp_path}/${exp_name}/efp566_${inter_dim}_${latent}_lr${lr}_batch_size${batch_size}
#mkdir -p ${result_dir}
#echo ${result_dir}