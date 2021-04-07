#!/usr/bin/env bash
# It is recommend to run the bash file in a screen session
source ~/.bash_profile
cd /extra/yadongl10/git_project/sandbox/multi_prongs/PFN
source activate tf180

# hyperparemeters
epochs=1000
stage='train' # [eval, train]
multip_fldr='/extra/yadongl10/git_project/sandbox/multi_prongs'
exp_path='/baldig/physicsprojects2/N_tagger/exp/cross_valid'
exp_name='20210405_correct_etacenter_search_PFN'
exp_dir=${exp_path}/${exp_name}
mkdir -p exp_dir
lr=1e-4

# start running
count=1
for fold_id in 0 8 9 #4 5 6 7 #
    do
    GPU=${count}
    ((count++))
    for num_hidden in 2
        do
        for psize in 1024 #140 160 170  #128 #50 100 150 200
            do
            fsize=${psize}
            for batch_size in 256
                do
                for dropout in 2e-1 #2e-1 3e-1 4e-1 5e-1 #6e-1 # 2e-1 25e-2 #
                    do
                    mkdir -p ${exp_dir}
                    cp ${multip_fldr}/PFN/run_pfn.sh ${exp_dir}
                    cp ${multip_fldr}/PFN/PFN.py ${exp_dir}
                    result_dir=${exp_path}/${exp_name}/fold${fold_id}_do${dropout}_num_hidden${num_hidden}_psize${psize}_fsize${fsize}_batchsize${batch_size}_ep${epochs}_lr${lr}
                    mkdir -p ${result_dir}
                    echo ${GPU}
                    python PFN.py --num_hidden ${num_hidden} --psize ${psize} --fsize ${fsize} --dropout ${dropout} --fold_id ${fold_id}\
                    --stage ${stage} --lr ${lr}  --batch_size ${batch_size} --result_dir ${result_dir} --GPU ${GPU} --epochs ${epochs} &
                done
            done
        done
    done
done
