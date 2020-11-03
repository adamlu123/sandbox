#!/usr/bin/env bash

count=0
alter_prior='imom' # Gaussian, imom
exp_path="/extra/yadongl10/git_project/nlpresult/20201015/${alter_prior}"
mkdir -p ${exp_path}

source activate pytorch

cp bash.sh ${exp_path}
cp LinearRegressonParameterSearch.py ${exp_path}

for rho in 0.25
    do
    for phi in 1
        do
        for p in 500 1000 1500
            do
            gpu=count
            ((count++))
            for lr in 1e-2
                do
                result_fldr=${exp_path}/lr${lr}_p${p}_phi${phi}_rho${rho}_alter_prior${alter_prior}
                echo ${result_fldr}
                mkdir -p ${result_fldr}
                bash bash.sh ${p} ${phi} ${alter_prior} ${result_fldr} ${gpu} ${lr} ${rho} &
    #            bash bash.sh p=${p},phi=${phi},alter_prior=${alter_prior},result_fldr=${result_fldr},gpu=${gpu}
    #            python LinearRegressonParameterSearch.py --p ${p} --phi ${phi} --alter_prior ${alter_prior} --result_dir ${result_fldr} --gpu ${gpu}
            done
        done
    done
done