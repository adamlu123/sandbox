#!/usr/bin/env bash

exp_path='/extra/yadongl10/BIG_sandbox/NN_Capacity/phase_transit/01252020'
log_path=${exp_path}/log

mkdir -p ${exp_path}
mkdir -p ${log_path}


for C in 5 10 15 20
    do
    result_fldr=${exp_path}/n_10000_C_${C}
    mkdir -p ${result_fldr}
    echo ${result_fldr}
    qsub -q arcus.q -P arcus_gpu.p -l hostname=arcus-4 -l gpu=1 -o ${log_path}/n_10000_C_${C}.out -v C=${C},result_fldr=${result_fldr} bash.sh
done
