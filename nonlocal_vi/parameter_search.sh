#!/usr/bin/env bash

exp_path='/extra/yadongl10/git_project/sandbox/nonlocal_vi/results/flow_slab/'
log_path=${exp_path}/log

mkdir -p ${exp_path}
mkdir -p ${log_path}

for p in 100 500 1000
    do
    result_fldr=${exp_path}/p_${p}
    mkdir -p ${result_fldr}
    echo p
    python
done