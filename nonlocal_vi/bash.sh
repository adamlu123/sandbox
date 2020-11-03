#!/usr/bin/env bash
p=$1
phi=$2
alter_prior=$3
result_fldr=$4
gpu=$5
lr=$6
rho=$7

echo ${result_fldr}
echo ${p}

source activate pytorch
cd /extra/yadongl10/git_project/sandbox/nonlocal_vi

python LinearRegressonParameterSearch.py --p ${p} --phi ${phi} --alter_prior ${alter_prior} --result_dir ${result_fldr} --gpu ${gpu} --lr ${lr} --rho ${rho}

