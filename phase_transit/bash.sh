#!/usr/bin/env bash
C=${C}
result_fldr=${result_fldr}
echo ${result_fldr}
source activate pytorch

cd /extra/yadongl10/git_project/sandbox/phase_transit

echo $PATH
echo ${C}

python single_thres_gate.py --C ${C} --result_dir ${result_fldr}


#python single_thres_gate.py --C 15 --result_dir '/extra/yadongl10/BIG_sandbox/NN_Capacity/phase_transit/01252020/n_10000_C_15'