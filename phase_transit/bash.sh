#!/usr/bin/env bash
C=${C}
result_fldr=${result_fldr}
echo ${result_fldr}
source activate pytorch

cd /extra/yadongl10/git_project/sandbox/phase_transit

echo $PATH

python single_thres_gate.py --C ${C} --result_dir ${result_fldr}
