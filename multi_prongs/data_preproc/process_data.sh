#!/usr/bin/env bash
cd /extra/yadongl10/git_project/sandbox/multi_prongs/data_preproc
source activate pytorch

for res in "res6" #"res3" "res4" "res5" # "res1" "res2" #
    do
    echo ${res}
    python process_data.py --subset ${res} &
done
