#!/bin/bash
for model_path in $(ls cifar100_model/model_*.pth);
do
eval_path=./data/whitebox_results/result_$(basename "$model_path").pickle
sbatch start_eval_job.sh $model_path $eval_path
done