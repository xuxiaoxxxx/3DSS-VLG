#!/bin/sh
set -x

exp_dir=$1
config=$2

mkdir -p ${exp_dir}
result_dir=${exp_dir}/result_eval

export PYTHONPATH=.
python -u run/evaluate.py \
  --config=${config} \
  save_folder ${result_dir} \
  2>&1 | tee -a ${exp_dir}/eval-$(date +"%Y%m%d_%H%M").log