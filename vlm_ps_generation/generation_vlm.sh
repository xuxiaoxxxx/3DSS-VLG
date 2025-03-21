set -x

exp_dir=$1
config=$2

mkdir -p ${exp_dir}

result_dir=${exp_dir}/result_eval

export PYTHONPATH=.
python -u vlm_ps_generation/generation_vlm.py \
  --config=${config} \
  save_folder ${result_dir}/best \
  2>&1 | tee -a ${exp_dir}/generation_vlm-$(date +"%Y%m%d_%H%M").log