#!/bin/bash
cd /root/autodl-tmp/PMME_and_Others_v3_20261020
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pmme
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
python -u train_DA_OT.py     --finetune_epochs 300 --train_epochs 3000     --data_list chengdu_shenzhen_pems --test_dataset metr-la     --config_filename ./configs/config.yaml     --seed 1 --classifier 0.3 --xySquaredRatio 10 --DAcoef 0.1     --DAepoch 1500 --DAwarmup 1500 --MemSize 10000 --projection 0     --DA OT --model_name SOFTS     --expert_guide_scale 5.0 --memory_mode original     --stage1_horizon_gamma 0.0 --stage2_horizon_gamma 0.0     --moe_fusion_mode add --moe_gate_coef 0.0     --dvgp_soft_selector --dvgp_temp 2.0 --dvgp_thresh_start 0.05 --dvgp_thresh_end 0.3     --checkpoint_prefix dp_pot_v58_dvgp     > dp_pot_v58_dvgp.log 2>&1
echo DONE
