#!/bin/bash

# 激活环境
#source /home/home_new/chensf/anaconda3/etc/profile.d/conda.sh
#conda activate BasicTS1

# 实验路径
EXP_DIR="/home/home_new/qsmx/pycodes/BasicTS/benchmarks/EXP_DYG_03"

# 定义所有的baselines
baselines=("HI" "DeepAR" "PatchTST" "Pyraformer" "Informer" "DSFormer" "Autoformer" "DCRNN" "WaveNet" "StemGNN" "DGCRN" "MegaCRN" "MTGNN" "STGCN" "GWNet" "Gate")

# 循环运行每个baseline
for baseline in "${baselines[@]}"
do
    echo "************************************************************************************************************"
    echo "Running baseline: $baseline"

    # 运行模型十遍
    for i in {1..10}
    do
        echo "Run #$i for baseline: $baseline"
        python experiments/train.py -c "$EXP_DIR/configs/$baseline.py" --gpus '0'
    done
done

# 退出conda
#conda deactivate