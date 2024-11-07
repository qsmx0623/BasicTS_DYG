#!/bin/bash
# 四线程
# 激活环境
#source /home/home_new/chensf/anaconda3/etc/profile.d/conda.sh
#conda activate BasicTS1

# 实验路径
EXP_DIR="benchmarks/EXP_DYG_03_SUB2"

# "AGCRN" "BigTS" "STDMAE" "D2STGNN" "DCRNN" "StemGNN" "DGCRN" "MegaCRN" "MTGNN" "STGCN" "GWNet" "GTS" "STAEformer" "STID" "STEP" "STNorm" "STGODE" "STWave"
# "D2STGCN" 因为cuda内存不足问题失败
# "STDMAE"和"STEP"为预训练模型，没有比较的必要

# 定义所有的baselines
baselines=("AGCRN" "BigTS" "DCRNN" "StemGNN" "DGCRN" "MegaCRN" "MTGNN" "STGCN" "GWNet" "GTS" "STAEformer" "STID" "STNorm" "STGODE" "STWave")

# 并行计数器
parallel_jobs=4
job_count=0

# 循环运行每个baseline
for baseline in "${baselines[@]}"
do
    echo "************************************************************************************************************"
    echo "Running baseline: $baseline"

    # 运行模型一次
    for i in {1..1}
    do
        echo "Run #$i for baseline: $baseline"
        python experiments/train.py -c "$EXP_DIR/configs/$baseline.py" --gpus '6' &
        ((job_count++))

        # 每四个线程等待一次，确保最多四个并行任务
        if (( job_count % parallel_jobs == 0 )); then
            wait
        fi
    done
done

# 等待所有后台任务完成
wait

# tmux是一个命令，目的是做一个稳定的实验文件夹，防止实验断掉。
# tensorbord 学一下。