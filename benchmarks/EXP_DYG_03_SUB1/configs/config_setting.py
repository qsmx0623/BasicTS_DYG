import os

#这个修改参数设置的脚本要慎重，不如手动

# 模型名称后缀
model_suffix = "_DYG_03"

# 新的训练参数设置
training_params = {
    'DATA_NAME': '"DYG_data_3_sub1-3"',
    'INPUT_LEN': 12,
    'OUTPUT_LEN': 12,
    'NUM_NODES': 5,
    'NUM_EPOCHS': 200,
    'CFG.GPU_NUM': 1,
    'CFG.TRAIN.NUM_EPOCHS': 100,
    'CFG.TRAIN.DATA.BATCH_SIZE': 64,
    'CFG.VAL.DATA.BATCH_SIZE': 64,
    'CFG.TEST.DATA.BATCH_SIZE': 64,
    'CFG.EVAL.HORIZONS': [3, 6, 12]
}


# 获取当前工作目录
current_dir = os.getcwd()

# 遍历当前目录下的所有Python文件
def process_python_files(directory):
    for file_name in os.listdir(directory):
        if file_name.endswith('.py'):  # 处理Python文件
            file_path = os.path.join(directory, file_name)
            try:
                update_training_params(file_path)
            except Exception as e:
                print(f"An error occurred while processing {file_path}: {e}")

# 更新单个Python文件中的训练参数
def update_training_params(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            for param_key, param_value in training_params.items():
                if f'{param_key} =' in line:
                    file.write(f'{param_key} = {param_value}\n')
                    break
            else:
                file.write(line)

# 指定当前目录作为要处理的目录
directory = current_dir

# 执行处理
process_python_files(directory)

print("训练参数已在所有配置文件中更新！")