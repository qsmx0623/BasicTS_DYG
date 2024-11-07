import os

# List of model names
model_names = [
    'Pyraformer', 'PatchTST', 'Informer', 'DSFormer', 'Autoformer',
    'DCRNN', 'WaveNet', 'StemGNN', 'DGCRN', 'MegaCRN',
    'MTGNN', 'STGCN', 'GWNet', 'Gate'
]

# Directory to save the .py files
output_directory = './output_scripts'

# Create the directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Define the content to write into each file
file_content = """# This is a script for the model
import numpy as np
import pandas as pd
import shutil
import os

# 设置文件路径
M = '/Users/shifanchen/Documents/WorkSpace/BasicTSupdate/checkpoints/DeepAR/IndPenisim_200_12_12/848bd1001dbdc020824f671fab1cd81e'  # 文件路径
file_path = f'{M}/test_results.npz'  # npz文件的真实路径
excel_file_path = f'{M}/test_results.xlsx'  # 生成的xlsx文件路径

# Load the npz file
data = np.load(file_path)

# Extract arrays 'prediction', 'target', and 'inputs'
prediction = data['prediction']
target = data['target']
inputs = data['inputs']

# Reshape each array to (34284, 5)
prediction_reshaped = prediction.reshape(-1, 8)
target_reshaped = target.reshape(-1, 8)
inputs_reshaped = inputs.reshape(-1, 8)

# Convert each to a pandas DataFrame
df_prediction = pd.DataFrame(prediction_reshaped)
df_target = pd.DataFrame(target_reshaped)
df_inputs = pd.DataFrame(inputs_reshaped)

# Create a writer object to write to Excel
with pd.ExcelWriter(excel_file_path) as writer:
    # Write each DataFrame to a different sheet
    df_prediction.to_excel(writer, sheet_name='Prediction', index=False)
    df_target.to_excel(writer, sheet_name='Target', index=False)
    df_inputs.to_excel(writer, sheet_name='Inputs', index=False)

print(f"Excel file saved to {excel_file_path}")

# 复制文件到指定目录
prediction_dir = '/Users/shifanchen/Documents/WorkSpace/BasicTSupdate/prediction'
os.makedirs(prediction_dir, exist_ok=True)  # 确保目录存在

# 获取当前文件名并生成新文件名
# 获取当前运行的脚本文件名
current_file_name = os.path.splitext(os.path.basename(__file__))[0]
copied_file_path = os.path.join(prediction_dir, f'test_results_{current_file_name}.xlsx')

# 拷贝文件
shutil.copy(excel_file_path, copied_file_path)
print(f"Copied Excel file to {copied_file_path}")
"""

# Create files for each model with the defined content
for model_name in model_names:
    file_name = f'{model_name}.py'
    file_path = os.path.join(output_directory, file_name)

    # Write the content to the file
    with open(file_path, 'w') as file:
        file.write(file_content)

    print(f"Created {file_name} with predefined content")