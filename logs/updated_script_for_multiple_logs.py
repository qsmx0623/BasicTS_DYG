import os
import csv

# 该脚本用于批量处理模型训练的日志文件，自动提取关键信息，方便对多次实验的结果进行比较和分析。
# test_metrics只会记录一次实验的结果

def extract_model_name(log_content):
    for line in log_content:
        if "Checkpoint" in line and "saved" in line:
            # This assumes the model name is always in the format described and is the second element in the path
            model_name = line.split('/')[3]
            return model_name
    return "Unknown"

def process_log_file(log_file_path):
    with open(log_file_path, 'r') as file:
        log_content = file.readlines()

    epoch_lines = [line for line in log_content if "Epoch" in line]
    metric_lines = [line for line in log_content if "test_MAE" in line]

    metrics_by_epoch = {}

    for i, epoch_line in enumerate(epoch_lines):
        epoch_number = int(epoch_line.split('Epoch')[1].split('/')[0].strip())
        metric_line = metric_lines[i]
        test_mae = float(metric_line.split('test_MAE: ')[1].split(',')[0])
        test_rmse = float(metric_line.split('test_RMSE: ')[1].split(',')[0])
        test_mape = float(metric_line.split('test_MAPE: ')[1].split(',')[0])
        test_wape = float(metric_line.split('test_WAPE: ')[1].split(',')[0])
        test_mse = float(metric_line.split('test_MSE: ')[1].split(']')[0])
        
        metrics_by_epoch[epoch_number] = (test_mae, test_rmse, test_mape, test_wape, test_mse)

    best_epoch = min(metrics_by_epoch, key=lambda epoch: metrics_by_epoch[epoch][0])
    best_epoch_metrics = metrics_by_epoch[best_epoch]
    model_name = extract_model_name(log_content)

    return model_name, best_epoch, best_epoch_metrics

def main(logs_directory):
    results = []

    for file_name in os.listdir(logs_directory):
        if file_name.endswith('.log'):
            print(f'Processing File:{file_name}')
            log_file_path = os.path.join(logs_directory, file_name)
            model_name, best_epoch, metrics = process_log_file(log_file_path)
            results.append([file_name, model_name, best_epoch] + list(metrics))

    # Exporting results to a CSV file
    with open('log_analysis_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['File Name', 'Model Name', 'Best Epoch', 'Test MAE', 'Test RMSE', 'Test MAPE', 'Test WAPE', 'Test MSE'])
        writer.writerows(results)
    print(f'Results have been saved to log_analysis_results.csv')

if __name__ == '__main__':
    #logs-directory is the current directory,so the log.txt documents should be downloaded here.
    logs_directory = './'
    main(logs_directory)
