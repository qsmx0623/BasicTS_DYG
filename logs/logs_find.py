import os
import re
import pandas as pd

# 日志文件所在目录路径
log_dir = "/Users/shifanchen/Documents/WorkSpace/logs/logs_obtain/logs_obtain_DYG"  # 修改为你的日志目录路径
output_file = os.path.join(log_dir, "detailed_results.xlsx")

# 定义存储数据的结构
results = {
    "Log File": [],
    "Model Name": [],
    "Horizon": [],
    "MAE": [],
    "RMSE": [],
    "WAPE": [],
    "MSE": [],
    "Time (s)": [],
    "Loss": []
}

# 正则表达式匹配模型名称
model_pattern = re.compile(r"Loading Checkpoint from 'checkpoints/(?P<model>[^/]+)/")

# 匹配测试结果（包括 time 和 loss）
test_result_pattern = re.compile(
    r"Result <test>: \[test/time: (?P<time>[0-9.]+) \(s\), "
    r"test/loss: (?P<loss>[0-9.]+), "
    r"test/MAE: (?P<mae>[0-9.]+), "
    r"test/RMSE: (?P<rmse>[0-9.]+), "
    r"test/WAPE: (?P<wape>[0-9.]+), "
    r"test/MSE: (?P<mse>[0-9.]+)\]"
)

# 匹配 horizon 的详细测试结果
horizon_pattern = re.compile(
    r"Evaluate best model on test data for horizon (?P<horizon>\d+), "
    r"Test MAE: (?P<mae>[0-9.]+), Test RMSE: (?P<rmse>[0-9.]+), "
    r"Test WAPE: (?P<wape>[0-9.]+), Test MSE: (?P<mse>[0-9.]+)"
)

# 遍历日志目录中的所有日志文件
for log_file in os.listdir(log_dir):
    if log_file.endswith(".log"):
        log_path = os.path.join(log_dir, log_file)
        print(f"Processing log file: {log_file}")

        # 初始化变量
        model_name = None
        latest_test_result = None  # 存储最后一次测试结果
        latest_horizon_results = {}  # 存储每个 horizon 的最新结果

        # 打开并解析日志文件
        with open(log_path, 'r') as f:
            for line in f:
                # 匹配模型名称
                model_match = model_pattern.search(line)
                if model_match:
                    model_name = model_match.group("model")

                # 匹配测试结果（保留最后一次出现的测试结果）
                test_match = test_result_pattern.search(line)
                if test_match:
                    latest_test_result = {
                        "Horizon": "Summary",
                        "MAE": float(test_match.group("mae")),
                        "RMSE": float(test_match.group("rmse")),
                        "WAPE": float(test_match.group("wape")),
                        "MSE": float(test_match.group("mse")),
                        "Time (s)": float(test_match.group("time")),
                        "Loss": float(test_match.group("loss"))
                    }

                # 匹配每个 horizon 的详细测试结果（保留最后一次出现的结果）
                horizon_match = horizon_pattern.search(line)
                if horizon_match:
                    horizon = int(horizon_match.group("horizon"))
                    latest_horizon_results[horizon] = {
                        "Horizon": horizon,
                        "MAE": float(horizon_match.group("mae")),
                        "RMSE": float(horizon_match.group("rmse")),
                        "WAPE": float(horizon_match.group("wape")),
                        "MSE": float(horizon_match.group("mse")),
                        "Time (s)": None,  # 没有 time 信息
                        "Loss": None  # 没有 loss 信息
                    }

        # 如果没有找到模型名，用未知模型名标记
        if not model_name:
            model_name = "Unknown"

        # 将最后一次测试结果存储到结果字典中
        if latest_test_result:
            results["Log File"].append(log_file)
            results["Model Name"].append(model_name)
            results["Horizon"].append(latest_test_result["Horizon"])
            results["MAE"].append(latest_test_result["MAE"])
            results["RMSE"].append(latest_test_result["RMSE"])
            results["WAPE"].append(latest_test_result["WAPE"])
            results["MSE"].append(latest_test_result["MSE"])
            results["Time (s)"].append(latest_test_result["Time (s)"])
            results["Loss"].append(latest_test_result["Loss"])

        # 将每个 horizon 的最新结果存储到结果字典中
        for horizon, result in latest_horizon_results.items():
            results["Log File"].append(log_file)
            results["Model Name"].append(model_name)
            results["Horizon"].append(result["Horizon"])
            results["MAE"].append(result["MAE"])
            results["RMSE"].append(result["RMSE"])
            results["WAPE"].append(result["WAPE"])
            results["MSE"].append(result["MSE"])
            results["Time (s)"].append(result["Time (s)"])
            results["Loss"].append(result["Loss"])

# 转换为 DataFrame 并保存为 Excel 文件
df = pd.DataFrame(results)
df.to_excel(output_file, index=False)

print(f"所有日志信息已成功提取，并保存为 Excel 文件：{output_file}")
