import os
import tensorflow as tf
import matplotlib.pyplot as plt

def load_event_file(event_file):
    """
    解析 TensorBoard 的 event 文件，返回所有标量数据。
    """
    scalar_data = {}

    # 解析 event 文件
    for event in tf.compat.v1.train.summary_iterator(event_file):
        for value in event.summary.value:
            if value.tag not in scalar_data:
                scalar_data[value.tag] = []
            scalar_data[value.tag].append((event.step, value.simple_value))

    return scalar_data

def plot_scalar_data(scalar_data):
    """
    使用 Matplotlib 对标量数据进行可视化。
    """
    plt.figure(figsize=(10, 5))

    for tag, data in scalar_data.items():
        steps, values = zip(*data)
        plt.plot(steps, values, label=tag)

    plt.xlabel('Steps')
    plt.ylabel('Values')
    plt.title('TensorBoard Scalars')
    plt.legend()
    plt.grid(True)
    plt.show()

# 指定 event 文件的路径
event_file = '/home/home_new/qsmx/pycodes/BasicTS/checkpoints/AGCRN/DYG_data_3_sub2-3_100_12_12/b421ee749e7d419c8b69d5065a64864c/tensorboard/events.out.tfevents.1730862636.ps.172773.0'

# 解析 event 文件
scalar_data = load_event_file(event_file)

# 可视化标量数据
plot_scalar_data(scalar_data)
