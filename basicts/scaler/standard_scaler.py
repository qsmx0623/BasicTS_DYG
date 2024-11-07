import json
import torch
import numpy as np
# from .base_scaler import BaseScaler
from basicts.scaler.base_scaler import BaseScaler

class StandardScaler(BaseScaler):
    """
    StandardScaler标准化数据集的特征，使其具有零均值和单位方差。
    这对于许多机器学习算法来说是一个常见的预处理步骤，以确保每个特征对模型的贡献均等。

    属性:
        mean (np.ndarray): 用于标准化的训练数据的均值。
            如果norm_each_channel为True，这是一个包含每个通道均值的数组。否则，它是一个标量。
        std (np.ndarray): 用于标准化的训练数据的标准差。
            如果norm_each_channel为True，这是一个包含每个通道标准差的数组。否则，它是一个标量。
        target_channel (int): 应用标准化的特定通道（特征）。
            默认情况下，它被设置为0，表示第一个通道。
    """

    def __init__(self, dataset_name: str, train_ratio: float, norm_each_channel: bool, rescale: bool):
        """
        通过加载数据集并将标准化器拟合到训练数据来初始化StandardScaler。

        该标准化器从训练数据计算均值和标准差，然后在transform操作期间用于标准化数据。

        参数:
            dataset_name (str): 用于加载数据的数据集名称。
            train_ratio (float): 用于训练的数据集比例。标准化器在这部分数据上进行拟合。
            norm_each_channel (bool): 指示是否分别标准化每个通道的标志。
                如果为True，均值和标准差是独立计算的。
            rescale (bool): 指示标准化后是否应用重新缩放的标志。此标志包含在内以与基类保持一致，但在标准化中不直接使用。
        """

        super().__init__(dataset_name, train_ratio, norm_each_channel, rescale)
        self.target_channel = 0  # 假设对第一个通道进行标准化

        # 加载数据集描述和数据
        description_file_path = f'/home/home_new/qsmx/pycodes/BasicTS/datasets/{dataset_name}/desc.json'
        with open(description_file_path, 'r') as f:
            description = json.load(f)
        data_file_path = f'/home/home_new/qsmx/pycodes/BasicTS/datasets/{dataset_name}/data.dat'
        data = np.memmap(data_file_path, dtype='float32', mode='r', shape=tuple(description['shape']))
        # print("原始数据:", data)

        # 根据train_ratio将数据拆分为训练集
        train_size = int(len(data) * train_ratio)
        train_data = data[:train_size, :, self.target_channel].copy()
        # print("训练数据:", train_data)
        # print("训练数据形状:", train_data.shape)

        # 计算均值和标准差
        if norm_each_channel: #这个通道指的是节点特征的通道
            self.mean = np.mean(train_data, axis=0, keepdims=True)
            self.std = np.std(train_data, axis=0, keepdims=True)
            self.std[self.std == 0] = 1.0  # 防止标准差为零导致除以零的错误
        else:
            self.mean = np.mean(train_data)
            self.std = np.std(train_data)
            if self.std == 0:
                self.std = 1.0  # 防止标准差为零导致除以零的错误

        # print(self.mean)
        # print(self.std)
        self.mean, self.std = torch.tensor(self.mean), torch.tensor(self.std)

    def transform(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        对输入数据应用标准化。

        该方法使用从训练数据计算的均值和标准差对输入数据进行标准化。
        标准化仅应用于指定的target_channel。

        参数:
            input_data (torch.Tensor): 要标准化的输入数据。

        返回:
            torch.Tensor: 与输入形状相同的标准化数据。
        """

        mean = self.mean.to(input_data.device)
        std = self.std.to(input_data.device)
        input_data[..., self.target_channel] = (input_data[..., self.target_channel] - mean) / std

        return input_data

    def inverse_transform(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        逆转标准化以恢复原始数据尺度。

        该方法使用从训练数据计算的均值和标准差将标准化数据转换回其原始尺度。
        这对于解释模型输出或在原始数据尺度上进行进一步分析非常有用。

        参数:
            input_data (torch.Tensor): 要转换回的标准化数据。

        返回:
            torch.Tensor: 转换回原始尺度的数据。
        """

        mean = self.mean.to(input_data.device)
        std = self.std.to(input_data.device)
        # 克隆输入数据以防止就地修改（PyTorch中不允许）
        input_data = input_data.clone()
        input_data[..., self.target_channel] = input_data[..., self.target_channel] * std + mean
        return input_data