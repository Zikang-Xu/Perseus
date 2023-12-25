import os.path
import math
import numpy as np
import pandas as pd
import warnings
from scipy.stats import pearsonr
def pearson_analysis(file_path='sample.csv'):
    # sample.csv数据来自于单个GPU的性能数据
    # 预处理：计算并写回Throughput
    # Throughput是slow分析的核心指标
    if not os.path.exists(file_path):
        print(f"{file_path} not exist!")
        return None
    data = pd.read_csv(file_path)
    data['Throughput'] = data['DRAM Read Throughput'] + data['DRAM Write Throughput']
    data.to_csv(file_path, index=False)
    print(f"Throughput data update and write back to {file_path}!")

    # 计算：Throughput与其它指标的Pearson相关系数
    columns_to_compare = [col for col in data.columns if col != 'File' and 'Throughput' not in col]
    correlation_values = {}
    # 捕获警告信息
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 将警告转换为异常
        try:
            for column in columns_to_compare:
                correlation = data['Throughput'].corr(data[column])
                correlation_values[column] = correlation
        except Warning as w:
            print(f"Warning: {w} when calculating column: {column}")
            # del correlation_values[column]
    # 按照Pearson相关系数的值从大到小排序并输出
    sorted_correlation = sorted(correlation_values.items(), key=lambda x: x[1], reverse=True)
    print("Throughput 与其他列的Pearson相关系数（从大到小）：")
    for column, corr in sorted_correlation:
        print(f"{column}: {corr}")
    # 返回相关性最高的指标
    print(sorted_correlation[0][0] + " is the most correlative metric")
    return sorted_correlation[0][0]

if __name__ == '__main__':
    pearson_analysis()

