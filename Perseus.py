from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def plotXvY(X, Y, xlabel='Compute Warps In Flight'):
    # 绘制散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, s=50, alpha=0.5)  # s表示点的大小，alpha表示透明度
    plt.ylabel('Throughput')
    plt.xlabel(xlabel)
    plt.title('Tv' + xlabel[0])
    plt.grid(True)
    plt.show()

def Perseus(metric='Compute Warps In Flight', file_path='sample.csv'):
    # 1. Outlier detection
    # 预处理，通过 DBSCAN + PCA 找出并排除异常值
    data = pd.read_csv(file_path)
    Y = data['Throughput']
    X = data[metric]
    # 合并 X 和 Y 成为形如 <Xi, Yi> 数对的数组
    # pairs_array = list(zip(X, Y))
    # # 将列表转换为NumPy数组
    # pairs_array = np.array(pairs_array)
    pairs_array = np.column_stack((X, Y)).astype(float)
    print("初始的点数量：", len(pairs_array))
    # plotXvY(X, Y, metric)

    # # 使用 PCA 对数据进行转换
    # pca = PCA(n_components=2)  # 假设你想要降到2维
    # transformed_pairs = pca.fit_transform(pairs_array)
    # # 获取第一个主成分（最重要的方向）
    # main_component = pca.components_[0]
    # # 设置一个阈值来判断异常值（可以根据实际情况调整）
    # threshold = 2.0  # 可调整阈值
    # # 计算每个点到主要方向的投影
    # projection = np.dot(pairs_array - np.mean(pairs_array, axis=0), main_component)
    #
    # # 根据阈值判断异常值的索引
    # outlier_indices = np.where(np.abs(projection) > threshold)[0]
    #
    # # 惩罚垂直于倾斜方向的异常值
    # for idx in outlier_indices:
    #     # 计算异常点到主要方向的投影
    #     proj_val = np.dot(pairs_array[idx] - np.mean(pairs_array, axis=0), main_component)
    #     # 通过减去异常点在主要方向上的投影，来惩罚垂直于主要方向的异常值
    #     pairs_array[idx] -= proj_val * main_component

    # 初始化DBSCAN
    dbscan = DBSCAN(eps=0.1, min_samples=100)
    # 使用DBSCAN进行聚类
    clusters = dbscan.fit_predict(pairs_array)
    # 找到有效簇的标签（不包括噪声点，标签为 -1）
    valid_labels = [label for label in np.unique(clusters) if label != -1]
    # 找到有效簇中的点
    valid_points = []
    for label in valid_labels:
        indices = np.where(clusters == label)[0]
        valid_points.extend(pairs_array[i] for i in indices)
    # 将 valid_points 转换为 NumPy 数组（可选）
    valid_points = np.array(valid_points)
    # 在这里，valid_points 就是保留的有效簇中的数据
    print("经过DBSCAN后, 保留的有效簇中的点数量：", len(valid_points))
    newX = valid_points[:, 0]
    newY = valid_points[:, 1]
    # plotXvY(newX, newY)

    # 2. Regression Model
    # 选择多项式回归
    # 使用 PolynomialFeatures 创建多项式特征
    poly = PolynomialFeatures(degree=2)  # 设置多项式的次数
    X_poly = poly.fit_transform(newX.reshape(-1, 1))

    # 使用线性回归拟合多项式特征
    model = LinearRegression()
    model.fit(X_poly, newY)

    # 进行预测
    Y_predicted = model.predict(X_poly)
    # print(predicted)
    # plotXvY(newX, Y_predicted)

    # 3. Identifying Fail-Slow Event
    # 区分慢速条目
    # 获取置信区间（95% 和 99.9%）
    confidence_95 = 1.96 * np.std(Y_predicted)  # 95% 置信区间的计算方法，1.96 是 95% 置信水平对应的 Z 值
    confidence_999 = 3.29 * np.std(Y_predicted)  # 99.9% 置信区间的计算方法，3.29 是 99.9% 置信水平对应的 Z 值

    # 打印置信区间的"下"限：因为在本实例中y轴是Throughput 所以慢速对应的是下界
    upper_bound_95 = Y_predicted - confidence_95
    # plotXvY(newX, upper_bound_95)
    upper_bound_999 = Y_predicted - confidence_999
    # plotXvY(newX, upper_bound_999)

    # plt.figure(figsize=(8, 6))
    # plt.scatter(newX, newY, s=50, alpha=0.5)  # s表示点的大小，alpha表示透明度
    # plt.ylabel('Throughput')
    # plt.scatter(newX, Y_predicted, s=50, alpha=0.5, color='red')  # s表示点的大小，alpha表示透明度
    # plt.scatter(newX, upper_bound_95, s=50, alpha=0.5, color='green')  # s表示点的大小，alpha表示透明度
    # plt.xlabel(metric)
    # plt.title('Tv' + metric[0])
    # plt.grid(True)
    # plt.show()
    # print(len(upper_bound_999), len(Y), len(X))

    # 指定滑动窗口大小为600 即一分钟 滑动窗口内的一半条目比阈值慢
    window_size = 600


    # 根据需要调整数据长度，以保证可以整除窗口大小
    data_length = len(newX)
    num_windows = data_length // window_size
    print("window_size = " + str(num_windows))
    for i in range(num_windows):
        start_index = i * window_size
        end_index = min((i + 1) * window_size, data_length)

        # 获取当前窗口的数据
        current_window_X = newX[start_index:end_index]
        current_window_Y = newY[start_index:end_index]

        current_window_upper_bound_95 = upper_bound_95[start_index:end_index]

        # 计算当前窗口中 Y 值小于 upper_bound_999 的情况数量
        num_values_below_upper_bound = sum(current_window_Y < current_window_upper_bound_95)

        # print("window" + str(i) + ": " + str(num_values_below_upper_bound))
        # 判断条件并输出窗口序号
        if num_values_below_upper_bound >= window_size / 2:
            print("满足条件的窗口序号：", i)

    # 4. Risk Score
    # 风险等级由慢速持续时间和严重程度联合决定
    # 也就是说人为设置一个风险评分机制
    # Risk Score = Nextreme × 100 + Nhigh × 25 + Nmoderate × 10 + Nlow × 5 + Nminor × 1

if __name__ == '__main__':
    Perseus()