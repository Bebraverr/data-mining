import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind': float_formatter})

d = pd.read_csv("F:/magic04.csv", usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),  header=None)


df = d.values
print("原始数据:\n", df)

col_num = np.size(df, axis=1)  # 获取数据列数
row_num = np.size(df, axis=0)  # 获取数据行数
mean_vector = np.mean(df, axis=0)
print("均值向量:\n", mean_vector)


centered_data_matrix = df - (1 * mean_vector)  # 计算Centered Data Matrix
print("中心数据矩阵:\n", centered_data_matrix, "\n")


t_centered_data_matrix = np.transpose(centered_data_matrix)  # 转置Centered Data Matrix
#计算中心数据矩阵内积
covariance_matrix_inner = (1 / row_num) * np.dot(t_centered_data_matrix, centered_data_matrix)
# 计算样本协方差矩阵

print("以中心数据矩阵列为内乘积的样本协方差矩阵：\n", covariance_matrix_inner, "\n")
# 计算中心数据点的和
def sum_of_centered_points():
    sum = np.zeros(shape=(col_num, col_num))
    for i in range(0, row_num):
        sum += np.dot(np.reshape(t_centered_data_matrix[:, i], (-1, 1)),
                      np.reshape(centered_data_matrix[i, :], (-1, col_num)))
    return sum


covariance_matrix_outer = (1 / row_num) * sum_of_centered_points()
print("样本协方差矩阵作为中心数据点之间的外积：\n", covariance_matrix_outer, "\n")

vector1 = np.array(centered_data_matrix[:, 1])
vector2 = np.array(centered_data_matrix[:, 2])
# 计算属性向量的单位向量
def unit_vector(vector):
    return vector / np.linalg.norm(vector)
# 计算属性向量之间的夹角
def angle_between(v1, v2):
    u_v1 = unit_vector(v1)
    u_v2 = unit_vector(v2)
    return np.arccos(np.clip(np.dot(u_v1, u_v2), -1.0, 1.0))

correlation = math.cos(angle_between(vector1, vector2))  # 计算各属性间的相关性
print("属性1和2之间的相关性： %.5f" % correlation, "\n")

# 绘制属性散点图
df1 = pd.DataFrame(df[:, 1])  # 创建绘图数据框
p1 = plt.scatter(df[:, 1], df[:, 2], c="red", s=30)
plt.show()

# 绘制属性1的概率密度函数
df2 = pd.read_csv("F:/magic04.csv", usecols=(0, ), names=["1"])

print(len(df))
df1 = df2["1"]
mean = df1.mean()

std = df1.std()
print(std)
def normfun(x,mu,sigma):
    pdf = np.exp(-((x-mu)**2)/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    return pdf
x = np.arange(0,350)
# x数对应的概率密度
y = normfun(x, mean, std)
# 参数,颜色，线宽
plt.plot(x,y, color='g',linewidth = 3)
#数据，数组，颜色，颜色深浅，组宽，显示频率
plt.hist(df1, bins =35, color = 'r',alpha=0.5,rwidth= 0.9, normed=True)
plt.title('distribution')
plt.xlabel('data')
plt.ylabel('Probability')
plt.show()

#求方差以及最大最小
i = 0
list = []
while(i<10):
    a = np.var(d[i])
    i = i+1
    list.append(a)
print("对应属性的方差是：", list)
print("属性", list.index(max(list)) + 1, "具有最大方差，且最大方差是：", max(list))
print("属性", list.index(min(list)) + 1, "具有最小方差，且最小方差是：", min(list))

# 计算协方差矩阵以及最大最小
a = np.cov(df, rowvar=False)  # 计算协方差矩阵
print("协方差矩阵是：\n", a)
max_cov = np.max(a)  # 找出协方差矩阵中的最大值
min_cov = np.min(a)  # 找出协方差矩阵中的最小值

# 利用for循环找出最大最小值的索引
for i in range(0, 10):
    for j in range(0, 10):
        if a[i, j] == max_cov:
            max_cov1 = i+1
            max_cov2 = j+1

for i in range(0, 10):
    for j in range(0, 10):
        if a[i, j] == min_cov:
            min_cov1 = i+1
            min_cov2 = j+1

print("最大斜方差是： %.3f (在属性 %d和属性 %d 之间)" % (max_cov, max_cov1, max_cov2))
print("最小斜方差是： %.3f (在属性 %d和属性 %d 之间)" % (min_cov, min_cov1, min_cov2))
