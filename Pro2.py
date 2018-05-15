from sklearn import preprocessing
import numpy as np
data = np.loadtxt("F:/iris.txt", delimiter=",", usecols=(0, 1, 2, 3), dtype=float)
target = np.loadtxt("iris.txt", delimiter=",", usecols=(4,), dtype=str)
# print(set(target))
print("核矩阵:")
a = np.dot(data, data.T)
kernel_matrix = a * a
print(kernel_matrix)
print("中心化")
mean = np.mean(kernel_matrix)
Ck = kernel_matrix-mean
print(Ck)
print("归一化")
Nk = preprocessing.scale(kernel_matrix)  # Numpy矩阵预处理类库
print(Nk)

