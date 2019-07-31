import numpy as np

# print(np.__version__)
# data = [[3.1, 5, 432, 5], [3, 4, 5, 6]]
# # print(data)
# npdata = np.array(data)
# print(npdata.shape, npdata.dtype)
#
# data2 = [["3", 5, 432, 5], [3, 4, 5, 6]]
# # print(data)
# npdata2 = np.array(data2)
# print(npdata2.shape, npdata2.dtype)

# print(np.arange(22))

# arr = np.arange(10)
# arr[:3] = 4
# print(arr)

# names = np.array(["Tony", "Jack", "Robin"])
# print((names == "Tony") | (names == "Robin"))

# arr = np.empty((8, 4))
# print(arr)
# for i in range(8):
#     arr[i] = i
# print(arr[[3, 7, 5, 3]])

# arr = np.arange(30).reshape((5, 2, 3))
# print(arr)
# print('        ')
# print(arr[[1, 3, 4, 2, 3]])

# print(arr.transpose(1, 2, 0).shape)
# print(arr.T.shape)

# x_arr = np.array([1, 2, 3, 5])
# y_arr = np.array([3, 33, 2, 7])
# condion = np.array([True, False, False, True])
#
# result = [(x if condion else y) for x, y, condion in zip(x_arr, y_arr, condion)]
# print(result)
# print(np.where(condion, x_arr, y_arr))

# arr = np.arange(24).reshape(4, 6)
# print(arr)
# print('  ')
# print(np.where(arr > 0, 1, 0))
# print('std', arr.std())
# print(arr.sum(0))  # 0轴
# print(arr.mean(axis=1))
# arr.sort(axis=0)
# print(arr.mean(axis=0))

# arr = np.arange(20).reshape(4, 5)
# print(arr)
# np.savetxt('test.txt', arr)
# print(np.loadtxt('test.txt'))

# arr = np.array([[1, 0, 0],
#                 [0, 2, 0],
#                 [0, 0, 4]])
# print(arr)
# print(np.linalg.eig(arr))  # 特征值 特征向量
# print(np.linalg.inv(arr))  # 矩阵的逆








