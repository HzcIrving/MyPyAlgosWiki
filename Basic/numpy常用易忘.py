""" 
https://blog.csdn.net/m0_74344139/article/details/134842295
"""

import numpy as np 

# 1. 初始化数组 -----------------------------------------
# 全0
a1 = np.zeros(shape=(3,3)) 
print(a1) 
# 空
a2 = np.empty(shape=(3,3))
print(a2) 


# 2. 排序，按idx返回 ------------------------------------- 
dists = np.array([0.7, 2.7, 3.6, 4.7, 1.5])
k_indices = np.argsort(dists, axis=0)
print(k_indices)   # [0 4 1 2 3] 

X = np.array([[0,5], [1,4], [3,2]])  
print(X.shape)
k_indices = np.argsort(X, axis=1)  
# [[0 1]
#  [0 1]
#  [1 0]]
print(k_indices) 

# 3. axis ----------------------------------------
import numpy as np

# 创建一个4x5的二维数组
arr = np.array([[1, 2, 3, 4, 5],
                 [6, 7, 8, 9, 10],
                 [11, 12, 13, 14, 15],
                 [16, 17, 18, 19, 20]])

# 计算沿axis=0（行）的均值
mean_axis0 = np.mean(arr, axis=0)
print("沿axis=0的均值：", mean_axis0)   # [ 8.5  9.5 10.5 11.5 12.5] 

# 计算沿axis=1（列）的均值
mean_axis1 = np.mean(arr, axis=1)
print("沿axis=1的均值：", mean_axis1)   # [ 3.  8. 13. 18.] 

# 计算沿axis=None（全部）的均值
mean_none = np.mean(arr, axis=None)
print("沿axis=None的均值：", mean_none)   # 10.5


# 4. numpy实现矩阵乘法 -----------------------------------------------
""" 
在NumPy中，矩阵乘法可以通过几种不同的方式实现，主要区别在于性能和可读性。以下是一些常见的实现矩阵乘法的NumPy函数：
""" 
# 4.1 dot 
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = np.dot(a, b) 
print("dot numpy:", result) 

# 4.2 matmul函数：这也是一个矩阵乘法函数，与dot函数非常相似，但它可以用于多维数组。 
import numpy as np
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
result = np.matmul(a, b.T)   # 2x2 x 2x1 = 2x1 
print("matmul numpy:", result) 

# 4.3 @运算符：这是Python 3.5引入的矩阵乘法运算符，可以直接用于两个数组。 
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
result = a @ b.T  # 2x2 x 2x1 = 2x1 
print("@ numpy:", result)  

# 4.4 elementwise加减乘除 
a = np.array([[1, 2], [3, 4]]) 
b = np.array([[5, 6], [7, 8]])
result = a * b  # elementwise  
print("elementwise numpy:", result)  
# 也可以用Multiply 
result = np.multiply(a , b)  # elementwise   
print(result)


# 5. 常用算子 ------------------------------- 
""" 
5.1 Softmax
"""
import numpy as np
def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    return exp_x / sum_exp_x
# 创建一个NumPy数组
x = np.array([[-1, 0, 1],[-1, 0, 1]])
# 计算Softmax函数的结果
y = softmax(x)
print(y)  