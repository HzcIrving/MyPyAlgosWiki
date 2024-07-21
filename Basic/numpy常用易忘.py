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
