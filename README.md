# MyPyAlgosWiki
备考备考备考 

## 1. IO代码段 

- 单个
    e.g. `a = int(input()) `

- 单行List 
    e.g. `a = list(map(int, input().strip().split()))` 
        若 in: [[1 2 3],...     strip('[]') 
        若 in: 1,2,3, ...  split(',') 
- 多行List 
    e.g. `a = [list(map(int, input().strip().split())) for _ in range(n)]`  
    若不规定行数 
    ```
    try: 
        while True: 
            a = list(map(int, input().strip().split())) 
    except: 
        ...  
    ```


- 打印单行 
    ```
    [[1.16666667 1.46666667]
    [7.33333333 9.        ]] 
    保留两位小数 || 单行打印 >>>  7.33,9.00;1.17,1.47 
    ```
    
    ```
    print(";".join([",".join(["{:.2f}".format(i) for i in pts]) for pts in centers]))
    ```  

## 2. 保留小数问题  
### 2.1 保留两位小数且不进位  
```
def get_floor(val):
    return "{:.2f}".format(np.floor(val*100)/100)
```  

### 2.2 保留两位小数且进位  
```
def get_ceil(val): 
    return "{:.2f}".format(np.ceil(val*100)/100)
```

## 3. 易错点 

### 3.1 无穷大 
- 正无穷 float("inf") 
- 负无穷 float("-inf")  


### 3.2 快速记忆Numpy的axis 
```
import numpy as np

# 创建一个4x5的二维数组
arr = np.array([[1, 2, 3, 4, 5],
                 [6, 7, 8, 9, 10],
                 [11, 12, 13, 14, 15],
                 [16, 17, 18, 19, 20]])

# 计算沿axis=0（行）的均值
mean_axis0 = np.mean(arr, axis=0)
print("沿axis=0的均值：", mean_axis0) # 沿着行 >>> [ 8.5  9.5 10.5 11.5 12.5] 输出维数等于列数 

# 计算沿axis=1（列）的均值
mean_axis1 = np.mean(arr, axis=1)
print("沿axis=1的均值：", mean_axis1) # 沿着列 >>> [ 3.  8. 13. 18.] 输出维数等于行数   

# 计算沿axis=None（全部）的均值
mean_none = np.mean(arr, axis=None)
print("沿axis=None的均值：", mean_none) # 沿着全部 >>> 10.5 输出维数为1
``` 


### 3.3 基于label的数据筛选思想 (`KMeans.py`)
``` 
labels = np.argmax(dists, axis=1) # 确定每个数据点的簇标签 

# 3. 重新计算簇中心 
centers = np.zeros((self.k,2))
for i in range(self.k):
    centers[i] = self.data[labels==i].mean(axis=0)
```

### 3.4 排序->统计标签->投票(`KNN.py`)
```
# 获取topK个最近邻居   
# 排序，并返回indices 
k_indices = np.argsort(dists)[:self.topk] 

# 获取k个最近邻居的标签 
k_nearest_labels = [self.label[i] for i in k_indices]  
print(k_nearest_labels)
    
# 投票   
# Counter是Python标准库collections模块中的一种容器，用于计数哈希对象。它继承自dict类，所以你可以把它当作一个字典，其中的键是元素，值是元素出现的 次数。
# 参数1表示只返回一个结果。
res = Counter(k_nearest_labels).most_common(1)  
print(res) -> [(0, 4)] 
``` 

### 3.5 sigmoid
$sigmoid(x) = \frac{1}{1+e^{-x}}$ 
```
# numpy实现 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

### 3.6 softmax
$softmax(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n}e^{x_j}}$ 
``` 
# numpy实现 
import numpy as np
def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True) # keepdims很重要
    return exp_x / sum_exp_x
# 创建一个NumPy数组
x = np.array([[-1, 0, 1],[-1, 0, 1]])
# 计算Softmax函数的结果
y = softmax(x)
print(y)  
```

### 3.7 卷积padding `ConvOp.py`
直接分析案例 

`X = np.pad(X, ((0,0),(pad_size, pad_size),(pad_size, pad_size)), mode)` 
- 对X进行padding, X Shape: (CHW) 
- C维不做padding 
- H维padding大小为pad_size 
- W维padding大小为pad_size 
- mode: 'constant' or 'edge' 
    - 'constant': 填充0 
    - 'edge': 填充边界值 