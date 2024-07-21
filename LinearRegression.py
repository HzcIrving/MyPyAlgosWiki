""" 
Ridge & LASSO 回归  
------------------------------------- 
- y:响应变量 
- x1~xn: 解释变量
- b0~bn: 模型参数  
- ε 误差项   

y = b0 + b1x1 + b2x2 + ... +bnxn + ε  
------------------------------------
S1
    构建设计矩阵  
S2  
    求解参数
        - 最小二乘 
S3 
    评估模型拟合度和预测能力
------------------------------------- 
Ridge回归 
> L2 防止过拟合   
LASSO 
> L1正则化项   
""" 

import numpy as np 

class LinearRegression:
    def __init__(self) -> None:
        pass 
    
    def fit(self, X, y): 
        pass  

if __name__ == "__main__":
    pass  