""" 
逻辑回归 
--------------------------------
0正常，1故障 
- 和n个数值特征有关 
- 模拟逻辑回归的单步SGD  
    Θ_j^t = Θ_j^(t-1) - α*1/m*sum_{i=1}^m * (g(x_i) - y_i) * x_i^j 
    
- g(x) : sigmoid 
    g(x) = 1 / (1+e^(-x))
--------------------------------
输入1: 
    >  参数矩阵(theta)  1x3 
    >  X_train: 训练集  10x3 
    >  Y_train: Label 10x1  
""" 
import numpy as np 

class LogisticReg: 
    def __init__(self) -> None:
        pass 
    
    def solve(self, lr, theta, X, y): 
        self.theta = theta 
        self.X = X 
        self.y = y 
        self.lr = lr # alpha 
        
        self.SGD() 
        
        return self.theta 
        
    
    def SGD(self): 
        # self.theta = self.theta - self.lr * np.mean(self.sigmoid(self.X, self.theta) @ self.X, axis=0)    
        feature_dim = self.X.shape[1]    
        for i in range(feature_dim): 
            tmp_res = self.sigmoid(self.X, self.theta) @ self.X[:, i] 
            tmp_res = np.sum(tmp_res) / self.X.shape[0] * self.lr  
            tmp_res = self.theta[i] - tmp_res 
            self.theta[i] = tmp_res.round(3)
        
        
    def sigmoid(self,X,Y):
        """ 
        X: 特征向量 
        Y: 神经网络参数 
        """
        gx = np.exp(-1*X @ Y.T) 
        gx = 1 / (1 + gx) 
        return gx 
