import numpy as np 


class Transformer:
    
    def __init__(self, debug=True): 
        if not debug:
            self.get_inputs()
    
    def get_inputs(self):  
        # 特征维数 d 
        self.feat_dim = int(input())  
        # 序列长度 n 
        self.seq_len = int(input()) 
        
        # X, Q, K, V权重矩阵    
        # X: d * n 
        # WQ: d * d 
        # WK: d * d 
        # WV: d * d 
        self.X, self.W_Q, self.W_K, self.W_V = self.get_matrix()   
        
        
    def solve(self): 
        self.X = np.array(self.X) 
        self.W_Q = np.array(self.W_Q) 
        self.W_K = np.array(self.W_K) 
        self.W_V = np.array(self.W_V) 
        
        # Attn 
        # res = Q * K^T 
        # Softmax(QK^T/sqrt(d))*V  
        # Softmax:           
        Query = (self.W_Q @ self.X).round(2)  
        Key = (self.W_K @ self.X).round(2) 
        Value = (self.W_V@self.X).round(2) 
        
        res = self.softmax((Query@Key.T)/np.sqrt(self.feat_dim))@Value 
        res = res.T  
        res = res.round(2)
        
        return res  
        
    def get_matrix(self):  
        def get_mat(): 
            res = []
            for _ in range(self.feat_dim):
                res.append(list(map(float, input().strip('[]').split())))    
            return res 
        
        X = get_mat() 
        W_Q = get_mat()   
        W_K = get_mat() 
        W_V = get_mat() 
        
        return [X,W_Q,W_K,W_V]

        
    @staticmethod
    def softmax(X): 
        """ 
        X: (Bs, 2)
        Softmax(x) = e^(x) / \sum_{i=1}^n e^(x_i) 
        """  
        return (np.exp(X) / np.sum(np.exp(X), axis=0)).round(2)   
    
    
if __name__ == "__main__":  
    res = []
    for _ in range(3):
        res.append(list(map(float, input().strip('[]').split())))    
    print(res)
    
        