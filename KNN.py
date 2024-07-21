""" 
KNN聚类 
---------------------- 
1. 基于实例的方法 
2. 核心思想： 一样样本在特征空间中的k个最近邻的样本中的大多数属于某一个类别，则该样本也属于这个类别 
---------------------- 
S1 
    选择邻居数量k 
S2 
    计算距离（L2 Distance） 
S3 
    找最近邻居（排序，选topk） 
S4 
    投票，k个最近邻类别投票，的票最多的就是预测类别  
"""
import numpy as np  
from collections import Counter

class KNN:
    def __init__(self, data, label, K) -> None:
        self.data = data 
        self.label = label  
        self.topk = K 
        
    
    def solve(self, X): 
        y_pred = np.array([self.predict(x) for x in X])  
        return y_pred
        
    
    def predict(self, data):
        # 计算每个样本之间距离
        dists = [np.sqrt(np.sum(data_train - data)**2) for data_train in self.data] 
               
        # 获取topK个最近邻居   
        # 排序，并返回indices 
        k_indices = np.argsort(dists)[:self.topk] 
        
        # 获取k个最近邻居的标签 
        k_nearest_labels = [self.label[i] for i in k_indices]  
        
        # 返回数量最多的类别作为这批预测的结果  
        # results = OrderedDict() 
        # for res in k_nearest_labels: l
        #     if res not in results.keys():
        #         results.setdefault(res, 1)  
        #     else:
        #         results[res] += 1     
        
        # 投票  
        res = Counter(k_nearest_labels).most_common(1)  
        
        return res[0][0]

if __name__ == "__main__": 
    X_train = np.array([[1.0, 1.1],[1.0, 1.1],[1.1,1.2],[2.1,2.2],[2.5,2.7],[3.1,3.2],[1.6,1.5],[2.3,5.6]]) 
    Y_train = np.array([0, 0, 0, 1, 1, 1, 0, 2]) 
    
    X = np.array([[0.9, 0.9]]) 
    
    solver = KNN(X_train, Y_train, 5) 
    print(solver.solve(X)) 