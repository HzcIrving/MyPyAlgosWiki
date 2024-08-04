import numpy as np 

"""  
[√]
KMeans PCode 
-------------------- 
1.目标  
    最小化每个点到其所属簇均值之间的距离之和 
    J = sum_1^n sum_j^k wij||x_i-u_j||^2 
        - wij = 1 if i == j 
        - uj 是j的均值  

step1: 
    - 随机选择K个簇中心
step2: 
    - 分配数据点到最近的簇 （距离准则）
step3: 
    - 重新计算簇中心（均值）
step4: 
    - 检查是否收敛
        4.1 -- 中心无明显变化
        4.2 -- 达到预定的迭代次数  
    - 重复上面步骤，直到收敛 
--------------------
""" 


class KMeans:
    def __init__(self, data, k, nIters) -> None:
        self.data = data # 数据主体 
        self.k = k # 簇中心 
        self.nIters = nIters # 最大迭代次数  
    
    def solve(self): 
        # 1. 初始化簇中心 (随机选取)
        centers = self.data[np.random.choice(self.data.shape[0], self.k, replace=False)]    
        print(centers.shape) # （2，2） 
        
        for _ in range(self.nIters):
            # 2. 分配数据点到最近的簇   
            # ** (6,2) -> (6,1,2) - (2,2) >>> (6,2,2)
            # dists = np.linalg.norm(self.data[:, np.newaxis]-centers, axis=2)   
            
            dists = np.zeros(shape=(6,2)) 
            dists[:, 0] = np.linalg.norm(self.data - centers[0,:], axis = 1) 
            dists[:, 1] = np.linalg.norm(self.data - centers[1,:], axis = 1)
             
            labels = np.argmax(dists, axis=1) # 确定每个数据点的簇标签 
            
            # 3. 重新计算簇中心 
            centers = np.zeros((self.k,2))
            for i in range(self.k):
                centers[i] = self.data[labels==i].mean(axis=0)
                
            
        # 4. 达到迭代最大值
        return centers, labels         
    
# 保留两位小数且不进位 
def get_floor(val):
    return "{:.2f}".format(np.floor(val*100)/100)  

if __name__ == "__main__":
    data = np.array([[1,2],[5,8],[1.5,1.8],[8,8],[1,0.6],[9,11]]) 
    k = 2 
    maxIters = 100 
    solver = KMeans(data, k, nIters=maxIters) 
    centers, labels = solver.solve() 
    print("簇中心:",  centers)
    print("标签:", labels)   
    
    #for pts in centers: 
        # print(",".join(map(str, [1,2]))) 
        # print(",".join(["{:.2f}".format(i) for i in pts]))  
    print(centers)
    print(";".join([",".join(["{:.2f}".format(i) for i in pts]) for pts in centers]))