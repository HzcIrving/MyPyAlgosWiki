import numpy as np 


# 
def test_case(): 
    n = 6 # 样本条数
    
    inputs = [
                [1, 1, 0, 1, 1], 
                [1, 0, 0, 1, 1],
                [0, 1, 0, 0, 1], 
                [0, 1, 0, 1, 0], 
                [0, 1, 0, 0, 0], 
                [0, 0, 0, 1, 0]
            ]  
    
    labels = [0, 1, 1, 0, 0, 0] 
    
    return n, inputs, labels 

def func(): 
    
    # 测试用例 
    n, inputs, labels = test_case() 
    
    X = np.array(inputs) 
    Y = np.array(labels) 
    H_D = compute_HD(X,Y)
    
    # 根据每个特征划分数据集  
    feature_nums = X.shape[1]  
    
    Entropy = np.zeros(shape=(feature_nums,))
    for feat_idx in range(feature_nums):  
        subXs, subLabels = datasetSplit(X, feat_idx, Y) 
        
        # 计算该特征的信息熵 
        H_DA = compute_HDA(subXs, subLabels, total_samples=n)  
        Entropy[feat_idx] = H_D - H_DA 
    
    print(np.max(Entropy)) 
    print(np.argmax(Entropy))
        

def datasetSplit(X,feature_id,labels): 
    subXs = [] 
    subLabels = []
    for i in range(2): # 0/1特征 
        Di = X[X[:,feature_id] == i]  
        Li = labels[np.where(X[:, feature_id]==i)]
        subXs.append(Di)  
        subLabels.append(Li)
        
    return subXs,subLabels
    
    
def compute_HD(X,Y): 
    # 1. 统计各类样本个数  
    unique_labels, counts = np.unique(Y, return_counts=True) 
    print(unique_labels, counts)  
    
    label_nums = unique_labels.shape[0] 
    print(label_nums)  
    
    counts = counts / X.shape[0]  
    print(counts) 
    
    log_counts = np.log2(counts)
    print(log_counts) 
    
    # multiply是element-wise的乘法  等价于"*"
    H_D = np.multiply(counts, log_counts)
    print(H_D)
    
    H_D = np.sum(H_D) * (-1.0)
    return H_D.round(2) 

def compute_HDA(X,Y, total_samples):  
    sub_n = len(X) 
    
    H_DA = 0.0  
    for i in range(sub_n):  
        H_DA += (len(X[i])/total_samples) * compute_HD(X[i], Y[i]) 
    
    return H_DA.round(2) 
    
    
    
if __name__ == "__main__": 
    func()
    
    
    
        