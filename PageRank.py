"""
图算法应用  PageRank 
- 核心思想：
    1） 若一个网页被很多其他网页链接，说明网页重要，PageRank值高  
    2） PageRank值高的网页链接到其他网页，说明被链接网页PageRank值也会提高 

- 迭代过程 
    V' = alpha*M*V + （1-alpha)e
    > alpha:阻尼系数，0.85 
    > M: nxn方阵，网页链接关系，被称为转移矩阵； 
    > V: Page Range值 
    > e: 访问每个网页的默认概率  

- case： 客户影响力评估 
    InfluenceRank(pi) = 1-alpha * (1/N) + alpha*sum(pj∈pi通话圈) InfluenceRank(pj) * Tji / Lj 
    >  InfluenceRank(pi) 客户pi影响力评估
    >  N 总客户数 
    >  alpha 阻尼系数，默认0.85 
    >  Tji 表示客户pj打给pi的次数 
    >  Lj 客户pj的主叫总次数  
    >  每个客户初始得分为这个客户的主叫总次数 （Lj=0, 表示该客户没有主叫呼出，此时    InfluenceRank(pj) * Tji / Lj  约定为 0)  

- 要求: 
    > 最后返回值保留2位小数 
    > TopK准则： 
        先影响力  
        后客户ID  降序排
    
""" 
import numpy as np  
from copy import deepcopy

def get_inputs():
    N = 4  # 客户总数  
    M = 5  # 客户通话统计条数 
    I = 3  # 迭代次数 
    K = 2  # 输出topK  
    
    # (A,B,C) 客户A打给客户B的通话总数C 
    # ID编码从0开始，连续编号 
    Rels = np.array([[0,1,5],
                     [0,3,5], 
                     [1,2,10], 
                     [1,3,10], 
                     [3,2,5]])  
    
    return N,M,I,K,Rels 

def get_inputs_by_hand(): 
    N,M,I,K = list(map(int, input().strip().split()))
    Rels = [] 
    for _ in range(M):
        Rels.append(list(map(int, input().strip().split())))
    
    return N,M,I,K,np.array(Rels)   

def Rels2Tij(Rels,N):   
    # Tij: n*n 
    Tij = np.zeros(shape=(N,N)) 
    
    for re in Rels: 
        Tij[re[0], re[1]] = re[2]  

    return Tij
    
    
def PageRank():  
    """ 
        InfluenceRank(pi) = 1-alpha * (1/N) + alpha*sum(pj∈pi通话圈) InfluenceRank(pj) * Tji / Lj 
    """

    # debug 
    # N,M,I,K,Rels = get_inputs()   
    N,M,I,K,Rels = get_inputs_by_hand()
    
    alpha = 0.85 
    normal = 1/N 
    constant_ = (1-alpha) * normal
      
    
    # Tij 
    T = Rels2Tij(Rels, N)    
    
    # 统计每个客户的主叫次数 
    L = [] 
    for t in T:  
        L.append(np.sum(t)) 
    print(L) 
    
    # 初始PangeRanke值 V0    
    V = np.array(deepcopy(L))  
    # V = V[np.newaxis, :] 
       
    
    # cnt = 0 
    # 迭代次数   
    for _ in range(I): 
        V_next = np.array([constant_ for _ in range(N)]) 
        for i in range(N): 
            for j in range(N): 
                if L[j] != 0: 
                    V_next[i] += alpha * V[j] * T[j,i] / L[j] 
                else:
                    V_next[i] += 0.0   

        V = V_next
    
    # 排序  
    res = []
    for i in range(N):
        res.append((i, V[i]))  
        
    res.sort(key=lambda x: (x[1], x[0]), reverse=True) 
    # print(res)  
    
    for i in range(K): 
        print("{} {:.2f}".format(int(res[i][0]), res[i][1]))
        
     
    
    

if __name__ == "__main__":
    PageRank()
    
    
    
    
    
    
    
    