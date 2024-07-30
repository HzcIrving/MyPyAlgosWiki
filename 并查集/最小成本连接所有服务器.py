""" 
n个服务器的网络，1~n编号 
-------------------------- 
已知
1. 服务器数量 
2. 服务器两两连接关系
3. connections[i] = [xi, yi, costi] 
    x-y; cost 
4. 求连接给定网络服务器的最小成本 
    - 用全部连接成本的总和
    - 每对服务器之间最少有一条连接
    - 无法连接所有n个服务器，返回-1  
""" 
import numpy as np  
        
def test_inputs(): 
    n_cons = 5 
    connections = [
        [1,2,1], 
        [1,3,7], 
        [1,4,5], 
        [1,5,3], 
        [2,3,2], 
        [2,5,6], 
        [3,5,3], 
        [4,5,4] 
    ] 

    return n_cons, connections 

def func(): 
    # 1. n服务器数 
    n, connections = test_inputs() 
    
    # 特殊情况 --- 不满足至少有一个连接，有悬空节点  
    if len(connections) < n - 1:
        return -1  
    
    # 特殊情况 --- 单个节点 
    if n == 1:
        return 0 
    
    # 2. 按照成本排序 
    connections.sort(key=lambda x: x[2])  
    print(connections) 
    
    # 3. 初始化root为自身 (生成树)
    root_for_each_node = [i for i in range(n)] 
    
    # 4. 并查集  
    # 4.1 find --- 要判断两个元素是否属于同一个集合，只需要看它们的根节点是否相同即可。
    def find(x):
        # 查找x的root
        if root_for_each_node[x] != x: 
            root_for_each_node[x] = find(root_for_each_node[x])  # 查找并更新 x 的root, 使其能指到底部  
        return root_for_each_node[x]  

    # 4.2 union --- 先找到两个集合的代表元素，然后将前者的父节点设为后者即可
    def union(x,y): 
        root_for_each_node[find(x)] = find(y) 
        
        
    res = 0 
    for x,y,cost in connections: 
        # 从小到大  
        if find(x-1) != find(y-1): 
            # 合并 
            union(x-1,y-1) 
            res += cost 
    
    # 检查 是否只有一个root 
    check_flag = find(0) 
    for i in range(1, n): 
        if (check_flag != find(i)): 
            return -1     

    
if __name__ == "__main__":
    func()