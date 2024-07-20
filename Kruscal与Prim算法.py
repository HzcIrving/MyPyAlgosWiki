""" 
Kruscal (贪心 + 并查集)  https://www.bilibili.com/video/BV1yK4y1f7YK/?spm_id_from=333.337.search-card.all.click&vd_source=6c24fa112801abdbc5741fa1a55aea2d 
- https://blog.csdn.net/Floatiy/article/details/79424763 
- https://www.cnblogs.com/unique-pursuit/p/16059734.html 

"""

class Kruskal:  
    """ 
    Kruskal算法是一种用来寻找加权无向图中最小生成树的算法。为了通俗地解释这个概念，我们可以用一个生活中的例子来说明：
    想象一下，你是一个城市规划师，需要在一系列村庄之间修建道路，使得所有村庄都通过道路相连，并且总的道路修建成本最低。
    Kruskal算法就像是你的规划策略。
    以下是Kruskal算法的步骤，用上面的例子来解释：
    1. **列出所有可能的路段**：首先，你需要列出所有可能的连接村庄的道路，并且知道每条道路的修建成本。
    2. **按成本排序**：然后，你将这些道路按照修建成本从低到高排序。
    3. **选择成本最低的道路**：接下来，你开始选择成本最低的道路来修建。但是，有一个重要的规则：如果一条新的道路修建后会导致某些村庄形成一个环（即已经可以通过其他道路到达），那么这条道路就不能选。
    4. **重复选择**：你继续按照成本从低到高的顺序选择道路，每次都确保不会形成环。
    5. **直到所有村庄相连**：这个过程一直持续，直到所有的村庄都通过道路相连，这时候你就得到了一个最小生成树，因为它包含了连接所有村庄的最小成本道路。
    用更通俗的话来说，Kruskal算法就像是你在一张地图上用彩笔画线，线的长度代表道路的成本。你的目标是画最少的线，并且线的总长度尽可能短，同时确保每座村庄都可以通过这些线到达其他任何村庄，还不能画成任何封闭的圈。
    通过这种方式，Kruskal算法帮助我们在保证连接所有节点的同时，找到一种成本最低的连接方式。

    """
    # 并查集 + 排序 
    def __init__(self, dist, tree): 
        # 树 
        # parent -> node1 
        #        -> node2  
        self.min_dist = dist  
        self.tree = tree 
         
    def search(self, i):
        # i:节点索引 
        if self.tree[i] != i:  
            self.tree[i] = self.search(self.tree[i]) # 递归搜  
        
        return self.tree[i]   
    
    def union(self, i, j): 
        # 合并两个节点到一颗树 
        # 并查集  
        # e.g. 
        # A->B: cost 2 
        # 则将A的tree(A.idx) = B  
        self.tree[self.search(i)] = self.search(j)  
        

# # ut for Kruscal 
# tree = [0, 0, 0, 2] 
# Kruskal = Kruskal(0, tree)  

# # 新节点 
# edge = [3, 2, 10] 
# Kruskal.union(edge[0], edge[1])   # -> (0,0,0,0)


citys = ['A','B','C','D','E','F','G']
# 7 x 7 的邻接矩阵
# 不相邻，cost = -1 

# 对称矩阵 data = data.T 
data = [[-1, 8, 7, -1, -1, -1, -1], 
        [8, -1, 6, -1, 9, 8, -1], 
        [7, 6, -1, 3, 4, -1, -1],
        [-1, -1, 3, -1, 2, -1, -1], 
        [-1, 9, 4, 2, -1, -1, 10], 
        [-1, 8, -1, -1, -1, -1, 2], 
        [-1, -1, -1, -1, 10, 2, -1]] 

# data[i][j] = cost  
# 构建(i,j,cost)数据结构  

# 1. **列出所有可能的路段**
queue = []  
connect_state_rec = [[True for _ in range(len(citys))] for _ in range(len(citys))]

# 构建数据结构以及图 
for i in range(len(data)):
    for j in range(len(data[0])): 
        if data[i][j] != -1 and connect_state_rec[i][j] and connect_state_rec[j][i]: 
            queue.append([i, j, data[i][j]])   
        else:
            connect_state_rec[i][j] = connect_state_rec[j][i] = False  

# 2. **按成本排序** 从小到大  
queue.sort(key=lambda x: (x[2], x[0]))  
print(queue)  
print(connect_state_rec)

tree = [i for i in range(len(citys))] # 初始化 
Kruskal = Kruskal(0, tree)

# 3. **选择成本最低的道路**
while(len(queue)): 
    edge = queue.pop(0) # (i,j,cost) 
    
    # 4. 判断两个点是否直接联通？ 
    # 不连通，使用Union将两顶点合并 
    if Kruskal.search(edge[0]) != Kruskal.search(edge[1]):    
        print(f"{citys[edge[0]]} -> {citys[edge[1]]} : cost {edge[2]}")  
        Kruskal.union(edge[0], edge[1])  
    
        # 5. **重复选择**  
        Kruskal.min_dist += edge[2]  

print("total cost:", Kruskal.min_dist)