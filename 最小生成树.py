""" 
最小生成树与Kruskal算法  

leetcode 1584: 
https://leetcode.cn/problems/min-cost-to-connect-all-points/   

并查集: 
https://zhuanlan.zhihu.com/p/93647900 
    > 合并 Union 不相交集合合并 
    > 查询 Find 查询两个元素是否在一个集合  
    
    # 初始化 
    int fa[MAXN]: 
    inline void init(int n) 
    {
        for(int i = 1; i<=n; ++i) 
            fa[i] = i; # 各自为战 
            rank[i] = 1； # 记录metrics（比如深度） 
    }
    
    # 查询  FIND 
    # 要判断两个元素是否属于同一个集合，只需要看它们的根节点是否相同即可。
    int find(int x){
        if(fa[x] == x) 
            return x; 
        else: {
            # method 1
            return find(fa[x]); # 对代表元素的查询：一层一层访问父节点，直至根节点 
            
            # method 2 --- 路径压缩，避免长链  
            fa[x] = find(fa[x])  # 将父节点设为根节点  
            return fa[x]; 
        }
    }  
     
    # 合并 UNION   
    # 先找到两个集合的代表元素，然后将前者的父节点设为后者即可
    inline void merge(int i, int j){ 
        fa[find(i)] = find(j);   
    }  
    # 注意，一开始，把所有元素的rank（秩）设为1。合并时比较两个根节点，把rank较小者往较大者上合并。 
    inline void merge(int i, int j){
        int x = find(i); 
        int y = find(j); # 根节点  
        if (rank[x] <= rank[y])  fa[x] = y; 
        else fa[y] = x; 
        if (rank[x] == rank[y] && x!=y) rank[y] ++ 
    } 
    
    # 最小生成树 -> 基于并查集方法生成（Kruscal) 
"""   

class Kruscal:
    def __init__(self, tree) -> None:
        self.tree = tree 
        self.minDist = 0  
        self.rank = [1] * len(self.tree)
    
    # 时间复杂度 
    # def Union(self, i, j): 
    #     """
    #     并查集合并
    #     将i合并到j
    #     """
    #     self.tree[self.Find(i)] = self.tree[j]  
    
    # def Union(self, i, j): 
    #     """   
    #     并查集合并
    #     将i合并到j
    #     """
    #     self.tree[self.Find(i)] = self.tree[j]  
    
    def Union(self,i,j): 
        # 查找i和j的祖宗节点
        fx, fy = self.Find(i), self.Find(j)
        # 如果祖宗节点相同，说明已经在同一个集合中，直接返回False
        if fx == fy:
            return False

        # 如果fx的rank小于fy的rank，交换两个节点的值
        # 都默认fy是短链, 不需要也可以通过  
        """ 
        如果 rank[fx] < rank[fy]，则交换 fx 和 fy 的值，使得 fx 始终表示 rank 较小的连通分量。然后将 fy 所在的连通分量合并到 fx 中，即将 fy 的根节点设置为 fx，同时更新 rank[fx] 的值。
        """
        if self.rank[fx] < self.rank[fy]:
            fx, fy = fy, fx
        
        # 将fy的rank值加到fx的rank值上 
        self.rank[fx] += self.rank[fy]
        # 将fy的父节点设置为fx
        self.tree[fy] = fx
        # 返回True
        return True


    def Find(self,i):  
        if self.tree[i] == i: 
            return self.tree[i]

        # 压缩路径 
        self.tree[i] = self.Find(self.tree[i]) # 递归查  
        return self.tree[i] # Root     

class Solution:
    def minCostConnectPoints(self, pts):    

        def calcDistance(pt1, pt2):
            return abs(pt2[0]-pt1[0]) + abs(pt2[1]-pt1[1])   

        self.pts = pts    

        # self.cost = [[0 for _ in range(len(self.pts)-1)] for _ in range(len(self.pts)-1)]  # 排除i,i
        self.costMap = [] 
        
        # Step1: 列出所有可能路径 
        for i in range(len(self.pts)): 
            for j in range(i+1, len(self.pts)):  
                self.costMap.append((calcDistance(self.pts[i], self.pts[j]),i,j))

        # Step2: 按成本排序 
        self.costMap.sort()
        
        # Step3: Kruscal算法（并查集)  
        # 1) 初始化
        self.minGenTree = [i for i in range(len(self.pts))]  
        Solver = Kruscal(self.minGenTree)  
        
        # 2) 最小生成树与最小cost
        cnt = 0  
        for dist, i, j in self.costMap: 
            if Solver.Union(i,j):  # 可以合并 
                Solver.minDist += dist 
                
                cnt += 1 
                if cnt >= len(self.minGenTree):
                    break # 早停     
        
        return Solver.minDist



# 下面这个时间复杂度有一些问题，只能过70/77 
# import math   

# class Solution:  
#     """ 
#     Kruskal算法是一种用来寻找加权无向图中最小生成树的算法。为了通俗地解释这个概念，我们可以用一个生活中的例子来说明：
#     这个case下， 想象一下，你是一个城市规划师，需要在一系列村庄之间修建道路，使得所有村庄都通过道路相连，并且总的道路修建成本最低。
#     Kruskal算法就像是你的规划策略。
#     以下是Kruskal算法的步骤，用上面的例子来解释：
#     1. **列出所有可能的路段**：首先，你需要列出所有可能的连接村庄的道路，并且知道每条道路的修建成本。
#     2. **按成本排序**：然后，你将这些道路按照修建成本从低到高排序。
#     3. **选择成本最低的道路**：接下来，你开始选择成本最低的道路来修建。但是，有一个重要的规则：如果一条新的道路修建后会导致某些村庄形成一个环（即已经可以通过其他道路到达），那么这条道路就不能选。
#     4. **重复选择**：你继续按照成本从低到高的顺序选择道路，每次都确保不会形成环。
#     5. **直到所有村庄相连**：这个过程一直持续，直到所有的村庄都通过道路相连，这时候你就得到了一个最小生成树，因为它包含了连接所有村庄的最小成本道路。
#     用更通俗的话来说，Kruskal算法就像是你在一张地图上用彩笔画线，线的长度代表道路的成本。你的目标是画最少的线，并且线的总长度尽可能短，同时确保每座村庄都可以通过这些线到达其他任何村庄，还不能画成任何封闭的圈。
#     通过这种方式，Kruskal算法帮助我们在保证连接所有节点的同时，找到一种成本最低的连接方式。
#     """
#     def __init__(self) -> None:
#         self.tree = None 
        
#     def minCostConnectPoints(self, pts):     
#         self.pts = pts  
#         # self.cost = [[0 for _ in range(len(self.pts)-1)] for _ in range(len(self.pts)-1)]  # 排除i,i
#         self.costMap = [] 
    
#         self.visit = [] 
        
#         # Step1: 列出所有可能路径 
#         for i in range(len(self.pts)): 
#             for j in range(len(self.pts)):  
#                 if j == i or (i,j) in self.visit: 
#                     continue  
#                 # self.cost[i][j] = self.calcDistance(self.pts[i], self.pts[j])    
#                 self.costMap.append((i,j,self.calcDistance(self.pts[i], self.pts[j])))  
#                 self.visit.append((i,j))
#                 self.visit.append((j,i))

#         # Step2: 按成本排序 
#         self.costMap.sort(key=lambda x: (x[2], x[0], x[1])) 
#         print(self.costMap)
        
#         # Step3: Kruscal算法（并查集)  
#         # 1) 初始化
#         self.minGenTree = [i for i in range(len(self.pts))]  
#         Solver = Kruscal(self.minGenTree)  
        
#         # 2) 最小生成树与最小cost
#         while self.costMap: 
#             i,j,dist = self.costMap.pop(0) # (i, j, dist) # dist是有序的，从小到大   
#             if Solver.Find(i) != Solver.Find(j):  # 检测是否已经联通（递归查询，并修正i、j的root)
#                 # 说明i,j为连   
#                 print(f"{i}-->{j}, cost:{dist}")
#                 Solver.Union(j,i) # 将j连到i  
#                 Solver.minDist += dist       
        
#         return Solver.minDist
    
#     @staticmethod
#     def calcDistance(pt1, pt2):
#         return abs(pt2[0]-pt1[0]) + abs(pt2[1]-pt1[1])
        
    
# class Kruscal:
#     def __init__(self, tree) -> None:
#         self.tree = tree 
#         self.minDist = 0 
    
#     def Union(self, i, j): 
#         """
#         并查集合并
#         将i合并到j
#         """
#         self.tree[self.Find(i)] = self.tree[j]  
    
#     # 时间复杂度有问题
#     # def Find(self,i): 
#     #     if self.tree[i] != i: 
#     #         self.tree[i] = self.Find(self.tree[i]) # 递归查  
#     #     return self.tree[i] # Root     
    
#     def Find(self,i): 
#         if self.tree[i] == i:
#             return self.tree[i] 

#         # 路径压缩 
#         self.tree[i] = self.tree[i] = self.Find(self.tree[i]) # 递归查 
#         return self.tree[i]
    

    

if __name__ == "__main__": 
    Solv = Solution()  
    pts = [[0,0],[2,2],[3,10],[5,2],[7,0]] 
    # pts = [[3,12],[-2,5],[-4,1]] 
    # pts = [[-1000000,-1000000],[1000000,1000000]]
    print(Solv.minCostConnectPoints(pts))    