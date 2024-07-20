""" 
Leetcode 
https://leetcode.cn/problems/vEAB3K/   

可信模拟题  
""" 


class Solution: 
    def __init__(self, graph) -> None:
        self.graph = graph 
        self.n_nodes = len(self.graph)    

    
    def isBipartiteDFS(self):    
        n_nodes = self.n_nodes
        graph = self.graph  
        visit = [0 for _ in range(self.n_nodes)]   # 记录访问节点    
        color = [0 for _ in range(self.n_nodes)]  # 记录颜色   
        self.flag = True  # 标志位 
        
        def DFS(node): 
            # 递归终止条件  
            if not self.flag:
                return 
            
            # 先标记visited标志  
            visit[node] = 1   
            
            for ne in graph[node]: 
                if visit[ne] == 0: # 未染色  
                    color[ne] = 1 if color[node] == 0 else 0 
                    DFS(ne) 
                else: 
                    if color[ne] == color[node]: 
                        self.flag = False 
                        return  
        
        # 遍历，考虑非联通图  
        for i in range(n_nodes): 
            if visit[i] == 0: 
                DFS(i)  
        
        return self.flag
        
    
    def isBipartiteBFS(self):  
        n_nodes = len(self.graph)  
        graph = self.graph
        visit = [0 for _ in range(self.n_nodes)]   # 记录访问节点    
        color = [0 for _ in range(self.n_nodes)]  # 记录颜色    
        
        for v in range(n_nodes): 
            if visit[v] == 0:  # 防止有非联通图 
                # 初始化 
                queue = [v] 
                visit[v] = 1  
                
                while queue: 
                    v = queue.pop(0)   # 弹出首节点 
                    for ne in graph[v]: 
                        if visit[ne] == 1: # 已经访问
                            if color[ne] == color[v]:
                                return False 
                        else: 
                            # 标记颜色 
                            color[ne] = 1 if color[v] == 0 else 0 
                            visit[ne] = 1 
                            queue.append(ne)  
                            
        return True         

        
         
    
if __name__ == "__main__": 
    print("Hello")  
    graph = [[1,2,3],[0,2],[0,1,3],[0,2]]   
    graph = [[1,3],[0,2],[1,3],[0,2]] 
    graph = [[],[2,4,6],[1,4,8,9],[7,8],[1,2,8,9],[6,9],[1,5,7,8,9],[3,6,9],[2,3,4,6,9],[2,4,5,6,7,8]] # 有非联通
    
    soluDFS = Solution(graph)  
    print(soluDFS.isBipartiteDFS()) 
    
    soluBFS = Solution(graph) 
    print(soluBFS.isBipartiteBFS()) 
    
    