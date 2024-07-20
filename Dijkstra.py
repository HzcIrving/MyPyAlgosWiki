""" 
Leetcode 743 Dijkstra算法 
---------------------------- 
https://leetcode.cn/problems/network-delay-time/  
"""
import heapq 

class Solution: 
    def __init__(self) -> None:
        pass
    
    def networkDelayTime(self, times, n, k): 
        """ 
        times[i] = (ui, vi, wi) 
        - ui: src node 
        - vi: tar node 
        - wi: time consumption from ui to vi    
        
        引入优先级对列（小根堆 heapq) 
        """ 
        g = [[] for _ in range(n)]  
        
        # g是从每个节点出发，到其他节点以及花费的时间  
        for src, tar, time in times:
            g[src-1].append((tar-1, time))    
        
       
        # 从k到每个节点的距离 
        dist = [float("inf") for _ in range(n)] 
        dist[k-1] = 0 # k2k 到自身距离为0 
        
        # 从k出发，到k的距离开销是0 
        # time, k 
        queue = [(0, k-1)]  
        while queue:  
            # 弹出最小 -------------------------
            # 类似于优先级队列 
            time, v = heapq.heappop(queue)  
            # ----------------------------------
            
            # 已经访问，且当前dist[v]更优 
            if dist[v] < time: 
                continue  
            
            # 检查有没有通路可以直接到达后续的g[tar] 
            for tar, time in g[v]:  
                # 从v出发 
                d = dist[v] + time    
                # 如果从当前节点 v 到邻接节点 tar 的距离 d 
                # 比已知的从k到tar的最短距离 dist[tar] 还要小
                # 那么就更新 dist[tar] 为 d，并将 (d, tar)  
                # 加入优先队列 queue 中，以便在下一轮中考虑从节点 tar 出发的边。
                if d < dist[tar]: 
                    dist[tar] = d 
                    heapq.heappush(queue, (d, tar))
            
        res = max(dist) 
        if res < float("inf"): 
            return res 
        else: 
            return -1 # 
        
        
                    
                    
    

if __name__ == "__main__": 
    solver = Solution()  
    # times = [[2,1,1],[2,3,1],[3,4,1]]  
    # n = 4
    # k = 2 
    # times = [[1,2,1]] 
    # n = 2
    # k = 1

    times = [[1,2,1],[2,3,2],[1,3,2]]
    n = 3 
    k = 1
    
    res = solver.networkDelayTime(times, n, k)
    print(res)
        
         
        