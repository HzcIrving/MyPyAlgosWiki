""" 
heapq 是 Python 的一个内置库，提供了堆队列算法的实现。

1) 堆是一种特殊的二叉树，它的每个结点的值都大于或等于（最大堆）或小于或等于（最小堆）其子结点的值。
2) heapq 库提供了在 Python 中实现堆的各种方法，常用于优先队列、堆排序等。

下面是 heapq 的一些常用用法：
"""  

import heapq 
import copy 

# 1. 堆化 -- 默认最小堆 
heap = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0] 
heapq.heapify(heap)  
print(heap)  # [0, 1, 2, 6, 3, 5, 4, 7, 8, 9] 

# 2. 插入 
heapq.heappush(heap, 0.1)  # [0, 0.1, 2, 6, 1, 5, 4, 7, 8, 9, 3] 
print(heap)  

# 3. 弹出 ---  函数弹出堆中最小（或最大）的元素。 
print(heapq.heappop(heap))  # 0 
print(heapq.heappop(heap))  # 0.1 
print(heap) # [1, 3, 2, 6, 9, 5, 4, 7, 8]   

# 4. 堆顶元素 
min_element = heap[0] 
print(min_element) 

# 5. 堆大小 
heap_size = len(heap) 
print(heap_size)
