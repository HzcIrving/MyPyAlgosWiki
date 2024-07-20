""" 
指数加权平均是一种常用的统计方法，用于计算数据序列的移动平均。这种方法通过给予最近的观测值更高的权重，从而赋予最近的观测值更大的影响力。以下是指数加权平均的算法流程：

1: 初始化：首先，需要初始化指数加权平均的变量，这通常是通过将序列的前几个值平均得到。
2: 权重衰减：每个观测值都有一个对应的权重，这个权重随着时间递减。通常，权重以指数形式递减，例如，上一个观测值的权重是(1-α)，下一个观测值的权重是α，其中α是一个介于0和1之间的参数，称为权重衰减率。
3: 更新过程：每次接收到新的观测值时，都会更新指数加权平均的值。更新的公式通常如下：
EWA(t) = (1-α) * EWA(t-1) + α * x(t)
4: 其中，EWA(t)是t时刻的指数加权平均值，x(t)是t时刻的观测值，α是权重衰减率。
5: 权重分配：随着时间的推移，每个观测值对应的权重会逐渐减小，因此观测值对当前指数加权平均值的影响也会逐渐减小。
"""  

import numpy as np 

# 保留两位小数且不进位 
def get_floor(val):
    return "{:.2f}".format(np.floor(val*100)/100)  

# 保留两位小数且进位  
def get_ceil(val): 
    return "{:.2f}".format(np.ceil(val*100)/100)

def func(adjust=True):  
    """ 
    使用 `adjust` 参数的好处主要体现在以下几个方面：
    1. **灵活性**：通过调整 `adjust` 参数，可以选择不同的计算方法来适应不同的数据分析需求。
    2. **适应性**：当 `adjust=True` 时，计算考虑了整个数据序列的动态变化，这对于分析时间
                  序列数据尤其有用，因为它能够反映出整个历史记录对当前值的影响。
    3. **简化计算**：当 `adjust=False` 时，计算更为简化，这可能对于某些快速估计或初步分析有用，
                    尤其是当数据量很大时，可以减少计算量。
    """
    def minMaxScale(lst): 
        min_val = min(lst) 
        max_val = max(lst)  
        scaled_lst = [(x-min_val)/(max_val-min_val) for x in lst]
        return scaled_lst  
            
    # traffic =  
    traffic = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5] 
    
    traffic_copy = traffic.copy() 
    traffic_copy = minMaxScale(traffic_copy) 
    for i in range(len(traffic_copy)):
        traffic_copy[i] = get_floor(traffic_copy[i])
    print(f"original:{traffic_copy}")
    
    alpha = 0.5  
    num = len(traffic) - 1  
    i = 0 
    t = num 
    alpha_ = 1 - alpha 
    
    # 先缩放列表元素  
    new_lst = minMaxScale(traffic) 
    
    # 缩放后序列计算加权平均后的流量序列  
    up = below = 0  
    
    output = []
    if adjust: 
        while i <= num: 
            up += (alpha_ ** (t-i))*new_lst[i] 
            below += alpha_ ** (t-i)  
            i+=1 
            output.append(get_floor(up/below)) 
        t+=1 
    
    else: 
        output.append(get_floor(new_lst[0])) 
        y = new_lst[0]  
        for i in range(1, len(new_lst)): 
            up = alpha * new_lst[i] + alpha_ * y  
            # print(up)
            output.append(str(get_floor(up)))
            y = up 

    # print(output)
    print(",".join(output))  
    


if __name__ == "__main__": 
    test_val = 23.953
    print(get_floor(test_val)) # 23.90  
    print(get_ceil(test_val)) # 24.00  
    
    func(adjust=True) 
    func(adjust=False)