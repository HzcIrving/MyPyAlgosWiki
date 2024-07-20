import numpy as np  
from collections import OrderedDict

def func():  
    
    def calc_grad(x, y):
        x_ = 3*x*(y**2) + 4*x*y + y 
        y_ = 3*(x**2)*y + 2*(x**2) + x + 2*y + 1 
        return x_, y_ 
    
    
    
    # 获得初始梯度值  
    init_value = list(map(float, input().strip().split()))
    
    # worker 
    async_order = list(map(int, input().strip().split()))
    
    # 学习率  
    learning_rate = float(input().strip()) 
    
    
    # 先在每个worker上保存梯度  ， 初始化梯度 
    grad_d = OrderedDict() 
    for w in async_order: 
        grad_d[w] = init_value   
        
    x_0, y_0 = init_value[0], init_value[1]  
    wx, wy = x_0, y_0 
    
    # i与对应worker 
    for i, worker in enumerate(async_order):
        # 先进行梯度计算  
        x_0, y_0 = grad_d[worker][0], grad_d[worker][1]  
        gx, gy = calc_grad(x_0, y_0) 
        
        # 模型权重 
        wx = wx - learning_rate * gx 
        wy = wy - learning_rate * gy 
        
        grad_d[worker] = [wx, wy]  
    
    # 输出   
    print(grad_d)
    output = [wx, wy] 
    print(f"{output[0]:.3f} {output[1]:.3f}")  
    
if __name__ == '__main__': 
    # func()  
    # 模拟输入
    # import io
    # import sys
    
    # # 重定向标准输入
    # sys.stdin = io.StringIO("1.0 2.0\n1 2 3\n0.1")
    
    # 调用主函数
    func()
    # 恢复标准输入
    # sys.stdin = sys.__stdin__