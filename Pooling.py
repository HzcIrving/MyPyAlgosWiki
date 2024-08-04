import numpy as np 

def get_input(): 
    # m,n,stride,size,p = list(map(int, input().strip().split()))  
    # print(m,n,stride,size,p)
    m = 5 
    n = 5 
    stride = 1 
    size = 2 
    p = 1 

    data = np.array([[1,0,1,0,1],
                     [1,0,1,0,1], 
                     [1,0,1,0,1],
                     [1,0,1,0,1], 
                     [1,0,1,0,1]])  
    return m, n, stride, size, p, data 

def get_input_by_hand():  
    m, n, stride, size, p = list(map(int, input().strip().split()))  
    
    # 输入  
    data = [] 
    for _ in range(m):
        data.append(list(map(int, input().strip().split())))  
        print(data) 
        
    data = np.array(data)
    print(data)
    
    return m, n, stride, size, p, data
    
def pooling(op="mean"): 
    # 1) 获得输入 
    m, n, stride, size, p ,data = get_input() 
    # m, n, stride, size, p , data = get_input_by_hand()
    
    # 2) pooling计算 --- 向下取整 
    out_m = (m - size) // stride + 1  
    out_n = (n - size) // stride + 1  
    
    outputs = np.zeros(shape=(out_m, out_n))
    print(outputs.shape)
    for i in range(out_m):
        for j in range(out_n): 
            if op == "mean" and p == 1: 
                outputs[i,j] = np.mean(data[i*stride:i*stride+size, j*stride:j*stride+size])
            else: # max 
                outputs[i,j] = np.max(data[i*stride:i*stride+size, j*stride: j*stride +size]) 
    
    print(outputs) 

if __name__ == "__main__": 
    pooling()