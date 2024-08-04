""" 
[√]
卷积计算 

"""

import numpy as np 

def test_case(): 
    chs = 1 
    height = 5 
    width = 5 
    
    X = np.array([[
        [1,1,1,0,0],
        [0,1,1,1,0], 
        [0,0,1,1,1],
        [0,0,1,1,0], 
        [0,1,1,0,0]
    ]]) 
    
    stride = 1 
    kernel_h = 3
    kernel_w = 3   # 默认是与input同chs数量
    kernel = np.array([[
        [1,0,1],
        [0,1,0],
        [0,0,1] 
    ]]) 
    
    padding= 0   
    padding_mode = 0
    
    return chs, height, width, X, stride, kernel_h, kernel_w, kernel, padding , padding_mode

def CONV():
    chs, H, W, X, stride, kh, kw, kernel, pad, pad_mode = test_case() 
    
    if pad == 0: 
        output_h = int((H - kh) // stride + 1)   
        output_w = int((W - kw) // stride + 1) 
    else:  
        pad_size = 1 
        output_h = int((H - kh + 2*pad_size) // stride + 1) 
        output_w = int((W - kh + 2*pad_size) // stride + 1) 
        
    # 是否对数据padding 
    if pad == 1:  
        pad_size = 1 
        # Channel维不padding 
        if pad_mode == 0: 
            mode = "constant"  # default 
        else:
            mode = "edge" 
            
        # (0,0) for ch 
        # (pad_size, pad_size) for h 
        # (pad_size, pad_size) for w 
        X = np.pad(X, ((0,0),(pad_size, pad_size),(pad_size, pad_size)), mode)
    
    outputs = np.zeros(shape=(output_h, output_w))
    
    for i in range(output_h):
        for j in range(output_w): 
            for c in range(chs):  
                outputs[i,j] += np.sum(X[c, i*stride:i*stride+kh, j*stride:j*stride+kw]*kernel[c,:,:])
    
    print(outputs)

if __name__ == "__main__": 
    CONV()
                
    