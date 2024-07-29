import ast  
import numpy as np 


""" 
LSTM 
LSTM（Long Short-Term Memory）是一种特殊的循环神经网络（RNN），用于处理序列数据。
> LSTM的主要目的是解决RNN在处理长序列数据时出现的梯度消失问题。
> LSTM通过引入门（gate）机制来控制信息的流动，从而实现对长序列数据的记忆。

1) 输入层 
    > 将输入序列转换为特征向量x_t，t表示时间步。 
2) 遗忘门（Forget Gate）
    > 计算当前时间步的遗忘门值f_t，用于决定是否保留cell c_t-1中的信息。  
    > f_t = sigmoid(W_f * [h_t-1, x_t])  
3) 输入门 (Input Gate)
    > 计算当前时间步的输入门值i_t，用于决定细胞状态c_t-1中应该保留哪些信息。   
    > i_t = sigmoid(W_i * [h_t-1, x_t])  
4) 细胞状态更新  
    > 计算当前时间步的细胞状态c_t，根据遗忘门值f_t和输入门值i_t更新细胞状态： 
        c_t = f_t * c_t-1 + i_t * tanh(W_c * [h_t-1, x_t]) 
5) 输出门（Output Gate）：
    > 计算当前时间步的输出门值o_t，用于决定细胞状态c_t中的信息如何输出。计算方法如下：  
        o_t = sigmoid(W_o * [h_t-1, c_t]) 
6) 隐层更新 
    > 计算当前时间步的隐藏状态h_t，根据细胞状态c_t和输出门值o_t更新隐藏状态：
""" 

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))   

def tanh(x): 
    s1 = np.exp(x) - np.exp(-x) 
    s2 = np.exp(x) + np.exp(-x) 
    s = s1 / s2 
    return s  

def LSTM_numpy(x, w_ih, b_ih, w_hh, b_hh): 
    N, L, D = x.shape 
    H = int(b_hh.shape[0] / 4)   # for hidden h & c (4通道分别计算？)
    
    # for h_t, c_t init 
    h_t = np.zeros(shape=(1,N,H)) 
    c_t = np.zeros(shape=(1,N,H)) 
    
    # seq 
    for t in range(L):    
        x_t = x[:, t, :] 
        # params sharing 
        x_m = x_t @ w_ih + b_ih + h_t @ w_hh + b_hh  
        
        # for input gate 
        i_t = sigmoid(x_m[:,:,:H]) 
        f_t = sigmoid(x_m[:,:,H:2*H]) 
        g_t = tanh(x_m[:,:,2*H:3*H])
        o_t = sigmoid(x_m[:,:,3*H]) 
        c_t = f_t * c_t + i_t * g_t  
        h_t = o_t * tanh(c_t) 
        
    return (h_t * 100).astype(np.int32)
    