from collections import OrderedDict 
# import numpy as np  
import random  

"""
常规的Dict被设计为非常擅长映射操作。 跟踪插入顺序是次要的
OrderedDict旨在擅长重新排序操作。 空间效率、迭代速度和更新操作的性能是次要的
OrderedDict在频繁的重排任务中还是比Dict更好，这使他更适用于实现各种 LRU 缓存
OrderedDict类的 popitem() 方法有不同的签名。它接受一个可选参数来指定弹出哪个元素
        弹出最后面元素：常规的Dict使用 d.popitem()  ，OrderedDict类使用od.popitem() 

        弹出第一个元素：常规的Dict使用 (k := next(iter(d)), d.pop(k))  ，OrderedDict类使用od.popitem(last=False)

类有一个 move_to_end() 方法，可以有效地将元素移动到任一端
        将K，V对移到最后面：常规的Dict使用 d[k] = d.pop(k)  ，OrderedDict类使用od.move_to_end(k, last=True)   

       将K，V对移到最前面：常规的Dict没有对应功能，OrderedDict类使用od.move_to_end(k, last=False)    
"""

d = OrderedDict() 

d['name'] = 'Hzc' 
d['age'] = 18 
d['money'] = 12222   

for k, v in d.items():
    print(k, v) 
    
d1 = OrderedDict(
    name = 'Luna', 
    age = 20, 
    money = 1111111 
)

for k, v in d1.items():
    print(k, v) 
    
print(d1.keys()) 
print(d1.values()) 

d1['dream'] = '23333' 
d1['other dream'] = 'love bf' 

print(d1) 

""" 1. 依据key 或者 Value进行排序 """ 
dd = {'banana': 3, 'apple':4, 'pear': 1, 'orange': 2} 
print(dd)
# ksort 
kd = OrderedDict(sorted(dd.items(), key=lambda x: x[0]))   
print(kd)   
kd = OrderedDict(sorted(dd.items(), key=lambda x: x[0], reverse=True))   
print(kd)  

dd = {'banana': (3, 5), 'apple': (4,5) , 'pear': (1, 3) , 'orange': (3, 9)}  
vd = OrderedDict(sorted(dd.items(), key=lambda x: x[1]))
print(vd)

# 现根据第一个ele进行排序，若相同，则根据第二个进行排序    
# from big to small 
vd = OrderedDict(sorted(dd.items(), key= lambda x: (x[1][0], x[1][1]), reverse=True)) 
print(vd ) 
import pdb; pdb.set_trace()
 
# small to big 
vd = OrderedDict(sorted(dd.items(), key= lambda x: (x[1][0], x[1][1]), reverse=False)) 
print(vd ) 

""" 2. 优秀特性之初始化字典 """  
name = ['A', 'B', 'C'] 
dic = OrderedDict() 
# dic.clear()   
for k in name: 
    dic.setdefault(k, random.randint(1, 10)) 

print(dic)  



""" 3. 优秀特性之移动key """              
# 将A移到最后 
dic.move_to_end('A') 
print(dic)  

dic.move_to_end('A', last=False) 
print(dic) 


""" 4.删除指定元素 """ 
dic.pop('A') 
print(dic) 
