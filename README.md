# MyPyAlgosWiki
备考备考备考 

## 1. IO代码段 

- 单个
    e.g. `a = int(input()) `

- 单行List 
    e.g. `a = list(map(int, input().strip().split()))` 
        若 in: [[1 2 3],...     strip('[]') 
        若 in: 1,2,3, ...  split(',') 
- 多行List 
    e.g. `a = [list(map(int, input().strip().split())) for _ in range(n)]`  
    若不规定行数 
    ```
    try: 
        while True: 
            a = list(map(int, input().strip().split())) 
    except: 
        ...  
    ```


- 打印单行
    一个case: 7.33,9.00;1.17,1.47 
    ```
    print(";".join([",".join(["{:.2f}".format(i) for i in pts]) for pts in centers]))
    ``` 
