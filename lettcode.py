import numpy as np




def main(num,target):
    map = {}
    for a in num:
        v=a
        b=0
        if map[target - v] >= 0:
            map[v] = b
        else:
           map[v] = b




if __name__=="__main__":
    target  = 9
    num = [2,7,11,15]
    main(num,target) 
