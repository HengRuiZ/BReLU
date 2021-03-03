import numpy as np
import matplotlib.pyplot as plt
import torch
import struct

def fibonacci(n):
    a=torch.ones(1,requires_grad=True)
    b=torch.ones(1,requires_grad=False)
    c=torch.ones(1,requires_grad=False)
    for i in range(n):
        c=a.detach()+b
        a.data=b
        b=c
    return a
def modify_blu(blufile):
    with open(blufile,'rb') as f:
        buf=f.read()
        avg_pool_blu=struct.unpack('f',buf[-36:-32])
    return 0
if __name__ == '__main__':
    modify_blu('blu152_n3.data')
