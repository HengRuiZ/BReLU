import numpy as np
import matplotlib.pyplot as plt
import torch

def fibonacci(n):
    a=torch.ones(1,requires_grad=True)
    b=torch.ones(1,requires_grad=False)
    c=torch.ones(1,requires_grad=False)
    for i in range(n):
        c=a.detach()+b
        a.data=b
        b=c
    return a
if __name__ == '__main__':
    x=np.arange(0,10,0.2)
    y=np.sin(x)
    fig,ax=plt.subplots()
    n,bins,patchs=ax.hist(y,50)
    plt.show()
    pass