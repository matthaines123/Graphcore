import torch 
import numpy as np
# initialization
x = torch.tensor(np.random.rand(1)).requires_grad_(True)

iterations = 0
learningRate = 0.5

while (x.grad is None or torch.abs(x.grad)>0.01):
    iterations += 1
    if (x.grad is not None):
        # zero grads
        x.grad.data.zero_()
    # compute fn
    y = (x*x)
    # compute grads
    y.backward()
    # move in direction of / opposite to grads
    x.data = x.data - learningRate*x.grad.data
    # use below line to move uphill 
    # x.data = x.data + 0.01*x.grad.data

print(x)
print('Iterations: {}'.format(iterations) )