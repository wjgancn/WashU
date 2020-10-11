### EDF --- An Autograd Engine for instruction
## (based on joint discussions with David McAllester)

import numpy as np

# Global list of different kinds of components
ops = []
params = []
values = []


# Global forward
def Forward():
    for c in ops: c.forward()

# Global backward    
def Backward(loss):
    for c in ops:
        c.grad = np.zeros_like(c.top)
    for c in params:
        c.grad = np.zeros_like(c.top)

    loss.grad = np.ones_like(loss.top)
    for c in ops[::-1]: c.backward() 

# SGD
def SGD(lr):
    for p in params:
        p.top = p.top - lr*p.grad


## Fill this out        
def init_momentum():
    for p in params:
        p.momvalue = np.zeros_like(p.top)


## Fill this out
def momentum(lr,mom=0.9):
    for p in params:
        p.top = p.top - lr * (p.grad + mom * p.momvalue)
        p.momvalue = p.grad

###################### Different kinds of nodes

# Values (Inputs)
class Value:
    def __init__(self):
        values.append(self)

    def set(self,value):
        self.top = np.float32(value).copy()

# Parameters (Weights we want to learn)
class Param:
    def __init__(self):
        params.append(self)
        self.momvalue = None

    def set(self,value):
        self.top = np.float32(value).copy()

### Operations

# Add layer (x + y) where y is same shape as x or is 1-D
class add:
    def __init__(self,x,y):
        ops.append(self)
        self.x = x
        self.y = y

    def forward(self):
        self.top = self.x.top + self.y.top

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + self.grad

        if self.y in ops or self.y in params:
            if len(self.y.top.shape) < len(self.grad.shape):
                ygrad = np.sum(self.grad,axis=tuple(range(len(self.grad.shape)-1)))
            else:
                ygrad= self.grad
            self.y.grad = self.y.grad + ygrad

# Matrix multiply (fully-connected layer)
class matmul:
    def __init__(self,x,y):
        ops.append(self)
        self.x = x
        self.y = y

    def forward(self):
        self.top = np.matmul(self.x.top,self.y.top)

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + np.matmul(self.y.top,self.grad.T).T
        if self.y in ops or self.y in params:
            self.y.grad = self.y.grad + np.matmul(self.x.top.T,self.grad)


# Rectified Linear Unit Activation            
class RELU:
    def __init__(self,x):
        ops.append(self)
        self.x = x

    def forward(self):
        self.top = np.maximum(self.x.top,0)

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + self.grad * (self.top > 0)


# Reduce to mean
class mean:
    def __init__(self,x):
        ops.append(self)
        self.x = x

    def forward(self):
        self.top = np.mean(self.x.top)

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + self.grad*np.ones_like(self.x.top) / np.float32(np.prod(self.x.top.shape))



# Soft-max + Loss (per-row / training example)
class smaxloss:
    def __init__(self,pred,gt):
        ops.append(self)
        self.x = pred
        self.y = gt

    def forward(self):
        y = self.x.top
        y = y - np.amax(y,axis=1,keepdims=True)
        yE = np.exp(y)
        yS = np.sum(yE,axis=1,keepdims=True)
        y = y - np.log(yS); yE = yE / yS

        truey = np.int64(self.y.top)
        self.top = -y[range(len(truey)),truey]
        self.save = yE

    def backward(self):
        if self.x in ops or self.x in params:
            truey = np.int64(self.y.top)
            self.save[range(len(truey)),truey] = self.save[range(len(truey)),truey] - 1.
            self.x.grad = self.x.grad + np.expand_dims(self.grad,-1)*self.save
        # No backprop to labels!    

# Compute accuracy (for display, not differentiable)        
class accuracy:
    def __init__(self,pred,gt):
        ops.append(self)
        self.x = pred
        self.y = gt

    def forward(self):
        truey = np.int64(self.y.top)
        self.top = np.float32(np.argmax(self.x.top,axis=1)==truey)

    def backward(self):
        pass


# Downsample by 2    
class down2:
    def __init__(self,x):
        ops.append(self)
        self.x = x
        
    def forward(self):
        self.top = self.x.top[:,::2,::2,:]

    def backward(self):
        if self.x in ops or self.x in params:
            grd = np.zeros_like(self.x.top)
            grd[:,::2,::2,:] = self.grad
            self.x.grad = self.x.grad + grd


# Flatten (conv to fc)
class flatten:
    def __init__(self,x):
        ops.append(self)
        self.x = x
        
    def forward(self):
        self.top = np.reshape(self.x.top,[self.x.top.shape[0],-1])

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + np.reshape(self.grad,self.x.top.shape)
            
# Convolution Layer
## Fill this out
class conv2:

    def __init__(self,x,k):
        ops.append(self)
        self.x = x
        self.k = k

    def forward(self):
        batch_size = self.x.top.shape[0]
        filters = self.k.top.shape[3]

        # Broadcast the x and k
        hatx = np.zeros(shape=[self.x.top.shape[0], self.x.top.shape[1], self.x.top.shape[2], self.x.top.shape[3],
                               filters], dtype=np.float32)
        hatk = np.zeros(shape=[batch_size, self.k.top.shape[0], self.k.top.shape[1], self.k.top.shape[2],
                               self.k.top.shape[3]], dtype=np.float32)

        for i in range(batch_size):
            hatk[i, :, :, :, :] = self.k.top
        for i in range(filters):
            hatx[:, :, :, :, i] = self.x.top

        # Loop the width and height of kernel
        width_k = self.k.top.shape[0]
        height_k = self.k.top.shape[1]

        width_x = self.x.top.shape[1]
        height_x = self.x.top.shape[2]

        img = 0
        for i in range(width_k):
            for j in range(height_k):
                hatk_cur = hatk[:, i, j, :, :]
                hatk_cur.shape = [hatk_cur.shape[0], 1, 1, hatk_cur.shape[1], hatk_cur.shape[2]]
                hatx_cur = hatx[:, i:width_x-width_k+i+1, j:height_x-height_k+j+1, :, :]
                img += np.sum(hatk_cur * hatx_cur, 3)

        self.top = img

        pass

    def backward(self):
        if self.x in ops or self.x in params:

            grad_pad = np.pad(self.grad, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant')

            flip_kernel = np.flipud(np.fliplr(self.k.top))
            flip_kernel = flip_kernel.swapaxes(2, 3)

            batch_size = grad_pad.shape[0]
            filters = flip_kernel.shape[3]

            hatx = np.zeros(shape=[grad_pad.shape[0], grad_pad.shape[1], grad_pad.shape[2], grad_pad.shape[3],
                                   filters], dtype=np.float32)
            hatk = np.zeros(shape=[batch_size,flip_kernel.shape[0], flip_kernel.shape[1], flip_kernel.shape[2],
                                   flip_kernel.shape[3]], dtype=np.float32)

            for i in range(batch_size):
                hatk[i, :, :, :, :] = flip_kernel
            for i in range(filters):
                hatx[:, :, :, :, i] = grad_pad

            # Loop the width and height of kernel
            width_k = flip_kernel.shape[0]
            height_k = flip_kernel.shape[1]

            width_x = grad_pad.shape[1]
            height_x = grad_pad.shape[2]

            img = 0
            for i in range(width_k):
                for j in range(height_k):
                    hatk_cur = hatk[:, i, j, :, :]
                    hatk_cur.shape = [hatk_cur.shape[0], 1, 1, hatk_cur.shape[1], hatk_cur.shape[2]]
                    hatx_cur = hatx[:, i:width_x - width_k + i + 1, j:height_x - height_k + j + 1, :, :]
                    img += np.sum(hatk_cur * hatx_cur, 3)

            self.x.grad = self.x.grad + img

        if self.k in ops or self.k in params:


            x_filters = self.x.top.shape[3]
            g_filters = self.grad.shape[3]

            # Broadcast the x and k
            hatx = np.zeros(shape=[self.x.top.shape[0], self.x.top.shape[1], self.x.top.shape[2], self.x.top.shape[3],
                                   g_filters], dtype=np.float32)
            hatg = np.zeros(shape=[self.grad.shape[0], self.grad.shape[1], self.grad.shape[2], x_filters,
                                   self.grad.shape[3]], dtype=np.float32)

            for i in range(g_filters):
                hatx[:, :, :, :, i] = self.x.top
            for i in range(x_filters):
                hatg[:, :, :, i, :] = self.grad

            # Loop the width and height of kernel
            width_k = self.grad.shape[1]
            height_k = self.grad.shape[2]

            width_x = self.x.top.shape[1]
            height_x = self.x.top.shape[2]

            img = 0
            for i in range(width_k):
                for j in range(height_k):
                    hatg_cur = hatg[:, i, j, :, :]
                    hatg_cur.shape = [hatg_cur.shape[0], 1, 1, hatg_cur.shape[1], hatg_cur.shape[2]]

                    hatx_cur = hatx[:, i:width_x - width_k + i + 1, j:height_x - height_k + j + 1, :, :]
                    img += np.sum(hatg_cur * hatx_cur, 0)

            self.k.grad = self.k.grad + img

            pass
