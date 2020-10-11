import numpy as np
np.random.seed(0)

from os.path import normpath as fn
from time import time

import edf ## This will be your code

visualization_loss_train = []
visualization_loss_val = []
visualization_accuracy_train = []
visualization_accuracy_val = []

# Load data
data = np.load(fn('inputs/mnist_26k.npz'))

train_im = np.float32(data['im_train'])/255.-0.5
train_lb = data['lbl_train']

val_im = np.float32(data['im_val'])/255.-0.5
val_lb = data['lbl_val']


#######################################

# Inputs and parameters
inp = edf.Value()
lab = edf.Value()

W1 = edf.Param()
B1 = edf.Param()
W2 = edf.Param()
B2 = edf.Param()

# Model
y = edf.matmul(inp,W1)
y = edf.add(y,B1)
y = edf.RELU(y)

y = edf.matmul(y,W2)
y = edf.add(y,B2) # This is our final prediction


# Cross Entropy of Soft-max
loss = edf.smaxloss(y,lab)
loss = edf.mean(loss)

# Accuracy
acc = edf.accuracy(y,lab)
acc = edf.mean(acc)

###################################

# Init Weights
def xavier(shape):
    sq = np.sqrt(3.0/np.prod(shape[:-1]))
    return np.random.uniform(-sq,sq,shape)


nHidden = 1024

W1.set(xavier((28*28,nHidden)))
B1.set(np.zeros((nHidden)))
W2.set(xavier((nHidden,10)))
B2.set(np.zeros((10)))


# Training loop

BSZ=75
lr=0.001

NUM_EPOCH=50
DISPITER=100
batches = range(0,len(train_lb)-BSZ+1,BSZ)

## Implement Momentum and uncomment following line
# edf.init_momentum()


niter=0; avg_loss = 0.; avg_acc = 0.
for ep in range(NUM_EPOCH+1):

    # As we train, let's keep track of val accuracy
    vacc = 0.; vloss = 0.; viter = 0
    for b in range(0,len(val_lb)-BSZ+1,BSZ):
        inp.set(val_im[b:b+BSZ,:]); lab.set(val_lb[b:b+BSZ])
        edf.Forward()
        viter = viter + 1;vacc = vacc + acc.top;vloss = vloss + loss.top
    vloss = vloss / viter; vacc = vacc / viter * 100
    print("%09d: #### %d Epochs: Val Loss = %.3e, Accuracy = %.2f%%" % (niter,ep,vloss,vacc))

    visualization_loss_val.append(vloss)
    visualization_accuracy_val.append(vacc)

    if ep == NUM_EPOCH:
        break

    # Shuffle Training Set
    idx = np.random.permutation(len(train_lb))

    # Train one epoch
    for b in batches:
        # Load a batch
        inp.set(train_im[idx[b:b+BSZ],:])
        lab.set(train_lb[idx[b:b+BSZ]])

        edf.Forward()
        avg_loss = avg_loss + loss.top; avg_acc = avg_acc + acc.top;
        niter = niter + 1
        if niter % DISPITER == 0:
            avg_loss = avg_loss / DISPITER; avg_acc = avg_acc / DISPITER * 100
            print("%09d: Training Loss = %.3e, Accuracy = %.2f%%" % (niter,avg_loss,avg_acc))

            visualization_loss_train.append(avg_loss)
            visualization_accuracy_train.append(avg_acc)

            avg_loss = 0.; avg_acc = 0.;

        edf.Backward(loss)
        edf.SGD(lr)
        # Replace previous line with following
        # edf.momentum(lr,0.9)

visualization_loss_train = np.array(visualization_loss_train)
visualization_loss_val = np.array(visualization_loss_val)
visualization_accuracy_train = np.array(visualization_accuracy_train)
visualization_accuracy_val = np.array(visualization_accuracy_val)

import scipy.io as sio

sio.savemat('./mnist/sgd_BSZ_%d_lr_%.6f_nHidden_%d.mat' % (BSZ, lr, nHidden),{

    'SltBSZ%dlr3fnH%d' % (BSZ, nHidden): visualization_loss_train,
    'SlvBSZ%dlr3fnH%d' % (BSZ, nHidden): visualization_loss_val,
    'SatBSZ%dlr3fnH%d' % (BSZ, nHidden): visualization_accuracy_train,
    'SavBSZ%dlr3fnH%d' % (BSZ, nHidden): visualization_accuracy_val

})