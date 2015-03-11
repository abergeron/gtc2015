import os
import numpy as np

# this generates random data which is bad for training, but it can
# validate that the data loading code is working.

# make directories
os.makedirs('train')
os.makedirs('val')
os.makedirs('labels')
os.makedirs('models')

# test
img_sum = np.zeros((3, 256, 256))

for i in range(10):
    img = np.random.randint(0, 256, (3, 256, 256, 256)).astype('uint8')
    img_sum += img.mean(axis=3)
    np.save('train/%04d.npy' % (i,), img)

img_sum /= 10

np.save('img_mean.npy', img_sum)

# valid
for i in range(10):
    img = np.random.randint(0, 256, (3, 256, 256, 256)).astype('uint8')
    np.save('val/%04d.npy' % (i,), img)


# labels
train_labels = np.random.randint(0, 100, size=(2560,))
np.save('labels/train_labels.npy', train_labels)
val_labels = np.random.randint(0, 100, size=(2560,))
np.save('labels/val_labels.npy', val_labels)

