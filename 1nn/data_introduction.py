import pickle as pk
import os
from matplotlib.pyplot import imshow

CIFAR_DIR = "../cifar-10-batches-py"
print(os.listdir(CIFAR_DIR))

with open(os.path.join(CIFAR_DIR,'data_batch_1'),'rb') as f:
    data = pk.load(f,encoding='bytes')
    # print(data)

image_arr = data[b'data'][100]
image_arr = image_arr.reshape((3,32,32))
image_arr = image_arr.transpose((1,2,0))

imshow(image_arr)