#!/usr/bin/env python
import sys
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(8,6))
fig.patch.set_color('white')

epoch = range(1,11)
accuracy_train = [0.4750, 0.5100, 0.6050, 0.6300, 0.6000, 0.5500, 0.5500, 0.5500, 0.6700, 0.5950]
plt.plot(epoch, accuracy_train, color='b', alpha=0.8,  linestyle='-', label='Train')
plt.scatter(10, 0.5684, color='r', label='Test')
plt.xlim(0,11)
plt.ylim(0.4,0.7)
plt.grid(linestyle='--')
plt.xlabel("epoch")
plt.ylabel("Accuracy")
plt.legend(loc='best')
plt.savefig("./train_result.png")
plt.show()

