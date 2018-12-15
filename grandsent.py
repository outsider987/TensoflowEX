import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

REAL_PARMETER = [2,1]
TF_PARMETER = [[5,2],[8,5],[120,50]] [0]


x = np.linspace(-1,1,200)

y_fun = lambda a,b: a * x + b
tf_y_fun= lambda a,b: a * x + b

print(REAL_PARMETER)
noise = np.random.randn(200)/10
y = y_fun(*REAL_PARMETER) + noise
plt.scatter(x,y)
plt.show()
plt.waitforbuttonpress()
