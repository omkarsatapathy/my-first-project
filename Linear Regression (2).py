import tensorflow as tf
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from tensorflow import keras
    
model = tf.keras.Sequential(
[keras.layers.Dense(units=1, input_shape=[1],activation = "sigmoid")
    ])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.random.randn(100)
ys = np.arange(0,100)
for i in range (0, 100):
        ys[i] = ((xs[i]*3 +5))
        i+=1
history = model.fit(xs, ys, epochs=1000)
print(model.predict([6.0]))
pd.DataFrame(history.history).plot(figsize= (8, 5))
plt.grid(True)
plt.gca().set_ylim(0.1)
plt.show()
