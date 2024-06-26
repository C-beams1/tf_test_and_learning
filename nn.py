import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape)
# print(y_train.shape)
x_train = x_train.reshape(-1,28*28).astype('float32') / 255.0
x_test = x_test.reshape(-1,28*28).astype('float32') / 255.0

# Sequential API (Very convenient, not very flexible)
model = keras.Sequential(
    [
        keras.Input(shape=(28*28)),
        layers.Dense(512,activation='relu'),
        layers.Dense(256,activation='relu'),
        layers.Dense(10),
    ]
)

# print(model.summary())
# import sys
# sys.exit(0)

# functional API(a bit more flexible)
inputs = keras.Input(shape=(784))
x = layers.Dense(512,activation='relu')(inputs)
x = layers.Dense(256,activation='relu')(x)
outputs = layers.Dense(10,activation='softmax')(x)

model = keras.Model(inputs=inputs,outputs=outputs)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer = keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)

model.fit(x_train,y_train,batch_size=32,epochs =5,verbose=2)
model.evaluate(x_test,y_test,batch_size=32,verbose=2)

