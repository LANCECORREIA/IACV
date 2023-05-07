import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import flwr as fl
import tensorflow as tf
import logging
logging.basicConfig(level=logging.DEBUG)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Add a channel dimension
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

y_train_resized = tf.image.resize(x_train, (32, 32))
x_train_resized = []
for idx in range(len(y_train_resized)):
  img = y_train_resized[idx].numpy()
  y = img.copy()
  y[14:20, 14:20] = 1.0

  x_train_resized.append(y)
x_test_resized = tf.image.resize(x_test, (32, 32))

x_train_resized = np.array(x_train_resized)

model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1)),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'),
])

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=["accuracy"])

# Define a function to train the model on a client
class MyClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train_resized, y_train_resized, validation_data = (x_test_resized,x_test_resized),epochs=1, batch_size=128)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test_resized, x_test_resized)
        return loss, len(x_test), {"accuracy": float(accuracy)}

# Create a Flower client
client = MyClient()

# Connect the client to the Flower server
fl.client.start_numpy_client(server_address="127.0.0.1:8000", client=client)
