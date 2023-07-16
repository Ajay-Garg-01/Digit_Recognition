import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_indices = np.random.permutation(len(x_train))
test_indices = np.random.permutation(len(x_test))

x_train = x_train[train_indices]
y_train = y_train[train_indices]
x_test = x_test[test_indices]
y_test = y_test[test_indices]


x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


random_index = np.random.randint(0, len(x_train))
random_x = x_train[random_index]
random_y = y_train[random_index]

model.fit(np.array([random_x]), np.array([random_y]), epochs=1)  


x_train_rest = np.delete(x_train, random_index, axis=0)
y_train_rest = np.delete(y_train, random_index, axis=0)
model.fit(x_train_rest, y_train_rest, epochs=50)
model.save("handwritten.model")

# Evaluate the model on the testing data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")