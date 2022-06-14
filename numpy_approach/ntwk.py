import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np



# use keras API
model = tf.keras.Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(8,)))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss = "MeanSquaredError", metrics=['accuracy'])

#Make up some data
d = np.identity(100)
d = np.concatenate((d, d[-10:, :]), axis = 0)
dummy_data = np.concatenate((d, np.ones((110, 1))), axis = 1)

X_train = dummy_data[0:100, 0:8]
y_train = dummy_data[0:100, 8]
X_test = dummy_data[100:110, 0:8]
y_test = dummy_data[100:110, 8]

model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=1)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)
print('Loss:', loss)

yhat = model.predict(np.ones((1, 8)))
print('Predicted: %.3f' % yhat)

