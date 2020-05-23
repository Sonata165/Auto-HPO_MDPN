import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.optimizers import Adam


# Generate dummy data
x_train = np.random.random((100, 100, 100, 3))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
x_test = np.random.random((20, 100, 100, 3))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

# 19 hyper-parameters in all, but 4 of them I don't know what are they, so in fact 15 in all TODO
c1_channel = 3
c1_kernel = 3
c1_size2 = 2 # ？？？
c1_size3 = 2 # ？？？
c2_channel = 3
c2_kernel = 3
c2_size2 = 2 # ？？？
c2_size3 = 2 # ？？？
p1_type = 'max' # Pooling Type (max / avg)
p1_kernel = 2 # kernel size
p1_stride = 2 # stride size
p2_type = 'max'
p2_kernel = 2
p2_stride = 2
n1 = 36 # hidden layer size
n2 = 24
n3 = 16
n4 = 8
learn_rate = 0.001

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(filters=c1_channel, kernel_size=c1_kernel, activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(filters=c1_channel, kernel_size=c1_kernel, activation='relu'))
model.add(Conv2D(filters=c1_channel, kernel_size=c1_kernel, activation='relu'))
model.add(MaxPooling2D(pool_size=p1_kernel, strides=p1_stride))

model.add(Conv2D(filters=c2_channel, kernel_size=c2_kernel, activation='relu'))
model.add(Conv2D(filters=c2_channel, kernel_size=c2_kernel, activation='relu'))
model.add(Conv2D(filters=c2_channel, kernel_size=c2_kernel, activation='relu'))
model.add(MaxPooling2D(pool_size=p2_kernel, strides=p2_stride))

model.add(Flatten())
model.add(Dense(n1, activation='relu'))
model.add(Dense(n2, activation='relu'))
model.add(Dense(n3, activation='relu'))
model.add(Dense(n4, activation='relu'))
model.add(Dense(10, activation='softmax'))

adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adam)

model.fit(x_train, y_train, batch_size=32, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=32)