def build_model():
    size = 28
    inputs = Input((size, size, 1))

    skip_x = []
    x = inputs
    x = Conv2D(16, (3, 3), padding="valid",kernel_initializer = random_normal())(x)
    x = Activation("relu")(x)
    
    x = Conv2D(16, (3, 3), padding="valid",kernel_initializer = random_normal())(x)
    x = Activation("relu")(x)
    #x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D((3, 3))(x)
    x = Conv2D(16, (3, 3), padding="valid",kernel_initializer = random_normal())(x)
    x = Activation("relu")(x)
    
    x = Conv2D(16, (3, 3), padding="valid",kernel_initializer = random_normal())(x)
    x = Activation("relu")(x)
    #x = BatchNormalization(axis=3)(x)

    x = Flatten ()(x)
    x = Dense (16, kernel_initializer = random_normal())(x)
    x = Activation("relu")(x)

    x = Dense (10)(x)
    x = Activation("softmax")(x)
    return Model(inputs, x)

def train_model():
    opt = tf.keras.optimizers.SGD(0.001)
    metrics = ["acc"]
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=metrics)
    for i in range (0, 5):
        model.fit (X_train, y_train, batch_size=32, epochs=20, verbose=2, validation_data=(X_test, y_test))
        model.save ("gaymodel2.h5")

import os
from main_Gia_modified import *
import tensorflow as tf

#Dung neu tensorflow an het vram cua GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

#load + format lai dataset
X_train, y_train = loadMnist("../data/")
X_test, y_test = loadMnist("../data/", kind='test')
#chia 255, pixel value tu [0, 255] sang [0, 1]
X_train = X_train/255.0
X_test = X_test/255.0
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]
y_train = tf.keras.utils.to_categorical (y_train)
y_test = tf.keras.utils.to_categorical (y_test)

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import random_normal
model = load_model('gaymodel2.h5')
#uncomment hai dong nay de train model
model = build_model()
train_model()

#output cua model2 la numpy array size: (1, 256)
model2= Model(model.input,model.get_layer('flatten').output)
model2.summary()
print (model2.predict (X_test[50][np.newaxis,...]).shape)
print (model2.predict (X_test[50][np.newaxis,...]))
