
def prepare_dataset():
    #load dataset
    global X_train, X_test, y_train, y_test
    X_train, y_train = loadMnist("../data/")
    X_test, y_test = loadMnist("../data/", kind='test')
    #chia 255, pixel value tu [0, 255] sang [0, 1]
    X_train = X_train/255.0
    X_test = X_test/255.0
    #format lai input cho phu hop keras
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    y_train = tf.keras.utils.to_categorical (y_train)
    y_test = tf.keras.utils.to_categorical (y_test)
    
def build_model():
    #define kich co anh
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
    
    """x = Dense (16, kernel_initializer = random_normal())(x)
    x = Activation("relu")(x)"""

    x = Dense (10, kernel_initializer = random_normal())(x)
    x = Activation("softmax")(x)
    return Model(inputs, x)

def train_model():
    opt = tf.keras.optimizers.SGD(0.0003)
    metrics = ["acc"]
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=metrics)
    #Train 20 lan, moi lan 20 epoch.
    for i in range (0, 20):
        model.fit (X_train, y_train, batch_size=32, epochs=40, verbose=2, validation_data=(X_test, y_test))
        model.save ("gaymodel2.h5")
        print("Model saved")

    
import os
from main_Gia_modified import *
import tensorflow as tf
#Dung neu tensorflow an het vram cua GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
  
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import random_normal

prepare_dataset()
#uncomment de tao model moi
model = build_model()
#load model cu len, neu tao model moi thi comment dong nay
#model = load_model('gaymodel2.h5')
#Train model
train_model()

#output cua model2 la numpy array size: (1, 256)
model2= Model(model.input,model.get_layer('flatten').output)
model2.summary()
print (model2.predict (X_test[50][np.newaxis,...]).shape)
print (model2.predict (X_test[50][np.newaxis,...]))
