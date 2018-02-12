import keras
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, LocallyConnected2D
from keras import backend as K
from keras.layers import Dense, MaxPool2D, Conv2D, Dropout
from keras.layers import Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils
from keras.initializers import Constant
from keras import applications
from keras import Model
from keras import optimizers
import cv2
import Image
from scipy.misc import toimage
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense

batch_size = 1500
num_classes = 10
epochs = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

def to_3(x):
    y = cv2.merge([x,x,x],x)
    y=  toimage(y)
    y= y.resize((56, 56), Image.NEAREST)
    return np.array(y.getdata(),
                    np.uint8).reshape(y.size[1], y.size[0], 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x_train1 = map(to_3,x_train)
x_test1 = map(to_3,x_test)

x_train = np.asarray(x_train)
x_test = np.asarray(x_test)

input_shape = (img_rows, img_cols, 1)

model_b = applications.VGG16(include_top=False,input_shape = (56,56,3), weights='imagenet')
print('Model loaded.')
top_model = Sequential()
top_model.add(Flatten(input_shape=model_b.output_shape[1:]))
top_model.add(Dense(32, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(num_classes, activation='softmax'))
for layer in model_b.layers:
    layer.trainable = False
model = Model(inputs= model_b.input, outputs= top_model(model_b.output))

datagen = ImageDataGenerator(
    featurewise_center = True,
    featurewise_std_normalization = True,
    width_shift_range = 0.2,
    height_shift_range = True,
    horizontal_flip=True)

datagen.fit(x_train)


model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

              model.fit_generator(
                      datagen.flow(x_train,y_train,batch_size = batch_size),

                      steps_per_epoch=len(x_train)/batch_size,

                      epochs=epochs,
                      validation_data=(x_test,y_test),
                      verbose=1)
preds = model.predict(x_test)
preds = np.argmax(preds, 1)

with open('submission.csv', 'w') as f:
  f.write('Id,Class\n')
  for i, c in enumerate(preds):
    f.write('{},{}\n'.format(i, c))


model.save('fine.h5')    
