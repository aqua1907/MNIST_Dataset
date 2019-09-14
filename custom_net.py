import tensorflow as tf
from tensorflow import keras
from keras.utils import plot_model
from os.path import split, realpath

class CustomNet:
    @staticmethod
    def build(height, width, depth, num_classes=None):
        he = keras.initializers.he_normal(seed=9)

        inputShape = (height, width, depth)
        chanDim = -1

        if keras.backend.image_data_format() == 'channel_first':
            inputShape = (depth, height, width)
            chanDim = 1


        model = keras.models.Sequential([
            keras.layers.Conv2D(32, 3, padding='same', activation='relu', kernel_initializer=he, 
                                input_shape=inputShape, name="conv1_input"),
            keras.layers.BatchNormalization(axis=chanDim),
            keras.layers.MaxPooling2D((3, 3), (2, 2)),
            keras.layers.Dropout(rate=0.25),
            keras.layers.Conv2D(64, 3, padding='same', activation='relu',kernel_initializer=he),
            keras.layers.BatchNormalization(axis=chanDim),
            keras.layers.Conv2D(64, 3, padding='same', activation='relu', kernel_initializer=he),
            keras.layers.BatchNormalization(axis=chanDim),
            keras.layers.MaxPooling2D((3, 3), (2, 2)),
            keras.layers.Dropout(rate=0.25),
            keras.layers.Conv2D(128, 3, padding='same', activation='relu', kernel_initializer=he),
            keras.layers.BatchNormalization(axis=chanDim),
            keras.layers.Conv2D(128, 3, padding='same', activation='relu', kernel_initializer=he),
            keras.layers.BatchNormalization(axis=chanDim),
            keras.layers.Conv2D(128, 3, padding='same', activation='relu', kernel_initializer=he),
            keras.layers.BatchNormalization(axis=chanDim),
            keras.layers.MaxPooling2D((3, 3), (2, 2)),
            keras.layers.Dropout(rate=0.25),
            keras.layers.Flatten(),
            keras.layers.Dense(256, 'relu', kernel_initializer=he),
            keras.layers.Dense(128, 'relu', kernel_initializer=he),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(rate=0.5),
            keras.layers.Dense(num_classes, 'softmax', kernel_initializer=he),
            ])
        
        return model

model = CustomNet.build(28, 28, 1, num_classes=27)
            
# plot_model(model, to_file=split(realpath(__file__))[0] + "\CustomNet.png", show_shapes=True)

