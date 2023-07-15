import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, BatchNormalization, Flatten, InputLayer, Resizing, Rescaling
from tensorflow.keras.regularizers import L2


class LenetModel(tf.keras.Model):
    def __init__(self, configurations):
        super(LenetModel, self).__init__()

        self.resize = Resizing(configurations['IMAGE_SIZE'], configurations['IMAGE_SIZE'])

        self.rescale = Rescaling(1./255)

        self.conv1 = Conv2D(filters=configurations['N_FILTERS'], kernel_size=configurations['KERNAL_SIZE'],
                            strides=configurations['N_STRIDES'], padding='valid', activation='relu',
                            kernel_regularizer=L2(configurations['REGULARIZATION_RATE']))
        self.batch_norm1 = BatchNormalization()
        self.max_pool1 = MaxPool2D(pool_size=configurations['POOL_SIZE'], strides=configurations['N_STRIDES'] * 2)
        self.dropout1 = Dropout(rate=configurations['DROPOUT_RATE'])

        self.conv2 = Conv2D(filters=configurations['N_FILTERS'] * 2 + 4, kernel_size=configurations['KERNAL_SIZE'],
                            strides=configurations['N_STRIDES'], padding='valid', activation='relu',
                            kernel_regularizer=L2(configurations['REGULARIZATION_RATE']))
        self.batch_norm2 = BatchNormalization()
        self.max_pool2 = MaxPool2D(pool_size=configurations['POOL_SIZE'], strides=configurations['N_STRIDES'] * 2)

        self.flatten = Flatten()

        self.dense1 = Dense(configurations['N_DENSE_1'], activation='relu',
                            kernel_regularizer=L2(configurations['REGULARIZATION_RATE']))
        self.batch_norm3 = BatchNormalization()

        self.dense2 = Dense(configurations['N_DENSE_2'], activation='relu',
                            kernel_regularizer=L2(configurations['REGULARIZATION_RATE']))
        self.batch_norm4 = BatchNormalization()
        self.dropout2 = Dropout(rate=configurations['DROPOUT_RATE'])

        self.dense3 = Dense(configurations['NUM_CLASSES'], activation='softmax')

    def call(self, inputs):
        x = self.resize(inputs)
        x = self.rescale(x)

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.max_pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.max_pool2(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.batch_norm3(x)

        x = self.dense2(x)
        x = self.batch_norm4(x)
        x = self.dropout2(x)

        output = self.dense3(x)

        return output