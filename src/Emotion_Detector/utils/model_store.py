import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, MaxPool2D, Dense, Dropout, BatchNormalization, Layer,MaxPooling2D,
                                     Flatten, InputLayer, Resizing, Rescaling, GlobalAveragePooling2D, Add, Activation)
from tensorflow.keras.models import Model
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
    

class CustomConv2D(Layer):
  def __init__(self, n_filters, kernel_size, n_strides, padding = 'valid'):
    super(CustomConv2D, self).__init__(name = 'custom_conv2d')

    self.conv = Conv2D(
        filters = n_filters,
        kernel_size = kernel_size,
        activation = 'relu',
        strides = n_strides,
        padding = padding)
    
    self.batch_norm = BatchNormalization()

  def call(self, x, training = True):

    x = self.conv(x)
    x = self.batch_norm(x, training)

    return x
    
class ResidualBlock(Layer):
  def __init__(self, n_channels, n_strides = 1):
    super(ResidualBlock, self).__init__(name = 'res_block')

    self.dotted = (n_strides != 1)

    self.custom_conv_1 = CustomConv2D(n_channels, 3, n_strides, padding = "same")
    self.custom_conv_2 = CustomConv2D(n_channels, 3, 1, padding = "same")

    self.activation = Activation('relu')

    if self.dotted:
      self.custom_conv_3 = CustomConv2D(n_channels, 1, n_strides)
    
  def call(self, input, training):

    x = self.custom_conv_1(input, training)
    x = self.custom_conv_2(x, training)

    if self.dotted:
      x_add = self.custom_conv_3(input, training)
      x_add = Add()([x, x_add])
    else:
      x_add = Add()([x, input])

    return self.activation(x_add)


    
class ResNet34(Model):
  def __init__(self,):
    super(ResNet34, self).__init__(name = 'resnet_34')
    
    self.conv_1 = CustomConv2D(64, 7, 2, padding = 'same')
    self.max_pool = MaxPooling2D(3,2)
    
    self.conv_2_1 = ResidualBlock(64)
    self.conv_2_2 = ResidualBlock(64)
    self.conv_2_3 = ResidualBlock(64)
    
    self.conv_3_1 = ResidualBlock(128, 2)
    self.conv_3_2 = ResidualBlock(128)
    self.conv_3_3 = ResidualBlock(128)
    self.conv_3_4 = ResidualBlock(128)

    self.conv_4_1 = ResidualBlock(256, 2)
    self.conv_4_2 = ResidualBlock(256)
    self.conv_4_3 = ResidualBlock(256)
    self.conv_4_4 = ResidualBlock(256)
    self.conv_4_5 = ResidualBlock(256)
    self.conv_4_6 = ResidualBlock(256)
    
    self.conv_5_1 = ResidualBlock(512, 2)
    self.conv_5_2 = ResidualBlock(512)
    self.conv_5_3 = ResidualBlock(512)

    self.global_pool = GlobalAveragePooling2D()

    self.fc_3 = Dense(7, activation = 'softmax')
    
  def call(self, x, training = True):
    x = self.conv_1(x)
    x = self.max_pool(x)

    x = self.conv_2_1(x, training)
    x = self.conv_2_2(x, training)
    x = self.conv_2_3(x, training)
    
    x = self.conv_3_1(x, training)
    x = self.conv_3_2(x, training)
    x = self.conv_3_3(x, training)
    x = self.conv_3_4(x, training)
    
    x = self.conv_4_1(x, training)
    x = self.conv_4_2(x, training)
    x = self.conv_4_3(x, training)
    x = self.conv_4_4(x, training)
    x = self.conv_4_5(x, training)
    x = self.conv_4_6(x, training)
    
    x = self.conv_5_1(x, training)
    x = self.conv_5_2(x, training)
    x = self.conv_5_3(x, training)

    x = self.global_pool(x)
    
    return self.fc_3(x)