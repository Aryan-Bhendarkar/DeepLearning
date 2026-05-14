import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU  # type: ignore

class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, filters, stride=1):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.conv1 = Conv2D(filters, kernel_size=(3,3), strides=stride, padding='same')
        self.bn1 = BatchNormalization()
        self.relu = ReLU()
        self.conv2 = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')
        self.bn2 = BatchNormalization()

        self.projection = Conv2D(filters, kernel_size=(1, 1),strides= stride,padding='same')
        self.projection_bn = BatchNormalization()
        
    def call(self, x, training=False):
        x_original = x
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        if self.stride != 1:
            x_original = self.projection(x_original)
            x_original = self.projection_bn(x_original)

        return x + x_original