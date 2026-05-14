import tensorflow as tf 
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, ReLU, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras import Sequential

from resnet_block import ResidualBlock

class ResNet18(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        # Stage 0: Conv(7x7, 64, stride=2) → BN → ReLU → MaxPool
        self.layer1 = Conv2D(64, kernel_size=(7, 7), strides=2, padding='same')
        self.bn1 = BatchNormalization()
        self.relu = ReLU()
        self.maxpool = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')

        # Stage 1:
        self.stage1 = Sequential([
            ResidualBlock(64, stride=1),
            ResidualBlock(64, stride=1)
        ])

        # Stage 2:
        self.stage2 = Sequential([
            ResidualBlock(128, stride=2),
            ResidualBlock(128, stride=1)
        ])

        # Stage 3:
        self.stage3 = Sequential([
            ResidualBlock(256, stride=2),
            ResidualBlock(256, stride=1)
        ])

        # Stage 4:
        self.stage4 = Sequential([
            ResidualBlock(512, stride=2),
            ResidualBlock(512, stride=1)
        ])

        # Global Average Pooling 
        self.gap = GlobalAveragePooling2D()

        # Dense layer
        self.fc = Dense(num_classes)


    def call(self, x, training=False):
        x = self.layer1(x)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.stage1(x, training=training )
        x = self.stage2(x, training=training)
        x = self.stage3(x, training=training)
        x = self.stage4(x, training=training)

        x = self.gap(x)
        x = self.fc(x)
        
        return x
