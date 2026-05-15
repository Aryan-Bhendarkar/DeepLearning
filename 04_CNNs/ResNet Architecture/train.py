import tensorflow as tf
from resnet18 import ResNet18
from dataset import load_cifar10
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy


(X_train, y_train), (X_test, y_test) = load_cifar10()

model = ResNet18(num_classes = 10)

model.compile(optimizer=Adam(learning_rate=0.001), loss = SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=64, epochs=20, validation_data=(X_test, y_test))

