import os
from typing import Sequence
import tensorflow as tf
from tensorflow.keras.layers.experimental import RandomFourierFeatures
class ModelGen:
    def __init__(self):
        self.train_in = []
        self.train_out = []
        self.model = None
        pass

    model_path = "./models"

    def initialize_dataset(self, train_in, train_out):
        self.train_in = train_in
        self.train_out = train_out
        return 0

    def train_model(self, epochs=20, optimizer='adam', loss='binary_crossentropy', metrics=['acc']):
        self.model.compile(optimizer, loss, metrics)
        history = self.model.fit(self.train_in, self.train_out, epochs=epochs)
        return history 

    def import_model(self, model):
        self.model = model
        pass

    def define_model_SVM(self, features_dim=4096):
        self.model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(4,)),
                RandomFourierFeatures(
                    output_dim=features_dim, scale=1, kernel_initializer="laplacian"
                ),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ]
        )
        return 0

    def define_model_ANN(self, layers):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(layers[0], input_shape=(4,), activation='relu'))
        for num in layers[1:-1]:
            self.model.add(tf.keras.layers.Dense(num, activation='relu'))
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        return 0

    def save_model(self, filename):
        self.model.save(self.model_path + "/" + filename + ".h5")
        return 0