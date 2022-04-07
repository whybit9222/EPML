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

    model_path = "./"

    def initialize_dataset(self, train_in, train_out):
        self.train_in = train_in
        self.train_out = train_out
        pass

    def train_model(self, epochs=20, optimizer='adam', loss='binary_crossentropy', metrics=['acc']):
        self.model.compile(optimizer, loss, metrics)
        history = self.model.fit(self.train_in, self.train_out, epochs)
        return history 

    def import_model(self, model):
        self.model = model
        pass

    def define_model_SVM(self, features_dim=4096):
        self.model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(4,)),
                RandomFourierFeatures(
                    features_dim, scale=1, kernel_initializer="laplacian"
                ),
            ]
        )
        pass

    def define_model_ANN(self, layers):
        self.model = tf.keras.Sequential()

        for idx, num in enumerate(layers):
            if idx==0:
                self.model.add(tf.keras.layers.Dense(num, input_shape=(4,), activation='relu'))
            elif idx!=len(layers-1):
                self.model.add(tf.keras.layers.Dense(num, activation='relu'))
            else:
                self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    pass

    def save_model(self, filename):
        self.model.save(filename + ".h5")
        pass