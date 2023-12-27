import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
from sklearn.decomposition import PCA
import sys

from utils import load_data, plot

class Preprocessor:
    def __init__(self, x_train):
        self.x_train = x_train
        self.x_mean = None

    def process_train(self):
        return self.process(self.x_train, save_mean=True)
    
    def process(self, x, save_mean=False):
        # Flatten
        x = x.reshape(-1, 28 * 28)

        # Mean centering
        x = x.astype("float32")
        if save_mean:
            self.x_mean = x.mean(axis=0)
        x -= self.x_mean
        return x

    def revert(self, x):
        x += self.x_mean
        x = x.reshape(-1, 28, 28)
        return x

if __name__=="__main__":
    np.random.seed(1969)

    (x_train, y_train), (x_test, y_test) = load_data()

    preprocessor = Preprocessor(x_train)
    x_train = preprocessor.process_train()

    # Train PCA model
    pca = PCA(n_components=32)
    pca.fit(x_train)
    
    # Test PCA model
    x_test = preprocessor.process(x_test)
    x_test_representation = pca.transform(x_test)
    x_test_reconstructed = pca.inverse_transform(x_test_representation)

    plot(x_test, x_test_reconstructed, preprocessor)
