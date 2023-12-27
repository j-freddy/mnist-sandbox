from dotenv import load_dotenv
load_dotenv()

from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import sys

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


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)

    return (x_train, y_train), (x_test, y_test)

def plot(x, x_reconstructed, idx, preprocessor):
    x = preprocessor.revert(x)
    x_reconstructed = preprocessor.revert(x_reconstructed)

    fig, axs = plt.subplots(2, 6, figsize=(12, 4))

    for i in range(6):
        axs[0][i].imshow(x[idx[i]], cmap="gray")
        axs[1][i].imshow(x_reconstructed[idx[i]], cmap="gray")
    
    # Remove axis ticks
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    axs[0][0].set_ylabel("Original")
    axs[1][0].set_ylabel("Reconstructed")

    plt.show()

if __name__=="__main__":
    np.random.seed(1969)

    (x_train, y_train), (x_test, y_test) = load_data()

    preprocessor = Preprocessor(x_train)
    x_train = preprocessor.process_train()

    # Train PCA model
    pca = PCA(n_components=50)
    pca.fit(x_train)
    
    # Test PCA model
    x_test = preprocessor.process(x_test)
    x_test_representation = pca.transform(x_test)
    x_test_reconstructed = pca.inverse_transform(x_test_representation)

    idx = np.random.randint(0, x_test.shape[0], 6)
    plot(x_test, x_test_reconstructed, idx, preprocessor)
