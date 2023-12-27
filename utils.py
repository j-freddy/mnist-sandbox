from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)

    return (x_train, y_train), (x_test, y_test)

def plot(x, x_reconstructed, preprocessor):
    x = preprocessor.revert(x)
    x_reconstructed = preprocessor.revert(x_reconstructed)

    num_samples = 6
    idx = np.random.randint(0, x.shape[0], num_samples)

    fig, axs = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))

    for i in range(num_samples):
        axs[0][i].imshow(x[idx[i]], cmap="gray")
        axs[1][i].imshow(x_reconstructed[idx[i]], cmap="gray")
    
    # Remove axis ticks
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    axs[0][0].set_ylabel("Original")
    axs[1][0].set_ylabel("Reconstructed")

    plt.show()
