import os
os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.regularizers import l1
import numpy as np
import sys

from utils import load_data, plot

class Preprocessor:
    def process(self, x):
        # Normalise to [0, 1]
        x = x.astype("float32") / 255.0

        # Flatten
        x = x.reshape(-1, 28 * 28)
        return x

    def revert(self, x):
        x = x.reshape(-1, 28, 28)
        x *= 255.0
        return x

if __name__=="__main__":
    np.random.seed(1969)

    (x_train, y_train), (x_test, y_test) = load_data()

    preprocessor = Preprocessor()
    x_train = preprocessor.process(x_train)

    input_size = 784
    hidden_size = 128
    code_size = 32

    input_img = Input(shape=(input_size,))
    hidden_1 = Dense(hidden_size, activation="relu")(input_img)
    code = Dense(code_size, activation="relu")(hidden_1)
    hidden_2 = Dense(hidden_size, activation="relu")(code)
    output_img = Dense(input_size, activation="sigmoid")(hidden_2)

    autoencoder = Model(input_img, output_img)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    autoencoder.fit(x_train, x_train, epochs=3)

    x_test = preprocessor.process(x_test)
    x_test_reconstructed = autoencoder.predict(x_test)
    print(x_test.shape)
    print(x_test_reconstructed.shape)
    plot(x_test, x_test_reconstructed, preprocessor)
