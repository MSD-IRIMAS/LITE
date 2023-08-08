import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import os
import time
import json

from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.metrics import accuracy_score
from codecarbon import track_emissions


class LITE:
    def __init__(
        self,
        output_directory,
        length_TS,
        n_classes,
        batch_size=64,
        n_filters=32,
        kernel_size=41,
        n_epochs=1500,
        verbose=True,
        use_custom_filters=True,
        use_dilation=True,
        use_multiplexing=True,
    ):

        self.output_directory = output_directory

        self.length_TS = length_TS
        self.n_classes = n_classes

        self.verbose = verbose

        self.n_filters = n_filters

        self.use_custom_filters = use_custom_filters
        self.use_dilation = use_dilation
        self.use_multiplexing = use_multiplexing

        self.kernel_size = kernel_size - 1

        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.build_model()

    def hybird_layer(
        self, input_tensor, input_channels, kernel_sizes=[2, 4, 8, 16, 32, 64]
    ):

        conv_list = []

        for kernel_size in kernel_sizes:

            filter_ = np.ones(shape=(kernel_size, input_channels, 1))
            indices_ = np.arange(kernel_size)

            filter_[indices_ % 2 == 0] *= -1

            conv = tf.keras.layers.Conv1D(
                filters=1,
                kernel_size=kernel_size,
                padding="same",
                use_bias=False,
                kernel_initializer=tf.keras.initializers.Constant(filter_),
                trainable=False,
                name="hybird-increasse-"
                + str(self.keep_track)
                + "-"
                + str(kernel_size),
            )(input_tensor)

            conv_list.append(conv)

            self.keep_track += 1

        for kernel_size in kernel_sizes:

            filter_ = np.ones(shape=(kernel_size, input_channels, 1))
            indices_ = np.arange(kernel_size)

            filter_[indices_ % 2 > 0] *= -1

            conv = tf.keras.layers.Conv1D(
                filters=1,
                kernel_size=kernel_size,
                padding="same",
                use_bias=False,
                kernel_initializer=tf.keras.initializers.Constant(filter_),
                trainable=False,
                name="hybird-decrease-" + str(self.keep_track) + "-" + str(kernel_size),
            )(input_tensor)
            conv_list.append(conv)

            self.keep_track += 1

        for kernel_size in kernel_sizes[1:]:

            filter_ = np.zeros(
                shape=(kernel_size + kernel_size // 2, input_channels, 1)
            )

            xmash = np.linspace(start=0, stop=1, num=kernel_size // 4 + 1)[1:].reshape(
                (-1, 1, 1)
            )

            filter_left = xmash**2
            filter_right = filter_left[::-1]

            filter_[0 : kernel_size // 4] = -filter_left
            filter_[kernel_size // 4 : kernel_size // 2] = -filter_right
            filter_[kernel_size // 2 : 3 * kernel_size // 4] = 2 * filter_left
            filter_[3 * kernel_size // 4 : kernel_size] = 2 * filter_right
            filter_[kernel_size : 5 * kernel_size // 4] = -filter_left
            filter_[5 * kernel_size // 4 :] = -filter_right

            conv = tf.keras.layers.Conv1D(
                filters=1,
                kernel_size=kernel_size + kernel_size // 2,
                padding="same",
                use_bias=False,
                kernel_initializer=tf.keras.initializers.Constant(filter_),
                trainable=False,
                name="hybird-peeks-" + str(self.keep_track) + "-" + str(kernel_size),
            )(input_tensor)

            conv_list.append(conv)

            self.keep_track += 1

        hybird_layer = tf.keras.layers.Concatenate(axis=2)(conv_list)
        hybird_layer = tf.keras.layers.Activation(activation="relu")(hybird_layer)

        return hybird_layer

    def _inception_module(
        self,
        input_tensor,
        dilation_rate,
        stride=1,
        activation="linear",
        use_hybird_layer=False,
        use_multiplexing=True,
    ):

        input_inception = input_tensor

        if not use_multiplexing:

            n_convs = 1
            n_filters = self.n_filters * 3

        else:
            n_convs = 3
            n_filters = self.n_filters

        kernel_size_s = [self.kernel_size // (2**i) for i in range(n_convs)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(
                tf.keras.layers.Conv1D(
                    filters=n_filters,
                    kernel_size=kernel_size_s[i],
                    strides=stride,
                    padding="same",
                    dilation_rate=dilation_rate,
                    activation=activation,
                    use_bias=False,
                )(input_inception)
            )

        if use_hybird_layer:

            self.hybird = self.hybird_layer(
                input_tensor=input_tensor, input_channels=input_tensor.shape[-1]
            )
            conv_list.append(self.hybird)

        if len(conv_list) > 1:
            x = tf.keras.layers.Concatenate(axis=2)(conv_list)
        else:
            x = conv_list[0]

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation="relu")(x)

        return x

    def _fcn_module(
        self,
        input_tensor,
        kernel_size,
        dilation_rate,
        n_filters,
        stride=1,
        activation="relu",
    ):

        x = tf.keras.layers.SeparableConv1D(
            filters=n_filters,
            kernel_size=kernel_size,
            padding="same",
            strides=stride,
            dilation_rate=dilation_rate,
            use_bias=False,
        )(input_tensor)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation=activation)(x)

        return x

    def build_model(self):

        self.keep_track = 0

        input_shape = (self.length_TS,)

        input_layer = tf.keras.layers.Input(input_shape)
        reshape_layer = tf.keras.layers.Reshape(target_shape=(self.length_TS, 1))(
            input_layer
        )

        inception = self._inception_module(
            input_tensor=reshape_layer,
            dilation_rate=1,
            use_hybird_layer=self.use_custom_filters,
        )

        self.kernel_size //= 2

        input_tensor = inception

        dilation_rate = 1

        for i in range(2):

            if self.use_dilation:
                dilation_rate = 2 ** (i + 1)

            x = self._fcn_module(
                input_tensor=input_tensor,
                kernel_size=self.kernel_size // (2**i),
                n_filters=self.n_filters,
                dilation_rate=dilation_rate,
            )

            input_tensor = x

        gap = tf.keras.layers.GlobalAveragePooling1D()(x)

        output_layer = tf.keras.layers.Dense(
            units=self.n_classes, activation="softmax"
        )(gap)

        self.model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=50, min_lr=1e-4
        )
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.output_directory + "best_model.hdf5",
            monitor="loss",
            save_best_only=True,
        )
        self.callbacks = [reduce_lr, model_checkpoint]

        self.model.compile(
            loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"]
        )

    def fit(self, xtrain, ytrain, xval=None, yval=None, plot_test=False):

        ytrain = np.expand_dims(ytrain, axis=1)
        ohe = OHE(sparse=False)
        ytrain = ohe.fit_transform(ytrain)

        if plot_test:

            yval = np.expand_dims(yval, axis=1)
            ohe = OHE(sparse=False)
            yval = ohe.fit_transform(yval)

        if plot_test:

            hist = self.model.fit(
                xtrain,
                ytrain,
                batch_size=self.batch_size,
                epochs=self.n_epochs,
                verbose=self.verbose,
                validation_data=(xval, yval),
                callbacks=self.callbacks,
            )
        else:

            hist = self.model.fit(
                xtrain,
                ytrain,
                batch_size=self.batch_size,
                epochs=self.n_epochs,
                verbose=self.verbose,
                callbacks=self.callbacks,
            )

        plt.figure(figsize=(20, 10))

        plt.plot(hist.history["loss"], lw=3, color="blue", label="Training Loss")

        if plot_test:
            plt.plot(
                hist.history["val_loss"], lw=3, color="red", label="Validation Loss"
            )

        plt.legend()
        plt.savefig(self.output_directory + "loss.pdf")
        plt.cla()

        plt.plot(
            hist.history["accuracy"], lw=3, color="blue", label="Training Accuracy"
        )

        if plot_test:
            plt.plot(
                hist.history["val_accuracy"],
                lw=3,
                color="red",
                label="Validation Accuracy",
            )

        plt.legend()
        plt.savefig(self.output_directory + "accuracy.pdf")

        plt.cla()
        plt.clf()

        tf.keras.backend.clear_session()

    def fit_and_track_emissions(
        self, xtrain, ytrain, xval=None, yval=None, plot_test=False
    ):
        @track_emissions(project_name="LITE", output_dir=self.output_directory)
        def _fit(xtrain, ytrain, xval, yval, plot_test):

            ytrain = np.expand_dims(ytrain, axis=1)
            ohe = OHE(sparse=False)
            ytrain = ohe.fit_transform(ytrain)

            if plot_test:

                yval = np.expand_dims(yval, axis=1)
                ohe = OHE(sparse=False)
                yval = ohe.fit_transform(yval)

            start_time = time.time()

            if plot_test:

                hist = self.model.fit(
                    xtrain,
                    ytrain,
                    batch_size=self.batch_size,
                    epochs=self.n_epochs,
                    verbose=self.verbose,
                    validation_data=(xval, yval),
                    callbacks=self.callbacks,
                )
            else:

                hist = self.model.fit(
                    xtrain,
                    ytrain,
                    batch_size=self.batch_size,
                    epochs=self.n_epochs,
                    verbose=self.verbose,
                    callbacks=self.callbacks,
                )

            self.train_duration = time.time() - start_time

            plt.figure(figsize=(20, 10))

            plt.plot(hist.history["loss"], lw=3, color="blue", label="Training Loss")

            if plot_test:
                plt.plot(
                    hist.history["val_loss"], lw=3, color="red", label="Validation Loss"
                )

            plt.legend()
            plt.savefig(self.output_directory + "loss.pdf")
            plt.cla()

            plt.plot(
                hist.history["accuracy"], lw=3, color="blue", label="Training Accuracy"
            )

            if plot_test:
                plt.plot(
                    hist.history["val_accuracy"],
                    lw=3,
                    color="red",
                    label="Validation Accuracy",
                )

            plt.legend()
            plt.savefig(self.output_directory + "accuracy.pdf")

            plt.cla()
            plt.clf()

            tf.keras.backend.clear_session()

        _fit(xtrain=xtrain, ytrain=ytrain, xval=xval, yval=yval, plot_test=plot_test)

        emissions = pd.read_csv(self.output_directory + "emissions.csv")

        co2 = emissions["emissions"][0]
        energy = emissions["energy_consumed"][0]
        country_name = str(emissions["country_name"][0])
        region = str(emissions["region"][0])

        os.remove(self.output_directory + "emissions.csv")

        tf.keras.backend.clear_session()

        dict_emissions = {
            "co2": co2,
            "energy": energy,
            "country_name": country_name,
            "region": region,
            "duration": self.train_duration,
        }

        with open(self.output_directory + "dict_emissions.json", "w") as fjson:
            json.dump(dict_emissions, fjson)

        return dict_emissions

    def predict(self, xtest, ytest):

        model = tf.keras.models.load_model(
            self.output_directory + "best_model.hdf5", compile=False
        )

        start_time = time.time()
        ypred = model.predict(xtest)
        duration = time.time() - start_time

        ypred_argmax = np.argmax(ypred, axis=1)

        return (
            np.asarray(ypred),
            accuracy_score(y_true=ytest, y_pred=ypred_argmax, normalize=True),
            duration,
        )
