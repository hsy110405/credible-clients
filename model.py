import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras


class CreditModel:

    def __init__(self):
        """
        Instantiates the model object, creating class variables if needed.
        """

        # TODO: Initialize your model object.
        #from xgboost import XGBClassifier
        #self.classifier = XGBClassifier()

        from keras.models import Sequential
        from keras.layers import Dense

        self.classifier = keras.Sequential()

        # Adding the input layer and the first hidden layer
        self.classifier.add(Dense(kernel_initializer="uniform", activation="relu", input_dim=23, units=12))

        # Adding the second hidden layer
        self.classifier.add(Dense(units=12, activation="relu", kernel_initializer="uniform"))

        # Adding the output layer
        self.classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))

        # Compiling Neural Network
        self.classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def fit(self, X_train, y_train):

        """
        Fits the model based on the given `X_train` and `y_train`.

        You should somehow manipulate and store this data to your model class
        so that you can make predictions on new testing data later on.
        """

        # TODO: Fit your model based on the given X and y.

        # self.classifier.fit(X_train,y_train)
        self.classifier.fit(X_train, y_train, batch_size=10, epochs=50)

    def predict(self, X_test):
        """
        Returns `y_hat`, a prediction for a given `X_test` after fitting.

        You should make use of the data that you stored/computed in the
        fitting phase to make your prediction on this new testing data.
        """

        # TODO: Predict on `X_test` based on what you learned in the fit phase.

        y_pred = self.classifier.predict(X_test)

        return y_pred
        #return self.classifier.predict(X_test)
