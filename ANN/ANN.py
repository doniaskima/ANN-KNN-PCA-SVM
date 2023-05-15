import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense

class ANN:
    def __init__(self, input_dim, output_dim, hidden_layers=1, hidden_units=32, activation='relu',
                 learning_rate=0.001, batch_size=32, epochs=10):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
    
    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(self.hidden_units, input_dim=self.input_dim, activation=self.activation))
        for _ in range(self.hidden_layers - 1):
            self.model.add(Dense(self.hidden_units, activation=self.activation))
        self.model.add(Dense(self.output_dim, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    def fit(self, X, Y):
        self.build_model()
        self.model.fit(X, Y, epochs=self.epochs, batch_size=self.batch_size)
    
    def predict(self, X):
        predictions = self.model.predict(X)
        return np.round(predictions).astype(int)
