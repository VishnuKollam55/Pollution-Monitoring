import numpy as np
from sklearn.preprocessing import MinMaxScaler

class SimpleLSTMPredictor:
    """
    Simplified LSTM-like predictor using weighted moving average.
    Used when TensorFlow is not available.
    """

    def __init__(self, sequence_length=7):
        self.sequence_length = sequence_length
        self.weights = None

    def fit(self, X, y):
        # Exponential weights (recent values more important)
        self.weights = np.exp(np.linspace(-1, 0, self.sequence_length))
        self.weights /= self.weights.sum()
        return self

    def predict(self, X):
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], X.shape[1])

        predictions = []
        for sequence in X:
            pred = np.average(sequence, weights=self.weights)
            pred += np.random.uniform(-5, 5)  # small variation
            predictions.append(pred)

        return np.array(predictions)
