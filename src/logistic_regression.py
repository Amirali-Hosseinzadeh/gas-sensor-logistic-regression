import numpy as np


class LogisticRegressionScratch:

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = 0
        self.loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def log_loss(self, y, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        loss = -np.mean(
            y * np.log(y_pred) +
            (1 - y) * np.log(1 - y_pred)
        )

        return loss

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []

        for _ in range(self.n_iterations):
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)

            loss = self.log_loss(y, y_pred)
            self.loss_history.append(loss)

            error = y_pred - y

            dw = (X.T @ error) / n_samples
            db = error.mean()

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X):
        z = X @ self.weights + self.bias
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


class OneVsRestLogisticRegression:

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.classes = None
        self.models = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.models = []

        for current_class in self.classes:

            y_binary = (y == current_class).astype(int)

            model = LogisticRegressionScratch(
                learning_rate=self.learning_rate,
                n_iterations=self.n_iterations
            )

            model.fit(X, y_binary)
            self.models.append(model)

        return self

    def predict_proba(self, X):
        probabilities = []

        for model in self.models:
            probability = model.predict_proba(X)
            probabilities.append(probability)

        return np.array(probabilities).T

    def predict(self, X):
        probabilities = self.predict_proba(X)

        class_indices = np.argmax(probabilities, axis=1)

        return self.classes[class_indices]