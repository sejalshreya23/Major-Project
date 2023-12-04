import numpy as np

class LinearSVM:
    def __init__(self, learning_rate=0.01, regularization_strength=0.01, num_epochs=1000):
        self.learning_rate = learning_rate
        self.regularization_strength = regularization_strength
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None

    def one_hot_encoding(self, sequence):
        mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
        return np.array([mapping[base] for base in sequence]).flatten()

    def train(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for epoch in range(self.num_epochs):
            for i in range(num_samples):
                decision_function = np.dot(X[i], self.weights) + self.bias
                hinge_loss_gradient = 0
                if y[i] * decision_function < 1:
                    hinge_loss_gradient = -y[i] * X[i]

                self.weights = self.weights - self.learning_rate * (
                    2 * self.regularization_strength * self.weights + hinge_loss_gradient
                )
                self.bias = self.bias - self.learning_rate * y[i]

    def predict(self, X):
        decision_function = np.dot(X, self.weights) + self.bias
        return np.sign(decision_function)

# Example usage
if __name__ == "__main__":
    positive_sequence = "ACGTGTATAAACGTAG"
    negative_sequence = "CGTACGACGTAAGCTA"
    test_sequence = "CGTATAAGGCTTAGCA"

    svm = LinearSVM()
    
    # One-hot encoding
    positive_data = svm.one_hot_encoding(positive_sequence)
    negative_data = svm.one_hot_encoding(negative_sequence)
    test_data = svm.one_hot_encoding(test_sequence)

    # Labeling
    labels = np.array([1, -1])

    # Training the Linear SVM
    svm.train(np.vstack([positive_data, negative_data]), labels)

    # Making predictions
    prediction = svm.predict(test_data)
    if prediction == 1:
        print("The motif is present in the test sequence.")
    else:
        print("The motif is not present in the test sequence.")
