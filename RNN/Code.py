import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Initialize parameters
input_size = 4  # Number of nucleotide types (A, C, G, T)
hidden_size = 10
output_size = 1
learning_rate = 0.01
num_epochs = 100

# Initialize weights and biases
W_xh = np.random.randn(hidden_size, input_size)
W_hh = np.random.randn(hidden_size, hidden_size)
W_hy = np.random.randn(output_size, hidden_size)
b_h = np.zeros((hidden_size, 1))
b_y = np.zeros((output_size, 1))

# Training loop
for epoch in range(num_epochs):
    # Example DNA sequence (binary encoding for simplicity)
    input_sequence = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Target label (1 for motif present, 0 for motif absent)
    target = 1

    # Forward pass
    h_t = np.zeros((hidden_size, 1))
    for t in range(len(input_sequence)):
        x_t = input_sequence[t].reshape(-1, 1)
        a_t = np.dot(W_xh, x_t) + np.dot(W_hh, h_t) + b_h
        h_t = np.tanh(a_t)
    output = sigmoid(np.dot(W_hy, h_t) + b_y)

    # Compute loss
    loss = -target * np.log(output) - (1 - target) * np.log(1 - output)

    # Backward pass
    dy = output - target
    dW_hy = np.dot(dy, h_t.T)
    db_y = dy
    dh = np.dot(W_hy.T, dy) * (1 - h_t**2)
    dW_xh, dW_hh, db_h = np.zeros_like(W_xh), np.zeros_like(W_hh), np.zeros_like(b_h)
    for t in reversed(range(len(input_sequence))):
        x_t = input_sequence[t].reshape(-1, 1)
        a_t = np.dot(W_xh, x_t) + np.dot(W_hh, h_t) + b_h
        dh = dh + np.dot(W_hh.T, dh) * (1 - h_t**2)
        dW_xh += np.dot(dh, x_t.T)
        dW_hh += np.dot(dh, h_t.T)
        db_h += dh

    # Update parameters
    W_xh -= learning_rate * dW_xh
    W_hh -= learning_rate * dW_hh
    W_hy -= learning_rate * dW_hy
    b_h -= learning_rate * db_h
    b_y -= learning_rate * db_y

    # Print the loss for each epoch
    print(f"Epoch {epoch + 1}, Loss: {loss}")
