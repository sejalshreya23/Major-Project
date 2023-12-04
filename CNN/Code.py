import numpy as np

# Define activation function (ReLU)
def relu(x):
    return np.maximum(0, x)

# Define the one-dimensional convolution operation
def conv1d(input_data, filters, biases, stride=1):
    input_size, input_channels = input_data.shape
    filter_size, filter_channels = filters.shape
    output_size = int((input_size - filter_size) / stride) + 1

    output = np.zeros((output_size, 1))

    for i in range(0, input_size - filter_size + 1, stride):
        output[int(i / stride)] = np.sum(input_data[i:i+filter_size] * filters) + biases

    return output

# Define the max pooling operation
def max_pooling(input_data, pool_size=2):
    input_size = input_data.shape[0]
    output_size = int(input_size / pool_size)

    output = np.zeros((output_size, 1))

    for i in range(0, input_size, pool_size):
        output[int(i / pool_size)] = np.max(input_data[i:i+pool_size])

    return output

# Define the fully connected layer
def fully_connected(input_data, weights, biases):
    return np.dot(input_data, weights) + biases

# Define the sigmoid activation function for the output layer
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# One-hot encoding function (for simplicity, assuming A=1, C=2, G=3, T=4)
def one_hot_encoding(sequence):
    mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
    return np.array([mapping[base] for base in sequence])

# Initialise parameters
filter_size = 3
num_filters = 2
stride = 1
pool_size = 2
learning_rate = 0.01
num_epochs = 100

# Example DNA sequence (length = 10)
sequence = "ATCGATCGAT"

# One-hot encode the sequence
encoded_sequence = one_hot_encoding(sequence)

# Initialize filters and biases
filters = np.random.randn(filter_size, num_filters)
biases_conv = np.random.randn(num_filters)
weights_fc = np.random.randn(num_filters * 4, 1)
biases_fc = np.random.randn(1)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    conv_output = conv1d(encoded_sequence, filters, biases_conv, stride)
    relu_output = relu(conv_output)
    pooled_output = max_pooling(relu_output, pool_size)
    flattened_output = pooled_output.flatten()
    fc_output = fully_connected(flattened_output, weights_fc, biases_fc)
    final_output = sigmoid(fc_output)

    # Backward pass (gradient descent not implemented for simplicity)

    # Print the predicted output for each epoch
    print(f"Epoch {epoch + 1}, Predicted Output: {final_output[0]}")
