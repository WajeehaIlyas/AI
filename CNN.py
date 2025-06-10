import gzip
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load MNIST dataset
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

# Normalize the images to [0, 1]
train_images = train_images / 255.0

# Convolution Operation
def convolution2d(input_image, kernel):
    #kernel and image dimensions
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = input_image.shape
    #output height and width of image after convolution
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

#initialize matrix with zeros for output
    output_image = np.zeros((output_height, output_width))
    #for each position in output image
    for i in range(output_height):
        for j in range(output_width):
            #extract the region of input image of same size as kernel and multiply it with kernel
            #sum the result and assign it to the output image
            output_image[i, j] = np.sum(input_image[i:i+kernel_height, j:j+kernel_width] * kernel)
    return output_image

# ReLU Activation
def relu(x):
    #return 0 for negative values
    return np.maximum(0, x)

# Max Pooling
def max_pooling(input_image, size=2, stride=2):
    output_height = (input_image.shape[0] - size) // stride + 1
    output_width = (input_image.shape[1] - size) // stride + 1
    #initialize matrix with zeros for output
    output_image = np.zeros((output_height, output_width))

    #for each position in output image, extract the region of input image of same size as kernel and take the maximum value
    for i in range(output_height):
        for j in range(output_width):
            output_image[i, j] = np.max(
                input_image[i*stride:i*stride+size, j*stride:j*stride+size]
            )
    return output_image

# Flatten the output
def flatten(input_image):
    return input_image.flatten()

# Fully Connected Layer
def fully_connected(input_vector, weights, biases):
    return np.dot(input_vector, weights) + biases

# Softmax Output Layer
def softmax(x):
    #convert the output of the fully connected layer to probabilities
    exp_x = np.exp(x - np.max(x)) 
    return exp_x / np.sum(exp_x)

# Cross-Entropy Loss
def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-8))  # small epsilon to avoid log(0)

# Forward Pass
def forward_pass(image, kernel, weights, biases):
    conv_out = convolution2d(image, kernel)
    relu_out = relu(conv_out)
    pool_out = max_pooling(relu_out)
    flat_out = flatten(pool_out)
    fc_out = fully_connected(flat_out, weights, biases)
    softmax_out = softmax(fc_out)
    return softmax_out, pool_out, flat_out

# Backpropagation: Compute Gradients and Update Weights
def backpropagate(image, y_true, y_pred, kernel, weights, biases, pool_out, flat_out, learning_rate):
    #difference between predicted and true labels
    output_error = y_pred - y_true
    weights_gradient = np.outer(flat_out, output_error)
    biases_gradient = output_error

    weights -= learning_rate * weights_gradient
    biases -= learning_rate * biases_gradient

    fc_error = np.dot(weights, output_error)
    #set error to 0 for flattened feature map where ReLU activation in forward pass was 0
    fc_error[flat_out <= 0] = 0  # ReLU backprop

    kernel_gradient = np.zeros_like(kernel)

    return weights, biases, kernel_gradient

# Initialize kernel, weights, and biases
kernel = np.random.randn(3, 3)

# Calculate output sizes
input_size = 28
conv_output_size = input_size - 3 + 1  # = 26
pool_output_size = (conv_output_size - 2) // 2 + 1  # = 13
flattened_size = pool_output_size * pool_output_size  # = 169

# Set weights and biases
weights = np.random.randn(flattened_size, 10)
biases = np.random.randn(10)

learning_rate = 0.1
epochs = 10
iterations_per_epoch = 100

# Training Loop
for epoch in range(epochs):
    epoch_loss = 0
    correct_predictions = 0
    print(f"Epoch {epoch + 1}/{epochs}")
    
    for i in range(min(len(train_images), iterations_per_epoch)):
        image = train_images[i]
        label = train_labels[i]

        # One-hot encoding
        y_true = np.zeros(10)
        y_true[label] = 1

        # Forward pass
        y_pred, pool_out, flat_out = forward_pass(image, kernel, weights, biases)

        # Loss
        loss = cross_entropy_loss(y_true, y_pred)
        epoch_loss += loss

        # Accuracy
        predicted_label = np.argmax(y_pred)
        if predicted_label == label:
            correct_predictions += 1

        # Backpropagation
        weights, biases, kernel_gradient = backpropagate(
            image, y_true, y_pred, kernel, weights, biases, pool_out, flat_out, learning_rate
        )

        if i % 10 == 0:
            print(f" Iteration {i + 1}/{iterations_per_epoch}, Loss: {loss:.4f}")

    accuracy = correct_predictions / iterations_per_epoch
    print(f"Epoch {epoch + 1} finished, Average Loss: {epoch_loss / iterations_per_epoch:.4f}, Accuracy: {accuracy * 100:.2f}%")

# Visualize sample image
plt.imshow(train_images[0], cmap='gray')
plt.title(f"Label: {train_labels[0]}")
plt.show()
