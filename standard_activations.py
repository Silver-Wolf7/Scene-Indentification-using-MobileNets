from keras import backend as K

'''
Those are the standard activation functions that were not available in 
tensorflow.keras.activations
'''

# defining the hard-swish activation function
def h_swish(x):
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0

# defining the leaky ReLU activation function
def leaky_relu(x, alpha=0.1):
    return K.relu(x, alpha=alpha)

# defining the combined ReLU activation function
def combined_relu(x):
    return K.relu(x, alpha=0.1, max_value=6.0)