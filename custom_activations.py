from keras import backend as K


# Custom activation A described in the report with saturation between -1 and 1 with maximum value 6 and minimum value -6
def custom_activation_A(x):
    # Defining the different pieces of the function
    piece1 = K.cast(K.greater(x, -1), dtype='float32') * K.cast(K.less(x, 1), dtype='float32') * 0
    piece2 = K.cast(K.greater_equal(x, 1), dtype='float32') * K.cast(K.less_equal(x, 6), dtype='float32') * (6/5 * x - 6/5)
    piece3 = K.cast(K.greater(x, 6), dtype='float32') * 6
    piece4 = K.cast(K.greater_equal(x, -6), dtype='float32') * K.cast(K.less_equal(x, -1), dtype='float32') * (6/5 * x + 6/5)
    piece5 = K.cast(K.less(x, -6), dtype='float32') * -6
    
    # Combining the pieces to create the full function
    return piece1 + piece2 + piece3 + piece4 + piece5

# Custom activation A described in the report with saturation between -0.5 and 0.5 with maximum value 7 and minimum value -7
def custom_activation_A_2(x):
    # Defining the different pieces of the function
    piece1 = K.cast(K.greater(x, -0.5), dtype='float32') * K.cast(K.less(x, 0.5), dtype='float32') * K.zeros_like(x)
    piece2 = K.cast(K.greater_equal(x, 0.5), dtype='float32') * K.cast(K.less_equal(x, 7), dtype='float32') * (6/6.5 * x - 6/6.5 * 0.5)
    piece3 = K.cast(K.greater(x, 7), dtype='float32') * 7.0
    piece4 = K.cast(K.greater_equal(x, -7), dtype='float32') * K.cast(K.less_equal(x, -0.5), dtype='float32') * (6/6.5 * x - 6/6.5 * 0.5)
    piece5 = K.cast(K.less(x, -7), dtype='float32') * (-7.0)
    
    # Combining the pieces to create the full function
    return piece1 + piece2 + piece3 + piece4 + piece5

# Custom activation A described in the report with saturation between -0.5 and 0.5 with maximum value 5 and minimum value -5
def custom_activation_A_3(x):
    # Defining the different pieces of the function
    piece1 = K.cast(K.greater(x, -0.5), dtype='float32') * K.cast(K.less(x, 0.5), dtype='float32') * K.zeros_like(x)
    piece2 = K.cast(K.greater_equal(x, 0.5), dtype='float32') * K.cast(K.less_equal(x, 5), dtype='float32') * (6/4.5 * x - 6/4.5 * 0.5)
    piece3 = K.cast(K.greater(x, 5), dtype='float32') * 5.0
    piece4 = K.cast(K.greater_equal(x, -5), dtype='float32') * K.cast(K.less_equal(x, -0.5), dtype='float32') * (6/4.5 * x + 6/4.5 * 0.5)
    piece5 = K.cast(K.less(x, -5), dtype='float32') * (-5.0)
    
    # Combining the pieces to create the full function
    return piece1 + piece2 + piece3 + piece4 + piece5

# Custom activation A described in the report with saturation between -0.5 and 0.5 with maximum value 6 and minimum value -6
def custom_activation_A_4(x):
    # Defining the different pieces of the function
    piece1 = K.cast(K.greater(x, -0.5), dtype='float32') * K.cast(K.less(x, 0.5), dtype='float32') * K.zeros_like(x)
    piece2 = K.cast(K.greater_equal(x, 0.5), dtype='float32') * K.cast(K.less_equal(x, 6), dtype='float32') * (6/5.5 * x - 6/5.5 * 0.5)
    piece3 = K.cast(K.greater(x, 6), dtype='float32') * 6.0
    piece4 = K.cast(K.greater_equal(x, -6), dtype='float32') * K.cast(K.less_equal(x, -0.5), dtype='float32') * (6/5.5 * x + 6/5.5 * 0.5)
    piece5 = K.cast(K.less(x, -6), dtype='float32') * -6.0
    
    # Combining the pieces to create the full function
    return piece1 + piece2 + piece3 + piece4 + piece5

# Custom activation A described in the report with no saturation and with maximum value 6 and minimum value -6
def custom_activation_A_5(x):
    piece1 = K.cast(K.less(x, -6), dtype='float32') * -6
    piece2 = K.cast(K.greater_equal(x, -6), dtype='float32') * K.cast(K.less_equal(x, 6), dtype='float32') * x
    piece3 = K.cast(K.greater(x, 6), dtype='float32') * 6

    return piece1 + piece2 + piece3

# Custom activation function A described in the report with saturation between -0.5 and 0 with maximum value 6 and minimum value -6
def custom_activation_A_6(x):
    # Defining the different pieces of the function
    piece1 = K.cast(K.greater_equal(x, -0.5), dtype='float32') * K.cast(K.less(x, 0), dtype='float32') * K.zeros_like(x)
    piece2 = K.cast(K.greater_equal(x, 0), dtype='float32') * K.cast(K.less_equal(x, 6.0), dtype='float32') * x
    piece3 = K.cast(K.greater(x, 6), dtype='float32') * 6.0
    piece4 = K.cast(K.greater_equal(x, -6.5), dtype='float32') * K.cast(K.less(x, -0.5), dtype='float32') * (x + 0.5)
    piece5 = K.cast(K.less(x, -6.5), dtype='float32') * -6.0
    
    # Combining the pieces to create the full function
    return piece1 + piece2 + piece3 + piece4 + piece5

# Custom activation function B described in the report with gradient 1
def custom_activation_B(x):
    # Defining the different pieces of the function
    piece1 = K.cast(K.greater(x, 6.0), dtype='float32') * 6.0
    piece2 = K.cast(K.greater_equal(x, -1), dtype='float32') * K.cast(K.less_equal(x, 6.0), dtype='float32') * x
    piece3 = K.cast(K.less(x, -1), dtype='float32') * -1
    
    # Combining the pieces to create the full function
    return piece1 + piece2 + piece3

# Custom activation function B described in the report with gradient 1.2
def custom_activation_B_2(x):
    # Defining the different pieces of the function
    piece1 = K.cast(K.greater(x, 5.0), dtype='float32') * 6.0
    piece2 = K.cast(K.greater_equal(x, -1/1.2), dtype='float32') * K.cast(K.less_equal(x, 5.0), dtype='float32') * (x*1.2)
    piece3 = K.cast(K.less(x, -1/1.2), dtype='float32') * -1
    
    # Combining the pieces to create the full function
    return piece1 + piece2 + piece3

# Custom activation function B described in the report with gradient 0.8
def custom_activation_B_3(x):
    # Defining the different pieces of the function
    piece1 = K.cast(K.greater(x, 7.5), dtype='float32') * 6.0
    piece2 = K.cast(K.greater_equal(x, -1.25), dtype='float32') * K.cast(K.less_equal(x, 7.5), dtype='float32') * (x*0.8)
    piece3 = K.cast(K.less(x, -1.25), dtype='float32') * -1
    
    # Combining the pieces to create the full function
    return piece1 + piece2 + piece3