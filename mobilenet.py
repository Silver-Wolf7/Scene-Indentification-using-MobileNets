from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings

from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Conv2D, ZeroPadding2D, DepthwiseConv2D, ReLU, ELU
from keras.utils.data_utils import get_file
from keras.utils.layer_utils import get_source_inputs
from keras.applications import imagenet_utils
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K
from tensorflow.keras.activations import gelu, selu

from attention_module import attach_attention_module

def elu2(x):
    return K.switch(K.less_equal(x, 1), K.exp(x) - 1, x)

def relu2(x):
    # Define the different pieces of the function
    piece1 = K.cast(K.greater(x, -1), dtype='float32') * K.cast(K.less(x, 1), dtype='float32') * 0
    piece2 = K.cast(K.greater_equal(x, 1), dtype='float32') * K.cast(K.less_equal(x, 6), dtype='float32') * (6/5 * x - 6/5)
    piece3 = K.cast(K.greater(x, 6), dtype='float32') * 6
    piece4 = K.cast(K.greater_equal(x, -6), dtype='float32') * K.cast(K.less_equal(x, -1), dtype='float32') * (6/5 * x + 6/5)
    piece5 = K.cast(K.less(x, -6), dtype='float32') * -6
    
    # Combine the pieces to create the full function
    return piece1 + piece2 + piece3 + piece4 + piece5

def relu3(x):
    # Define the different pieces of the function
    piece1 = K.cast(K.greater(x, -0.5), dtype='float32') * K.cast(K.less(x, 0.5), dtype='float32') * K.zeros_like(x)
    piece2 = K.cast(K.greater_equal(x, 0.5), dtype='float32') * K.cast(K.less_equal(x, 7), dtype='float32') * (6/6.5 * x - 6/6.5 * 0.5)
    piece3 = K.cast(K.greater(x, 7), dtype='float32') * 6.0
    piece4 = K.cast(K.greater_equal(x, -7), dtype='float32') * K.cast(K.less_equal(x, -0.5), dtype='float32') * (6/6.5 * x - 6/6.5 * 0.5)
    piece5 = K.cast(K.less(x, -7), dtype='float32') * (-6.0)
    
    # Combine the pieces to create the full function
    return piece1 + piece2 + piece3 + piece4 + piece5

def relu4(x):
    # Define the different pieces of the function
    piece1 = K.cast(K.greater(x, -0.5), dtype='float32') * K.cast(K.less(x, 0.5), dtype='float32') * K.zeros_like(x)
    piece2 = K.cast(K.greater_equal(x, 0.5), dtype='float32') * K.cast(K.less_equal(x, 5), dtype='float32') * (6/4.5 * x - 6/4.5 * 0.5)
    piece3 = K.cast(K.greater(x, 5), dtype='float32') * 6.0
    piece4 = K.cast(K.greater_equal(x, -5), dtype='float32') * K.cast(K.less_equal(x, -0.5), dtype='float32') * (6/4.5 * x + 6/4.5 * 0.5)
    piece5 = K.cast(K.less(x, -5), dtype='float32') * (-6.0)
    
    # Combine the pieces to create the full function
    return piece1 + piece2 + piece3 + piece4 + piece5

def relu5(x):
    # Define the different pieces of the function
    piece1 = K.cast(K.greater(x, -0.5), dtype='float32') * K.cast(K.less(x, 0.5), dtype='float32') * K.zeros_like(x)
    piece2 = K.cast(K.greater_equal(x, 0.5), dtype='float32') * K.cast(K.less_equal(x, 6), dtype='float32') * (6/5.5 * x - 6/5.5 * 0.5)
    piece3 = K.cast(K.greater(x, 6), dtype='float32') * 6.0
    piece4 = K.cast(K.greater_equal(x, -6), dtype='float32') * K.cast(K.less_equal(x, -0.5), dtype='float32') * (6/5.5 * x + 6/5.5 * 0.5)
    piece5 = K.cast(K.less(x, -6), dtype='float32') * -6.0
    
    # Combine the pieces to create the full function
    return piece1 + piece2 + piece3 + piece4 + piece5

def relu6(x):
    piece1 = K.cast(K.less(x, -6), dtype='float32') * -6
    piece2 = K.cast(K.greater_equal(x, -6), dtype='float32') * K.cast(K.less_equal(x, 6), dtype='float32') * x
    piece3 = K.cast(K.greater(x, 6), dtype='float32') * 6

    return piece1 + piece2 + piece3

def h_swish(x):
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0

def leaky_relu(x, alpha=0.1):
    return K.relu(x, alpha=alpha)

def combined_relu(x):
    return K.relu(x, alpha=0.1, max_value=6.0)

def preprocess_input(x):
    """Preprocesses a numpy array encoding a batch of images.
    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].
    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, mode='tf')

def MobileNet(input_shape=None,
                alpha=1.0,
                depth_multiplier=1,
                dropout=1e-3,
                include_top=True,
                weights=None,
                input_tensor=None,
                pooling=None,
                classes=1000,
			    attention_module=None,
                ratio=8,
                kernel_size=7,
                c_bias=True,
                s_bias=False):
    
    if K.backend() != 'tensorflow':
        raise RuntimeError('Only TensorFlow backend is currently supported, '
                           'as other backends do not support '
                           'depthwise convolution.')

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as ImageNet with `include_top` '
                         'as true, `classes` should be 1000')

    # Determine proper input shape and default size.
    if input_shape is None:
        default_size = 224
    else:
        if K.image_data_format() == 'channels_first':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]

        if rows == cols and rows in [128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=default_size,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if K.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = _conv_block(img_input, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1, attention_module=attention_module,
                              ratio=ratio, kernel_size=kernel_size, c_bias=c_bias, s_bias=s_bias)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2, attention_module=attention_module,
                              ratio=ratio, kernel_size=kernel_size, c_bias=c_bias, s_bias=s_bias)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3, attention_module=attention_module,
                              ratio=ratio, kernel_size=kernel_size, c_bias=c_bias, s_bias=s_bias)

    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4, attention_module=attention_module,
                              ratio=ratio, kernel_size=kernel_size, c_bias=c_bias, s_bias=s_bias)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5, attention_module=attention_module,
                              ratio=ratio, kernel_size=kernel_size, c_bias=c_bias, s_bias=s_bias)

    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6, attention_module=attention_module,
                              ratio=ratio, kernel_size=kernel_size, c_bias=c_bias, s_bias=s_bias)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7, attention_module=attention_module,
                              ratio=ratio, kernel_size=kernel_size, c_bias=c_bias, s_bias=s_bias)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8, attention_module=attention_module,
                              ratio=ratio, kernel_size=kernel_size, c_bias=c_bias, s_bias=s_bias)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9, attention_module=attention_module,
                              ratio=ratio, kernel_size=kernel_size, c_bias=c_bias, s_bias=s_bias)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10, attention_module=attention_module,
                              ratio=ratio, kernel_size=kernel_size, c_bias=c_bias, s_bias=s_bias)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11, attention_module=attention_module,
                              ratio=ratio, kernel_size=kernel_size, c_bias=c_bias, s_bias=s_bias)

    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12, attention_module=attention_module,
                              ratio=ratio, kernel_size=kernel_size, c_bias=c_bias, s_bias=s_bias)
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13, attention_module=attention_module,
                              ratio=ratio, kernel_size=kernel_size, c_bias=c_bias, s_bias=s_bias)

    if include_top:
        if K.image_data_format() == 'channels_first':
            shape = (int(1024 * alpha), 1, 1)
        else:
            shape = (1, 1, int(1024 * alpha))

        x = GlobalAveragePooling2D()(x)
        x = Reshape(shape, name='reshape_n_1')(x)
        x = Dropout(dropout, name='dropout')(x)
        x = Conv2D(classes, (1, 1),
                   padding='same', name='conv_preds')(x)
        x = Activation('softmax', name='act_softmax')(x)
        x = Reshape((classes,), name='reshape_final')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name='se_mobilenet_%0.2f_%s' % (alpha, rows))

    # Load weights.
    if weights == "imagenet":
        if alpha == 1.0:
            alpha_text = "1_0"
        elif alpha == 0.75:
            alpha_text = "7_5"
        elif alpha == 0.50:
            alpha_text = "5_0"
        else:
            alpha_text = "2_5"

        model.load_weights("mobilenet_%s_%d_tf_no_top.h5" % (alpha_text, rows))

    return model


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1', )(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return ReLU(6.0, name='conv1_relu')(x)

def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1, attention_module=None,
                          ratio=8, c_bias=True, s_bias=False, kernel_size=7):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = ZeroPadding2D(
            ((0, 1), (0, 1)), name="conv_pad_%d" % block_id
        )(inputs)
    x = DepthwiseConv2D((3, 3),
                        padding="same" if strides == (1, 1) else "valid",
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = ReLU(6.0, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
    x = ReLU(6.0, name='conv_pw_%d_relu' % block_id)(x)

    # attention_module
    if attention_module is not None:
        x = attach_attention_module(x, attention_module, ratio, kernel_size, c_bias, s_bias)
		
    return x