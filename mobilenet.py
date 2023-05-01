from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Conv2D, ZeroPadding2D, DepthwiseConv2D, ReLU
from keras.layers import Activation
from keras.layers import Dropout
from keras.models import Model
from keras.layers import Input
from keras.layers import Reshape
from keras.utils.layer_utils import get_source_inputs
from keras_applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from tensorflow.keras.activations import gelu, selu, elu
from standard_activations import h_swish, combined_relu, leaky_relu
from custom_activations import custom_activation_A, custom_activation_A_2, custom_activation_A_3
from custom_activations import custom_activation_A_4, custom_activation_A_5, custom_activation_A_6
from custom_activations import custom_activation_B, custom_activation_B_2, custom_activation_B_3

'''
To chose activation function to use, type one of the names in the activation function list,
to use the default ReLU6 just keep it as None
'''
activation = None
# list of available activation functions
activation_functions = [gelu, selu, elu, custom_activation_A, custom_activation_A_2, custom_activation_A_3,
                        custom_activation_A_4, custom_activation_A_5, custom_activation_A_6, combined_relu,
                        custom_activation_B, custom_activation_B_2, custom_activation_B_3, h_swish, leaky_relu]

# MobileNetV1 code from keras source code with slight modifications
# https://github.com/keras-team/keras/blob/v2.12.0/keras/applications/mobilenet.py
# The code was also modified with the help
# https://github.com/kobiso/CBAM-keras/blob/master/models/mobilenets.py
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
    if activation == None:
        x = ReLU(6.0, name='conv_pw_%d_relu' % block_id)(x)
    elif activation in activation_functions:
        x = Activation(activation, name='conv_pw_%d_relu' % block_id)(x)
    else:
        raise ValueError("This activation function does not exist")
    
    return x

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
    
    if activation == None:
        x = ReLU(6.0, name='conv_pw_%d_relu' % block_id)(x)
    elif activation in activation_functions:
        x = Activation(activation, name='conv_pw_%d_relu' % block_id)(x)
    else:
        raise ValueError("This activation function does not exist")

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
    
    if activation == None:
        x = ReLU(6.0, name='conv_pw_%d_relu' % block_id)(x)
    elif activation in activation_functions:
        x = Activation(activation, name='conv_pw_%d_relu' % block_id)(x)
    else:
        raise ValueError("This activation function does not exist")
		
    return x
