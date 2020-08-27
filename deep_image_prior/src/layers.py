import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

# Batchnorm epsilon
BN_EPS = 1e-5
LEAKY_RELU_ALPHA = 0.2

class Encoder(keras.layers.Layer):
    def __init__(self, out_channels, kernel_size, use_bias):
        super(Encoder, self).__init__()
        self.encode = keras.Sequential([
            
            ReflectionPadding2D(((kernel_size - 1) // 2, (kernel_size - 1) // 2)),
            keras.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, strides=1, padding='valid', use_bias=use_bias),
            keras.layers.AveragePooling2D(pool_size=2, strides=2),
            keras.layers.BatchNormalization(epsilon=BN_EPS),
            keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA),
            
            ReflectionPadding2D(((kernel_size - 1) // 2, (kernel_size - 1) // 2)),
            keras.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, strides=1, padding='valid', use_bias=use_bias),
            keras.layers.BatchNormalization(epsilon=BN_EPS),
            keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)
        ])
    
    def call(self, x):
        return self.encode(x)

class Decoder(keras.layers.Layer):
    def __init__(self, out_channels, kernel_size, use_bias):
        super(Decoder, self).__init__()
        self.decode = keras.Sequential([
            keras.layers.BatchNormalization(epsilon=BN_EPS),
            
            
            ReflectionPadding2D(((kernel_size - 1) // 2, (kernel_size - 1) // 2)),
            keras.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, strides=1, padding='valid', use_bias=use_bias),
            keras.layers.BatchNormalization(epsilon=BN_EPS),
            keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA),
            
            ReflectionPadding2D(((kernel_size - 1) // 2, (kernel_size - 1) // 2)),
            keras.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, strides=1, padding='valid', use_bias=use_bias),
            keras.layers.BatchNormalization(epsilon=BN_EPS),
            keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA),
            
            keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        ])
        
    def call(self, x, skip_tensor):
        
        if skip_tensor is None:
            return self.decode(x)
        
        # Calculate cropping for skip_tensor to concatenate with x
        _, h2, w2, _ = skip_tensor.shape
        _, h1, w1, _ = x.shape
        h_diff, w_diff = h2 - h1, w2 - w1
        
        cropping = ((int(np.ceil(h_diff / 2)), int(np.floor(h_diff / 2))),
                    (int(np.ceil(w_diff / 2)), int(np.floor(w_diff / 2))))
        
        skip_tensor = keras.layers.Cropping2D(cropping=cropping)(skip_tensor)        
        x = keras.layers.concatenate([x, skip_tensor], axis=3)
        
        return self.decode(x)
    
    
class Skip(keras.layers.Layer):
    def __init__(self, out_channels, kernel_size, use_bias):
        super(Skip, self).__init__()
        self.skip = keras.Sequential([
            ReflectionPadding2D(((kernel_size - 1) // 2, (kernel_size - 1) // 2)),
            keras.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, strides=1, padding='valid', use_bias=use_bias),
            keras.layers.BatchNormalization(epsilon=BN_EPS),
            keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)
        ])
        
    def call(self, x):
        return self.skip(x)

    
class ReflectionPadding2D(keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [keras.layers.InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')
    
    

# class Concat(keras.layers.Layer):
#     def __init__(self, skip, deeper):
#         super(Concat, self).__init__()
#         self.skip = skip
#         self.deeper = deepr
        
#     def call(self, x):
#         x_skip = self.skip(x)
#         x_deeper = self.deeper(x)
        
#         _, h1, w1, _ = x_skip.shape
#         _, h2, w2, _ = x_deeper.shape
        
#         h_diff, w_diff = h2 - h1, w2 - w1
#         cropping = ((int(np.ceil(h_diff / 2)), int(np.floor(h_diff / 2))),
#                     (int(np.ceil(w_diff / 2)), int(np.floor(w_diff / 2))))
        
#         x_skip = keras.layers.Cropping2D(cropping=cropping)(x_skip)        
#         return keras.layers.concatenate([x_skip, x_deeper], axis=3)



    
# def act(act_fun = 'LeakyReLU'):
#     '''
#     Either string defining an activation function or module (e.g. nn.ReLU)
#     '''
#     if isinstance(act_fun, str):
#         if act_fun == 'LeakyReLU':
#             return keras.layers.LeakyReLU(alpha=0.2)
#         elif act_fun == 'ELU':
#             return keras.layers.ELU()
#         else:
#             assert False
#     else:
#         return act_fun()
    
    
# def bn():
#     return keras.layers.BatchNormalization(epsilon=1e-5)

# def conv(out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
#     downsampler = None
#     if stride != 1 and downsample_mode != 'stride':
#         if downsample_mode == 'avg':
#             downsampler = keras.layers.AveragePooling2D(stride, stride)
#         elif downsample_mode == 'max':
#             downsampler = keras.layers.MaxPool2D(stride, stride)
# #         elif downsample_mode  in ['lanczos2', 'lanczos3']:
# #             downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True)
#         else:
#             assert False

#         stride = 1

#     padder = None
#     (kernel_size - 1) // 2 = int((kernel_size - 1) / 2)
#     if pad == 'reflection':
#         padder = ReflectionPad2d(((kernel_size - 1) // 2, (kernel_size - 1) // 2))
#         (kernel_size - 1) // 2 = 0
  
#     convolver = keras.layers.Conv2d(filters=out_f, kernel_size=kernel_size, strides=stride, padding=(kernel_size - 1) // 2, use_bias=bias)


#     layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
#     return keras.Sequential(layers)
