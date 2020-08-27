import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

from deep_image_prior.src.layers import *

class SkipUnet(keras.Model):
    def __init__(self):
        super(SkipUnet, self).__init__()
        self.down1 = Encoder(out_channels=8, kernel_size=3, use_bias=False)
        self.down2 = Encoder(out_channels=16, kernel_size=3, use_bias=True)
        self.down3 = Encoder(out_channels=32, kernel_size=3, use_bias=True)
        self.down4 = Encoder(out_channels=64, kernel_size=3, use_bias=True)
        self.down5 = Encoder(out_channels=128, kernel_size=3, use_bias=True)
        
#         self.skip1 = Skip(out_channels=4, kernel_size=1, use_bias=True)
#         self.skip2 = Skip(out_channels=4, kernel_size=1, use_bias=True)
#         self.skip3 = Skip(out_channels=4, kernel_size=1, use_bias=True)
        self.skip4 = Skip(out_channels=4, kernel_size=1, use_bias=True)
        self.skip5 = Skip(out_channels=4, kernel_size=1, use_bias=True)
        
        self.up5 = Decoder(out_channels=128, kernel_size=3, use_bias=True)
        self.up4 = Decoder(out_channels=64, kernel_size=3, use_bias=True)
        self.up3 = Decoder(out_channels=32, kernel_size=3, use_bias=True)
        self.up2 = Decoder(out_channels=16, kernel_size=3, use_bias=True)
        self.up1 = Decoder(out_channels=8, kernel_size=3, use_bias=True)
        
        self.last = keras.layers.Conv2D(filters=3, kernel_size=3, strides=1, padding='same', use_bias=True)
        self.classify = keras.layers.Activation('sigmoid')
        
    def call(self, x):
        x = self.down1(x)
#         skip_tensor1 = self.skip1(x)
        skip_tensor1 = None
        
        x = self.down2(x)
#         skip_tensor2 = self.skip2(x)
        skip_tensor2 = None
        
        x = self.down3(x)
#         skip_tensor3 = self.skip3(x)
        skip_tensor3 = None
        
        x = self.down4(x)
        skip_tensor4 = self.skip4(x)
        
        x = self.down5(x)
        skip_tensor5 = self.skip5(x)
        
        x = self.up5(x, skip_tensor5)
        x = self.up4(x, skip_tensor4)
        x = self.up3(x, skip_tensor3)
        x = self.up2(x, skip_tensor2)
        x = self.up1(x, skip_tensor1)
        
        x = self.last(x)
        x = self.classify(x)
        
        return x
    

        
        

# def skip(
#         num_input_channels=2, num_output_channels=3, 
#         num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
#         filter_size_down=3, filter_size_up=3, filter_skip_size=1,
#         need_sigmoid=True, need_bias=True, 
#         pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', 
#         need1x1_up=True):
#     """Assembles encoder-decoder with skip connections.
#     Arguments:
#         act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
#         pad (string): zero|reflection (default: 'zero')
#         upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
#         downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')
#     """
#     assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

#     n_scales = len(num_channels_down) 

#     if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
#         upsample_mode   = [upsample_mode]*n_scales

#     if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
#         downsample_mode   = [downsample_mode]*n_scales
    
#     if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
#         filter_size_down   = [filter_size_down]*n_scales

#     if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
#         filter_size_up   = [filter_size_up]*n_scales

#     last_scale = n_scales - 1 

#     cur_depth = None

#     model = keras.Sequential()
#     model_tmp = model

#     input_depth = num_input_channels
#     for i in range(len(num_channels_down)):

#         deeper = keras.Sequential()
#         skip = keras.Sequential()

#         if num_channels_skip[i] != 0:
#             model_tmp.add(Concat(skip, deeper))
#         else:
#             model_tmp.add(deeper)
        
#         model_tmp.add(bn())

#         if num_channels_skip[i] != 0:
#             skip.add(conv(num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
#             skip.add(bn())
#             skip.add(act(act_fun))
            
#         # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

#         deeper.add(conv(num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
#         deeper.add(bn())
#         deeper.add(act(act_fun))

#         deeper.add(conv(num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
#         deeper.add(bn())
#         deeper.add(act(act_fun))

#         deeper_main = keras.Sequential()

#         if i != len(num_channels_down) - 1:
#             deeper.add(deeper_main)

#         deeper.add(keras.layers.UpSampling2D(size=(2, 2), mode=upsample_mode[i]))

#         model_tmp.add(conv(filter_size_up[i], 1, bias=need_bias, pad=pad))
#         model_tmp.add(bn())
#         model_tmp.add(act(act_fun))


#         if need1x1_up:
#             model_tmp.add(conv( num_channels_up[i], 1, bias=need_bias, pad=pad))
#             model_tmp.add(bn())
#             model_tmp.add(act(act_fun))

#         input_depth = num_channels_down[i]
#         model_tmp = deeper_main

#     model.add(conv(num_output_channels, 1, bias=need_bias, pad=pad))
#     if need_sigmoid:
#         model.add(keras.layers.Activation('sigmoid'))

#     return model


