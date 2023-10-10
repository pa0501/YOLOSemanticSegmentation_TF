import keras.layers
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

#from tensorflow.python import keras
#from keras import layers

# TF implementation
class Conv(layers.Layer):
    def __init__(
            self,
            output_channel,
            kernel_size=1,
            strides=1,
            padding='same',
            activation='swish'):
        super().__init__()
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides
        self.padding = padding
        self.conv = layers.Conv2D(
            filters=self.output_channel,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=False)
        self.bn = layers.BatchNormalization()

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_channel": self.output_channel,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "activation": self.activation,
            "convolution": self.conv,
            "batch normalization": self.bn
        })
        return config

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = layers.Activation(self.activation)(x)
        return x

class DeConv(layers.Layer):
    def __init__(
            self,
            output_channel,
            kernel_size=1,
            strides=1,
            padding='same',
            activation='swish'):
        super().__init__()
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides
        self.padding = padding
        self.conv = layers.Conv2DTranspose(
            filters=self.output_channel,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=False)
        self.bn = layers.BatchNormalization()

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_channel": self.output_channel,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "activation": self.activation,
            "convolution": self.conv,
            "batch normalization": self.bn
        })
        return config

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = layers.Activation(self.activation)(x)
        return x

class Bottleneck(layers.Layer):
    # Standard bottleneck
    def __init__(
            self,
            output_channel,
            shortcut=True,
            kernel_size=(3, 3),
            expansion=0.5):
        super().__init__()
        self.c2 = output_channel
        self.kernel_size = kernel_size
        self.e = expansion
        self.c1 = int(self.c2 * self.e)
        self.cv1 = Conv(output_channel=self.c1, kernel_size=kernel_size[0])
        self.cv2 = Conv(output_channel=self.c2, kernel_size=kernel_size[1])
        self.add = shortcut

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_channel": self.c2,
            "kernel_size": self.kernel_size,
            "shortcut": self.add,
            "expansion": self.e,
            "c1": self.c1,
            "conv1": self.cv1,
            "conv2": self.cv2
        })
        return config

    def call(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        return x + x2 if self.add else x2



class C2f(layers.Layer):
    # CSP Bottleneck with 2 convolutions
    def __init__(
            self,
            output_channel,
            repeat=1,
            shortcut=False,
            expansion=0.5):
        super().__init__()
        self.c2 = output_channel
        self.e = expansion
        self.n = repeat
        self.c = int(self.c2 * self.e)  # hidden channels
        self.cv1 = Conv(output_channel=2 * self.c, kernel_size=1)
        self.cv2 = Conv(output_channel=self.c2, kernel_size=1)
        self.m = [Bottleneck(self.c,
                             shortcut,
                             kernel_size=((3, 3), (3, 3)),
                             expansion=1.0)
                  for _ in range(self.n)]

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_channel": self.c2,
            "hidden channel": self.c,
            "repeat": self.n,
            "bottleneck": self.m,
            "expansion": self.e,
            "conv1": self.cv1,
            "conv2": self.cv2
        })
        return config

    def call(self, x):
        y = tf.split(self.cv1(x),
                     num_or_size_splits=2,
                     axis=3)  # channel axis
        y.extend([m(y[-1]) for m in self.m])
        out = self.cv2(tf.concat(y, axis=3))
        return out


class SPPF(layers.Layer):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, input_channel, output_channel, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        self.c_ = input_channel // 2  # hidden channels
        self.c2 = output_channel
        self.k = k
        self.cv1 = Conv(output_channel=self.c_, kernel_size=1)
        self.cv2 = Conv(output_channel=self.c2, kernel_size=1)
        self.m = layers.MaxPool2D(pool_size=self.k,
                                  strides=1,
                                  padding='same')

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_channel": self.c2,
            "hidden channel": self.c_,
            "conv1": self.cv1,
            "conv2": self.cv2,
            "maxpool kernel": self.k,
            "maxpool": self.m
        })
        return config

    def call(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(tf.concat([x, y1, y2, self.m(y2)], axis=3))

class YOLOv8_BaseModel(tf.keras.Model):
    """ ***YOLO Semantic Segmentation Model***
    A combination of YOLOv8 and YOLO-C model.
    Input channel must be a multiple of 32.
    Backbone and Neck: https://arxiv.org/abs/2304.00501.
    Feature Fusion: https://arxiv.org/abs/1712.00960.
    Segment Head: https://www.mdpi.com/2077-1312/11/7/1475.
    """
    def __init__(self, shape_in, nc=4):
        super().__init__()
        # Backbone
        self.shape_in = shape_in  # NHWC
        self.nc = nc  # number of classes
        self.inputs = tf.keras.layers.Input(shape=self.shape_in)
        self.cv1 = Conv(output_channel=64, kernel_size=3, strides=2)  # p1
        self.cv2 = Conv(output_channel=128, kernel_size=3, strides=2)  # p2
        self.c2f1 = C2f(output_channel=128, repeat=3, shortcut=True)
        self.cv3 = Conv(output_channel=256, kernel_size=3, strides=2)  # p3
        self.c2f2 = C2f(output_channel=256, repeat=6, shortcut=True)
        self.cv4 = Conv(output_channel=512, kernel_size=3, strides=2)  # p4 c2=512 x w
        self.c2f3 = C2f(output_channel=512, repeat=6, shortcut=True)  # c2=512 x w
        self.cv5 = Conv(output_channel=512, kernel_size=3, strides=2)  # p5 c2=512 x w x r
        self.c2f4 = C2f(output_channel=512, repeat=3, shortcut=True)
        self.sppf = SPPF(input_channel=512, output_channel=512, k=5)  # k: maxpool size; c2=512 x w x r

        self.seq1 = tf.keras.Sequential([
            self.c2f1,
            self.cv3,
            self.c2f2
        ])
        self.seq2 = tf.keras.Sequential([
            self.cv4,
            self.c2f3
        ])
        self.seq3 = tf.keras.Sequential([
            self.cv5,
            self.c2f4,
            self.sppf
        ])

        # Neck
        self.upsample = layers.UpSampling2D(size=2, interpolation='bilinear')
        # self.upsample = lambda x: tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 2), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        self.c2f5 = C2f(output_channel=512, repeat=3, shortcut=False)  # - c2=512 x w
        self.c2f6 = C2f(output_channel=256, repeat=3, shortcut=False)  # p3
        self.cv6 = Conv(output_channel=256, kernel_size=3, strides=2)  # p3
        self.c2f7 = C2f(output_channel=512, repeat=3, shortcut=False)  # p4 - c2=512 x w
        self.cv7 = Conv(output_channel=512, kernel_size=3, strides=2)  # p4 - c2=512 x w
        self.c2f8 = C2f(output_channel=512, repeat=3, shortcut=False)  # p5 - c2=512 x w x r

        # Feature Pyramid Generator
        self.cv8 = Conv(output_channel=512, kernel_size=3, strides=2)  # 1st stage of feature pyramid
        self.cv9 = Conv(output_channel=1024, kernel_size=3, strides=2)  # 2nd stage of feature pyramid

        # Feature Decoder
        self.dcv1 = DeConv(output_channel=512, kernel_size=3, strides=2)  #
        self.c2f9 = C2f(output_channel=512, repeat=3, shortcut=True)

        self.dcv2 = DeConv(output_channel=512, kernel_size=3, strides=2)  #
        self.c2f10 = C2f(output_channel=512, repeat=3, shortcut=True)

        self.dcv3 = DeConv(output_channel=256, kernel_size=3, strides=2)  #
        self.c2f11 = C2f(output_channel=256, repeat=3, shortcut=True)

        self.dcv4 = DeConv(output_channel=128, kernel_size=3, strides=2)  #
        self.c2f12 = C2f(output_channel=128, repeat=3, shortcut=True)

        self.dcv5 = DeConv(output_channel=64, kernel_size=3, strides=2)  #
        self.c2f13 = C2f(output_channel=64, repeat=3, shortcut=True)

        self.cv0 = Conv(output_channel=self.nc, kernel_size=1, strides=1)

def Yolov8(input_shape, nc=4):
    m = YOLOv8_BaseModel(input_shape, nc=nc)
    img_height = input_shape[0]
    img_width = input_shape[1]

    #Backbone: CSPDarknet53 feature extractor
    p1 = m.cv1(m.inputs)  # p1 h/2
    p2 = m.cv2(p1)  # p2 h/4
    xs1 = m.seq1(p2)  # p3 h/8
    xs2 = m.seq2(xs1) # p3 h/16
    xs3 = m.seq3(xs2)  #  Base h/32

    # Neck
    '''Bottom-up path augmentation. P: feature map before, N: feature map after
                Ni+1 = Pi+1 + Ni.conv(k=3, s=2, p="same")
    '''
    xs4 = m.c2f5(tf.concat([m.upsample(xs3), xs2], axis=3)) # P4 h/16 up-sample 1
    p3 = m.c2f6(tf.concat([m.upsample(xs4), xs1], axis=3)) # P3 h/8   up-sample 2
    p4 = m.c2f7(tf.concat([m.cv6(p3), xs4], axis=3))  # P4 h/16     down-sample 1
    p5 = m.c2f8(tf.concat([m.cv7(p4), xs3], axis=3))  # P5 h/32     down-sample 2

    # Feature Fusion + Pyramid Feature Generation
    '''1. Concatenate features from the neck (with channel=256)
       2. Generate a series of pyramid features (3 at P3, P4, P5)
    '''
    p5 = layers.UpSampling2D(size=4, interpolation='bilinear')(layers.Conv2D(256, (1, 1))(p5))
    p4 = layers.UpSampling2D(size=2, interpolation='bilinear')(layers.Conv2D(256, (1, 1))(p4))
    f0 = layers.BatchNormalization()(tf.concat([p3, p4, p5], axis=3))  # Fused feature  p3
    f0 = layers.Conv2D(256, (1, 1))(f0)
    f1 = m.cv8(f0)  # p4
    f2 = m.cv9(f1)  # p5

    # Segment Head
    ''' Based on U-Net, FCN decoder architecture
        Use features to progressively up-sample and concatenate into final output: f2 -> f1 -> f0 -> p2 -> p1 -> p0
        (steps are mirrored to the backbone):
            a. In backbone, conv with s = 2 to down-sample
            b. In segment head, transposed conv with s = 2 to up-sample'''

    p0 = m.c2f9(m.dcv1(f2))  # p4 h/16
    p0 = m.c2f10(m.dcv2(tf.concat([p0, f1], axis=3)))  # p3 h/8
    p0 = m.c2f11(m.dcv3(tf.concat([p0, f0], axis=3)))  # p2 h/4
    p0 = m.c2f12(m.dcv4(tf.concat([p0, p2], axis=3)))  # p1 h/2
    p0 = m.c2f13(m.dcv5(tf.concat([p0, p1], axis=3)))  # p0

    outputs = m.cv0(p0)
    outputs = tf.keras.layers.Softmax()(outputs)
    yolo_model = tf.keras.models.Model(inputs=m.inputs, outputs=outputs, name='YOLOv8-Seg')
    return yolo_model



def test(x):
    return layers.Conv2DTranspose(
                                filters=512,
                                kernel_size=3,
                                strides=2,
                                padding="same",
                                use_bias=False)(x)

if __name__ == "__main__":
    input_shape = (256, 256, 1)
    batch_size = 4
    input_tensor = tf.random.normal((batch_size,) + input_shape)
    nclasses = 4
    model = Yolov8(input_shape, nc=nclasses)
    # img = tf.convert_to_tensor(np.expand_dim(np.zeros((1024, 1024)), axis=-1)) #-> output: 1024x1024x4
    # Print model summary
    # model.summary()


    #test_shape = (256, 256, 768)

    #input_test = tf.random.normal((batch_size,) + test_shape)
    #out_test = test(input_test)
    # Generate random input tensor


    # Pass input_tensor through the model
    output = model(input_tensor)
    print(output)
