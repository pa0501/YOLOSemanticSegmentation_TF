import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.nn as nn
from keras import layers
import numpy as np
#TF implementation
class Conv(layers.Layer):
    def __init__(
        self,
        output_channel,
        kernel_size=1,
        strides=1,
        activation="swish"):
        
        super().__init__()

        self.kernel_size = kernel_size
        self.activation = activation
        
        self.conv = layers.Conv2D(filters=output_channel,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding="valid",
                                  use_bias=False)
        
        self.bn = layers.BatchNormalization(momentum=BATCH_NORM_MOMENTUM,
                                            epsilon=BATCH_NORM_EPSILON)
    
    def call(self, x):
        if self.kernel_size > 1:
            inputs = layers.ZeroPadding2D(padding=kernel_size//2)(x)

        x = self.bn(self.conv(inputs))
        x = layers.Activation(self.activation)(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "output_channel": self.output_channel,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "activation": self.activation
        })
        return config
    
class Bottleneck(layers.Layer):
    # Standard bottleneck
    def __init__(self,
                 output_channel,
                 shortcut=None,
                 k=(3, 3),
                 e=0.5):
        super().__init__()
        self.c2 = output_channel
        self.c1 = int(output_channel=self.c2 * e)
        self.cv1 = Conv(output_channel=self.c1, kernel_size=k[0])
        self.cv2 = Conv(output_channel=self.c2, kernel_size=k[1])
        self.add = shortcut
        
    def call(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        return x + x2 if self.add else x2
    
class C2f(layers.Layer):
    # CSP Bottleneck with 2 convolutions
    def __init__(self,
                 output_channel,
                 repeat=1,
                 shortcut=None,
                 e=0.5):
        super().__init__()
        self.c2 = output_channel
        self.c = int(self.c2 * e) # hidden channels
        self.cv1 = Conv(output_channel=2*self.c, kernel_size=1)
        self.cv2 = Conv(output_channel=self.c2, kernel_size=1)
        self.m = [Bottleneck(self.c,
                             shortcut,
                             kernel_size=((3, 3), (3, 3)),
                             e=1.0)
                  for _ in range(repeat)]

    def call(self, x):
        y = tf.split(self.cv1(x),
                     num_or_size_splits=2,
                     axis=-1) #channel axis
        y.extend([m(y[-1]) for m in self.m])
        return self.cv2(tf.concat(y, axis=-1))

class SPPF(layers.Layer):
     # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(output_channel=c_, kernel_size=1)
        self.cv2 = Conv(output_channel=c2, kernel_size=1)
        self.m = layers.MaxPool2D(pool_size=k, strides=1, padding='same')

    def call(self, x, training=None):
        self.c1 = tf.shape(x)[-1] #the input channel
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(tf.concat([x, y1, y2, self.m(y2)], axis=3))

class DFL(layers.Layer):
     #Integral module of Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.c1 = c1
        x = tf.range(c1, dtype=tf.float32)
        weights_ini = tf.reshape(x, (1, 1, self.c1, 1)) #weights initialization (kernel_h, kernel_w, in_ch, out_ch)
        self.conv = layers.Conv2D(filters=1, kernel_size=1, use_bias=False, trainable=False, weights = [weights_ini])
        
    def call(self, x):
        b, c, a = x.shape   # batch, channels, anchors (1, 9600, 64)
        #print("b, a, c:", b, a, c)
        x_reshaped = tf.reshape(x, (b, a, 4, self.c1)) #(batch, anchors, 4, c1)
        conv_output = self.conv(tf.nn.softmax(x_reshaped, axis=3))
        return tf.reshape(conv_output, (b, 4, a))

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points = []
    stride_tensor = []  
    assert feats is not None
    dtype = feats[0].dtype
    for i, stride in enumerate(strides):
        _, h, w, _ = feats[i].shape
        sx = tf.range(start=grid_cell_offset,
                      limit=w + grid_cell_offset,
                      dtype=dtype)  # shift x
        sy = tf.range(start=grid_cell_offset,
                      limit=h + grid_cell_offset,
                      dtype=dtype)  # shift y
        sy, sx = tf.meshgrid(sy, sx, indexing='ij')
        stacked_points = tf.stack((sx, sy), axis=-1)
        anchor_points.append(tf.reshape(stacked_points, (-1, 2)))
        stride_tensor.append(tf.fill((h * w, 1), stride))
        
    return tf.concat(anchor_points, axis=0), tf.concat(stride_tensor, axis=0)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = tf.split(distance, 2, axis=dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return tf.concat((c_xy, wh), axis=dim)  # xywh bbox
    return tf.concat((x1y1, x2y2), axis=dim)  # xyxy bbox


class Detect(layers.Layer):
    #YOLOv8 Detect head for detection models
    """Concatenates and returns predicted bounding boxes and class probabilities."""
    shape = None
    anchors = []
    strides = []
    
    def __init__(self, nc=80, ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = tf.Variable(initial_value=tf.zeros(self.nl))  # strides computed during build

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels

        self.cv2 = [tf.keras.Sequential([
            Conv(output_channel=c2, kernel_size=3),
            Conv(output_channel=c2, kernel_size=3),
            layers.Conv2D(filters=(4*self.reg_max),
                          kernel_size=1,
                          strides=1,
                          padding='valid')
        ]) for _ in ch]

        self.cv3 = [tf.keras.Sequential([
            Conv(output_channel=c3, kernel_size=3),
            Conv(output_channel=c3, kernel_size=3),
            layers.Conv2D(filters=self.nc,
                          kernel_size=1,
                          strides=1,
                          padding='valid')
        ]) for _ in ch]
        
        if self.reg_max > 1:
            self.dfl = DFL(self.reg_max)
        else:
            self.dfl = tf.keras.layers.Identity()
                 
    def call(self, x, training=False):
        shape = x[0].shape  # BHWC

        for i in range(self.nl):
            x[i] = tf.concat((self.cv2[i](x[i]), self.cv3[i](x[i])), 3) #1, h, w, 144
        if training:
            return x
        elif self.shape != shape:
            self.anchors, self.strides = (tf.transpose(_, perm=(1, 0)) for _ in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        bs = shape[0]
        
        concatenated_xi = []
        for xi in x:
            xi_reshaped = tf.reshape(xi, (bs, self.no, -1))
            concatenated_xi.append(xi_reshaped)
        concatenated_tensor = tf.concat(concatenated_xi, axis=2) #1, 144, 8400 
        #print("concatenated_tensor+FT:", concatenated_tensor)
        
        splits = tf.split(concatenated_tensor, (self.reg_max * 4, self.nc), axis=1) #[1, 64, 8400], [1, 80, 8400]
        box, cls = splits
        dbox = dist2bbox(self.dfl(box), tf.expand_dims(self.anchors, axis=0), xywh=True, dim=1) * self.strides
        y = tf.concat([dbox, tf.sigmoid(cls)], axis=1)
        return (y, x)

class Proto(layers.Layer):
    # YOLOv8 mask Proto module for segmentation models
    def __init__(self, c_=256, c2=32):# number of protos, number of masks
        super(Proto, self).__init__()
        self.cv1 = Conv(c_, kernel_size=3)
        self.upsample = layers.Conv2DTranspose(filters=c_, kernel_size=2, strides=2, padding='valid')
        self.cv2 = Conv(output_channel=c_, kernel_size=3)
        self.cv3 = Conv(output_channel=c2, kernel_size=1)

    def call(self, x):
        x = self.cv1(x)
        x = self.upsample(x)
        x = self.cv2(x)
        x = self.cv3(x)
        return x

class Segment(Detect):
    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(self.npr, self.nm)   # protos
        self.detect = Detect.call
        
        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = [tf.keras.Sequential([
            Conv(output_channel=c4, kernel_size=3),
            Conv(output_channel=c4, kernel_size=3),
            tf.keras.layers.Conv2D(filters=self.nm,
                                   kernel_size=1,
                                   strides=1,
                                   padding='valid')
        ]) for x in ch]

    def call(self, x, training=None):
        p = self.proto(x[0], training=training)  # mask protos
        bs = p.shape[0]  # batch size
        mc_parts = []
        for i in range(self.nl):
        # Convolution and reshape operations
            conv_result = self.cv4[i](x[i], training=training)
            reshaped_result = tf.reshape(conv_result, (bs, self.nm, -1))
        # Add the processed result to the list
            mc_parts.append(reshaped_result)
            
        mc = tf.concat(mc_parts, axis = 2) # mask coefficients
        x = self.detect(self, x, training=training)
        print("training:", training)
        if training:
           return x, mc, p
        return (tf.concat([x[0], mc], axis=2), (x[1], mc, p))
  
class Yolov8Seg_Model(tf.keras.Model):
    #Yolov8 Segmentation Model, input channel must be a multiple of 32
    #Including Segmentation head with encoding/post-processing methode
    #Result should be
    #Base model structure based on https://arxiv.org/abs/2304.00501
    def __init__(self, input_shape, nc=4, training=None):
        super(Yolov8Seg_Model, self).__init__()
        # Backbone
        #self.inputs = tf.keras.layers.Input(shape=input_shape[1:], batch_size = input_shape[0])
        self.nc = nc # number of classes
        self.training = training
        self.inputs = tf.keras.Input(shape=input_shape)
        self.cv1 = Conv(c2=64, k=3, s=2, p='same')  #p1
        self.cv2 = Conv(c2=128, k=3, s=2, p='same')  #p2
        self.c2f1 = C2f(c2=128, n=3, shortcut=True)
        self.cv3 = Conv(c2=256, k=3, s=2, p='same')  #p3
        self.c2f2 = C2f(c2=256, n=6, shortcut=True)
        self.cv4 = Conv(c2=512, k=3, s=2, p='same')  #p4 c2=512 x w
        self.c2f3 = C2f(c2=512, n=6, shortcut=True)  #   c2=512 x w
        self.cv5 = Conv(c2=512, k=3, s=2, p='same')  #p5 c2=512 x w x r
        self.c2f4 = C2f(c2=512, n=3, shortcut=True)
        self.sppf = SPPF(c1=512, c2=512, k=5)  #k: maxpool size; c2=512 x w x r

        self.seq1 = tf.keras.Sequential([
            self.cv1,
            self.cv2,
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
        # Head
        self.upsample = keras.layers.UpSampling2D(size=2, interpolation='bilinear')
        #self.upsample = lambda x: tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 2), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        self.c2f5 = C2f(c2=512, n=3, shortcut=False) #   - c2=512 x w
        self.c2f6 = C2f(c2=256, n=3, shortcut=False) #p3
        self.cv6 = Conv(c2=256, k=3, s=2, p='same')  #p3
        self.c2f7 = C2f(c2=512, n=3, shortcut=False) #p4 - c2=512 x w
        self.cv7 = Conv(c2=512, k=3, s=2, p='same')  #p4 - c2=512 x w
        self.c2f8 = C2f(c2=512, n=3, shortcut=False) #p5 - c2=512 x w x r
        self.segment_head = lambda nc, ch: Segment(nc=nc, ch=ch)
        
    def call(self, inputs): 
        x1 = self.seq1(inputs, training=self.training)
        x2 = self.seq2(x1, training=self.training)
        x3 = self.seq3(x2, training=self.training)
        # Head
        x4 = self.c2f5(tf.concat([model.upsample(x3), x2], axis=3), training=self.training)
        xs1 = self.c2f6(tf.concat([model.upsample(x4), x1], axis=3), training=self.training)
        xs2 = self.c2f7(tf.concat([model.cv6(xs1, training=self.training), x4], axis=3), training=self.training)
        xs3 = self.c2f8(tf.concat([model.cv7(xs2, training=self.training), x3], axis=3), training=self.training)
        # Segmentation
        ch = [xs1.shape[-1], xs2.shape[-1], xs3.shape[-1]]
        seg_inputs = [xs1, xs2, xs3]
        segment = self.segment_head(nc=self.nc, ch=ch)
        outputs = segment(seg_inputs, training=self.training)
        return outputs
        

        
def Yolov8_Seg(input_shape, nc=80, training=None):
    model = Yolov8Seg_Model(input_shape)
    print("modelinput", tf.shape(model.inputs))
    x1 = model.seq1(model.inputs, training=training)
    x2 = model.seq2(x1, training=training)
    x3 = model.seq3(x2, training=training)
    # Head
    x4 = model.c2f5(tf.concat([model.upsample(x3), x2], axis=3), training=training)
    xs1 = model.c2f6(tf.concat([model.upsample(x4), x1], axis=3), training=training)
    xs2 = model.c2f7(tf.concat([model.cv6(xs1, training=training), x4], axis=3), training=training)
    xs3 = model.c2f8(tf.concat([model.cv7(xs2, training=training), x3], axis=3), training=training)
    # Segmentation
    ch = [xs1.shape[-1], xs2.shape[-1], xs3.shape[-1]]
    seg = model.segment(nc=nc, ch=ch)
    outputs = (xs1, xs2, xs3)
    
    segment_model = tf.keras.models.Model(inputs=model.inputs, outputs=outputs, name='Yolov8-Segmentation')
    return segment_model


    
if __name__ == "__main__":
    input_shape = (800, 800, 3)
    #input_tensor = tf.keras.layers.Input(shape=input_shape)
    nclasses = 4 
    model = Yolov8Seg_Model(input_shape, nc=nclasses, training=False)
    #img = tf.convert_to_tensor(np.expand_dim(np.zeros((1024, 1024)), axis=-1)) #-> output: 1024x1024x4
    # Print model summary
    #model.summary()

    # Generate random input tensor
    batch_size = 4
    input_tensor = tf.random.normal((batch_size,) + input_shape)

    # Pass input_tensor through the model
    output = model(input_tensor)
    print(output)
