import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.nn as nn
from keras.layers import Layer

#TF implementation
class Conv(Layer):
    # Standard convolution with args(ch_out, kernel, stride, padding, group, dilation, activation). Default data format is channel last

    default_act = tf.keras.layers.Activation(tf.keras.activations.swish)  # default activation
    
    def __init__(self, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()

        self.conv = keras.layers.Conv2D(filters=c2, kernel_size=k, strides=s, padding=p, groups=g, dilation_rate=d, use_bias=False)
        self.bn = keras.layers.BatchNormalization()
        
        if act is True:
            self.act = self.default_act
        elif isinstance(act, keras.layers.Layer):
            self.act = act
        else:
            self.act = keras.layers.Activation(act) if act is not None else keras.layers.Identity()

    def call(self, x, training=None):
        return self.act(self.bn(self.conv(x), training=training))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

    
class Bottleneck(Layer):
    # Standard bottleneck
    def __init__(self, c2, shortcut=None, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c_, k[0], s = 1, p='same')
        self.cv2 = Conv(c2, k[1],s = 1, p='same', g=g)
        self.add = shortcut
        
    def call(self, x, training=None):
        x1 = self.cv1(x, training=training)
        x2 = self.cv2(x1, training=training)
        return x + x2 if self.add else x2
    
class C2f(Layer):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c2, n=1, shortcut=None, g=1, e=0.5): #ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e) # hidden channels
        self.cv1 = Conv(2 * self.c, k=1, s=1, p='valid')
        self.cv2 = Conv(c2, k=1, s=1, p='valid')
        self.m = [Bottleneck(self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)]

    def call(self, x, training=None):
        y = tf.split(self.cv1(x, training=training), num_or_size_splits=2, axis=-1) #split at channel axis
        y.extend([m(y[-1], training=training) for m in self.m])
        return self.cv2(tf.concat(y, axis=-1), training=training)

class SPPF(Layer):
     # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c_, k=1,s=1,p='valid')
        self.cv2 = Conv(c2, k=1,s=1,p='valid')
        self.m = tf.keras.layers.MaxPool2D(pool_size=k, strides=1, padding='same')

    def call(self, x, training=None):
        self.c1 = x.shape[-1] #the input channel
        x = self.cv1(x, training=training)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(tf.concat([x, y1, y2, self.m(y2)], axis=3), training=training)

class DFL(Layer):
     #Integral module of Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        x = tf.range(c1, dtype=tf.float32)
        weights_ini = tf.reshape(x, (1, 1, c1, 1)) #weights initialization (kernel_h, kernel_w, in_ch, out_ch)
        self.conv = tf.keras.layers.Conv2D(filters=1, kernel_size=1, use_bias=False, trainable=False, weights = [weights_ini])
        self.c1 = c1

    def call(self, x):
        b, a, c = x.shape   # batch, anchors, channels (1, 9600, 64)
        x_reshaped = tf.reshape(x, (b, a, 4, self.c1)) #(batch, anchors, 4, c1)
        conv_output = self.conv(tf.nn.softmax(x_reshaped, axis=3))
        return tf.reshape(conv_output, (b, 4, a))
        #return conv_output
    
def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points = []
    stride_tensor = []  
    assert feats is not None
    dtype = feats[0].dtype
    for i, stride in enumerate(strides):
        _, h, w, _ = feats[i].shape
        sx = tf.range(start=grid_cell_offset, limit=w + grid_cell_offset, dtype=dtype)  # shift x
        sy = tf.range(start=grid_cell_offset, limit=h + grid_cell_offset, dtype=dtype)  # shift y
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
    #return tf.concat((x1y1, x2y2), axis=dim)  # xyxy bbox
    return lt

class Detect(Layer):
    #YOLOv8 Detect head for detection models
    """Concatenates and returns predicted bounding boxes and class probabilities."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
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
            Conv(c2, k=3, s=1, p='same'),
            Conv(c2, k=3, s=1, p='same'),
            tf.keras.layers.Conv2D(filters=(4*self.reg_max), kernel_size=1, strides=1, padding='valid')
        ]) for x in ch]

        self.cv3 = [tf.keras.Sequential([
            Conv(c3, k=3, s=1, p='same'),
            Conv(c3, k=3, s=1, p='same'),
            tf.keras.layers.Conv2D(filters=self.nc, kernel_size=1, strides=1, padding='valid')
        ]) for x in ch]
        
        if self.reg_max > 1:
            self.dfl = DFL(self.reg_max)
        else:
            self.dfl = tf.keras.layers.Identity()
                 
    def call(self, x, training=None):
        shape = x[0].shape  # BHWC
        #print("shape_x0:", shape)
        #print("shape_cv2:", self.cv2[0](x[0]))
        for i in range(self.nl):
            x[i] = tf.concat((self.cv2[i](x[i], training=training), self.cv3[i](x[i], training=training)), 3) #1, h, w, 144
        if training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (tf.transpose(x, perm=(1, 0)) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
            
        concatenated_xi = []
        for xi in x:
            xi_reshaped = tf.reshape(xi, (shape[0], -1, self.no))
            concatenated_xi.append(xi_reshaped)
        concatenated_tensor = tf.concat(concatenated_xi, axis=1) #1, 8400, 144 /channel last
        #print("concatenated_tensor+FT:", concatenated_tensor)
        
        splits = tf.split(concatenated_tensor, (self.reg_max * 4, self.nc), axis=2) #[1, 8400, 64], [1, 8400, 80]
        #print("Splits shapes:", [split.shape for split in splits])
        box, cls = splits
        dbox = dist2bbox(self.dfl(box), tf.expand_dims(self.anchors, axis=0), xywh=True, dim=1) * self.strides
        dbox = tf.transpose(dbox, perm=[0, 2, 1])
        y = tf.concat([dbox, tf.sigmoid(cls)], axis=2)
        return y if self.export else (y, x)
        
    def bias_init(self):
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.assign(1.0)  # Initialize box bias to 1.0
            b[-1].bias[:m.nc].assign(tf.math.log(5 / m.nc / (640 / s) ** 2))  # Initialize class bias

class Proto(Layer):
    # YOLOv8 mask Proto module for segmentation models
    def __init__(self, c_=256, c2=32):# number of protos, number of masks
        super(Proto, self).__init__()
        self.cv1 = Conv(c_, k=3, s=1, p='same')
        self.upsample = tf.keras.layers.Conv2DTranspose(filters=c_, kernel_size=2, strides=2, padding='valid')
        self.cv2 = Conv(c_, k=3, s=1, p='same')
        self.cv3 = Conv(c2, k=1, s=1, p='valid')

    def call(self, x, training=None):
        x = self.cv1(x, training=training)
        x = self.upsample(x)
        x = self.cv2(x, training=training)
        x = self.cv3(x, training=training)
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
            Conv(c4, k=3, s=1, p='same'),
            Conv(c4, k=3, s=1, p='same'),
            tf.keras.layers.Conv2D(filters=self.nm, kernel_size=1, strides=1, padding='valid')
        ]) for x in ch]

    def call(self, x, training=None):
        p = self.proto(x[0], training=training)  # mask protos
        bs = p.shape[0]  # batch size
        mc_parts = []
        for i in range(self.nl):
        # Convolution and reshape operations
            conv_result = self.cv4[i](x[i], training=training)
            reshaped_result = tf.reshape(conv_result, (bs, -1, self.nm))
        # Add the processed result to the list
            mc_parts.append(reshaped_result)
            
        mc = tf.concat(mc_parts, axis = 1) # mask coefficients
        x = self.detect(self, x, training=training)
        print("export:", self.export)
        print("training:", training)
        if training:
           return x, mc, p
        return (tf.concat([x, mc], axis=1), p) if self.export else (tf.concat([x[0], mc], axis=2), (x[1], mc, p))
  
class Yolov8Seg(tf.keras.Model):
    #Yolov8 Segmentation Model, input channel must be a multiple of 32
    #Base model structure based on https://arxiv.org/abs/2304.00501
    def __init__(self):
        super(Yolov8Seg, self).__init__()
        # Backbone
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
        self.upsample = keras.layers.UpSampling2D(size=2, interpolation='nearest')
        #self.upsample = lambda x: tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 2), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        self.c2f5 = C2f(c2=512, n=3, shortcut=False) #   - c2=512 x w
        self.c2f6 = C2f(c2=256, n=3, shortcut=False) #p3
        self.cv6 = Conv(c2=256, k=3, s=2, p='same')  #p3
        self.c2f7 = C2f(c2=512, n=3, shortcut=False) #p4 - c2=512 x w
        self.cv7 = Conv(c2=512, k=3, s=2, p='same')  #p4 - c2=512 x w
        self.c2f8 = C2f(c2=512, n=3, shortcut=False) #p5 - c2=512 x w x r
        self.segment = lambda nc, ch: Segment(nc=nc, ch=ch)
        
    def call(self, img, nc, training=None):
        # Backbone
        x1 = self.seq1(img, training=training)
        x2 = self.seq2(x1, training=training)
        x3 = self.seq3(x2, training=training)
        # Head
        x4 = self.c2f5(tf.concat([self.upsample(x3), x2], axis=3), training=training)
        xs1 = self.c2f6(tf.concat([self.upsample(x4), x1], axis=3), training=training)
        xs2 = self.c2f7(tf.concat([self.cv6(xs1, training=training), x4], axis=3), training=training)
        xs3 = self.c2f8(tf.concat([self.cv7(xs2, training=training), x3], axis=3), training=training)
        # Segmentation
        ch = [xs1.shape[-1], xs2.shape[-1], xs3.shape[-1]]
        seginputs = [xs1, xs2, xs3]
        
        seg = self.segment(nc=nc, ch=ch)
        out = seg(seginputs, training=training)
        return out


#Test yolov8 model
model = Yolov8Seg()
batch_size = 1
inputtensor = tf.random.normal((batch_size, 640, 640, 3))

# Training mode
#outputs_train = model(inputtensor, nc=2, training=True)
#print("seg_output:", outputs_train)

# Inference mode 
outputs_inference = model(inputtensor, nc=2, training=False)
print("seg_output:", outputs_inference)



#testing detect/segment head
#input_channels = [256, 512, 512]
#detect_model = Detect(ch=input_channels)
#segment_model = Segment(ch=input_channels)
#batch_size = 1
#tensor1 = tf.random.normal((batch_size, 80, 80, 256))
#tensor2 = tf.random.normal((batch_size, 40, 40, 512))
#tensor3 = tf.random.normal((batch_size, 40, 40, 512))
#input_tensors = [tensor1, tensor2, tensor3]
#dbox_output = detect_model(input_tensors)
#seg_output = segment_model(input_tensors)
#print("seg_output:", seg_output)
