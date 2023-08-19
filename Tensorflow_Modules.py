import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.nn as nn
from keras.layers import Layer

#TF implementation
class Conv(Layer):
    # Standard convolution with args(ch_out, kernel, stride, padding, group, dilation, activation). Default data format is channel last

    default_act = tf.keras.layers.Activation(tf.keras.activations.swish)  # default activation
    
    def __init__(self, c2, k=1, s=1, p='valid', g=1, d=1, act=True):
        super().__init__()

        
        self.conv = keras.layers.Conv2D(filters=c2, kernel_size=k, strides=s, padding=p, groups=g, dilation_rate=d, use_bias=False)
        self.bn = keras.layers.BatchNormalization()
        
        if act is True:
            self.act = self.default_act
        elif isinstance(act, keras.layers.Layer):
            self.act = act
        else:
            self.act = keras.layers.Activation(act) if act is not None else keras.layers.Identity()

    def call(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

    
class Bottleneck(Layer):
    # Standard bottleneck
    def __init__(self, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c_, k[0],s = 1)
        self.cv2 = Conv(c2, k[1],s = 1, g=g)
        self.add = shortcut
        
    def call(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        return x + x2 if self.add else x2
    
class C2f(Layer):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c2, n=1, shortcut=False, g=1, e=0.5): #ch_out, number, shortcut, groups, expansion
        super(C2f, self).__init__()
        self.c = int(c2 * e) # hidden channels
        self.cv1 = Conv(2 * self.c, k=1, s=1, p='valid')
        self.cv2 = Conv(c2, k=1, s=1, p='valid')
        self.m = [Bottleneck(self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)]

    def call(self, x):
        y = tf.split(self.cv1(x), num_or_size_splits=2, axis=-1) #split at channel axis
        y.extend([m(y[-1]) for m in self.m])
        return self.cv2(tf.concat(y, axis=-1))

class SPPF(Layer):
     # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super(SPPF, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c_, k=1,s=1,p='valid')
        self.cv2 = Conv(c2, k=1,s=1,p='valid')
        self.m = tf.keras.layers.MaxPool2D(pool_size=k, strides=1, padding='same')

    def call(self, x):
        self.c1 = x.shape[-1] #the input channel
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(tf.concat([x, y1, y2, self.m(y2)], axis=3))

class DFL(Layer):
    def __init__(self, c1=16):
        super().__init__()
        x = tf.range(c1, dtype=tf.float32)
        weights_ini = tf.reshape(x, (1, 1, 1, c1)) #weights initialization
        self.conv = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, use_bias=False, trainable=False, weights = [weights_ini])
        self.c1 = c1

    def call(self, x):
        b, a, c = x.shape   # batch, anchors, channels
        x_reshaped = tf.transpose(tf.reshape(x, (b, 4, a, self.c1)), perm=(0, 2, 1, 3))
        softmax_x = tf.nn.softmax(x_reshaped, axis=1)
        conv_output = self.conv(softmax_x)
        return tf.reshape(conv_output, (b, 4, a))

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
    lt, rb = torch.split(distance, 2, dim)
    lt, rb = tf.split(distance, (self.reg_max * 4, self.nc), axis=2)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = tf.split(distance, 2, axis=dim)
    lt, rb = tf.split(distance, (self.reg_max * 4, self.nc), axis=2)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return tf.concat((c_xy, wh), axis=dim)  # xywh bbox
    return tf.concat((x1y1, x2y2), axis=dim)  # xyxy bbox

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
        print("self.nl:", self.nl)
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = tf.Variable(initial_value=tf.zeros(self.nl), trainable=False)  # strides computed during build

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
        print("shape_x0:", shape)
        #print("shape_cv2:", self.cv2[0](x[0]))
        for i in range(self.nl):
            x[i] = tf.concat((self.cv2[i](x[i]), self.cv3[i](x[i])), 3) #1, h, w, 144
        if training:
            return x
        elif self.dynamic or self.shape != shape:
            print("test:")
            self.anchors, self.strides = (tf.transpose(x, perm=(0, 1)) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
            
        concatenated_xi = []
        for xi in x:
            xi_reshaped = tf.reshape(xi, (shape[0], -1, self.no))
            concatenated_xi.append(xi_reshaped)
        concatenated_tensor = tf.concat(concatenated_xi, axis=1) #1, 9600, 144 /channel last
        #print("concatenated_tensor+FT:", concatenated_tensor)
        
        splits = tf.split(concatenated_tensor, (self.reg_max * 4, self.nc), axis=2) #[1, 9600, 64], [1, 9600, 80]
        #print("Splits shapes:", [split.shape for split in splits])
        box, cls = splits
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        
        return concatenated_tensor
input_channels = [256, 512, 512]

# Create an instance of the Detect class
detect_model = Detect(ch=input_channels)

# Prepare an input tensor
batch_size = 1

tensor1 = tf.random.normal((batch_size, 80, 80, 256))
tensor2 = tf.random.normal((batch_size, 40, 40, 512))
tensor3 = tf.random.normal((batch_size, 40, 40, 512))

# Forward pass through the Detect model
input_tensors = [tensor1, tensor2, tensor3]
dbox_output = detect_model(input_tensors, training=False)

# Print the shape of the dbox_output tensor
#print("x1:", dbox_output, "shape1",dbox_output.shape)
#print("sy:", dbox_output[1])
