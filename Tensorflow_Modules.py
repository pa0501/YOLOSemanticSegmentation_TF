import tensorflow as tf
from tensorflow.python import keras
from keras import layers
import numpy as np
from PIL import Image


# TF implementation
class Conv(layers.Layer):
    def __init__(
            self,
            output_channel,
            kernel_size=1,
            strides=1,
            padding="same",
            activation="swish"):
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
            data_format='channels_first',
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
            "batchnormalization": self.bn
        })
        return config

    def call(self, x, ):
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
            "c1": self.c1,
            "conv1": self.cv1,
            "conv2": self.cv2
        })
        return config

    def call(self, x):
        y = tf.split(self.cv1(x),
                     num_or_size_splits=2,
                     axis=1)  # channel axis
        y.extend([m(y[-1]) for m in self.m])
        return self.cv2(tf.concat(y, axis=1))


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
                                  padding='same',
                                  data_format='channels_first')

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
        return self.cv2(tf.concat([x, y1, y2, self.m(y2)], axis=1))


class DFL(layers.Layer):
    """
    Integral module of Distribution Focal Loss (DFL)
    proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        super().__init__()
        self.c1 = c1
        x = tf.range(self.c1, dtype=tf.float32)
        weights_ini = tf.reshape(x, (1, 1, self.c1, 1))
        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=1,
            use_bias=False,
            trainable=False,
            data_format='channels_first',
            weights=[weights_ini])

    def call(self, x):
        b = tf.shape(x)[0]  # batch
        a = tf.shape(x)[2]  # anchors

        x = tf.reshape(x, (b, 4, self.c1, a))
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        conv_output = self.conv(tf.nn.softmax(x, axis=1))
        return tf.reshape(conv_output, (b, 4, a))


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points = []
    stride_tensor = []
    assert feats is not None
    dtype = feats[0].dtype
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
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
    # YOLOv8 Detect head for detection models
    shape = None
    anchors = []
    strides = []

    def __init__(self, nc=80, ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x model)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = (8.0, 16.0, 32.0)  # values only for model l
        #  self.stride = tf.Variable(initial_value=tf.zeros(self.nl))  # strides computed during build

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels

        self.cv2 = [tf.keras.Sequential([
            Conv(output_channel=c2, kernel_size=3),
            Conv(output_channel=c2, kernel_size=3),
            layers.Conv2D(filters=(4 * self.reg_max),
                          kernel_size=1,
                          data_format='channels_first')
        ]) for _ in ch]

        self.cv3 = [tf.keras.Sequential([
            Conv(output_channel=c3, kernel_size=3),
            Conv(output_channel=c3, kernel_size=3),
            layers.Conv2D(filters=self.nc,
                          kernel_size=1,
                          data_format='channels_first')
        ]) for _ in ch]

        if self.reg_max > 1:
            self.dfl = DFL(self.reg_max)
        else:
            self.dfl = layers.Identity()

    def get_config(self):
        config = super().get_config()
        config.update({
            "number of class": self.nc,
            "number of detection layers": self.nl,
            "DFL channels": self.reg_max,
            "DFL function": self.dfl,
            "number of outputs per anchor": self.no,
            "strides computed during build": self.strides,
            "conv2": self.cv2,
            "conv3": self.cv3
        })
        return config

    def call(self, x):
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = tf.concat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)  # 1, 144, h, w

        if self.shape != shape:
            self.anchors, self.strides = (tf.transpose(_, perm=(1, 0)) for _ in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        concatenated_xi = []
        bs = tf.shape(x[0])[0]
        for xi in x:
            xi_reshaped = tf.reshape(xi, (bs, self.no, -1))
            concatenated_xi.append(xi_reshaped)
        concatenated_tensor = tf.concat(concatenated_xi, axis=2)  # 1, 144, 8400

        box, cls = tf.split(concatenated_tensor, (self.reg_max * 4, self.nc), axis=1)  # [1, 64, 8400], [1, 80, 8400]

        dbox = dist2bbox(self.dfl(box), tf.expand_dims(self.anchors, axis=0), xywh=False, dim=1) * self.strides
        y = tf.concat([dbox, tf.sigmoid(cls)], axis=1)
        return y


class Proto(layers.Layer):
    # YOLOv8 mask Proto module for segmentation models
    def __init__(self, c_=256, c2=32):  # number of protos, number of masks
        super(Proto, self).__init__()
        self.c_ = c_
        self.c2 = c2
        self.cv1 = Conv(self.c_, kernel_size=3)
        self.upsample = layers.Conv2DTranspose(filters=self.c_,
                                               kernel_size=2,
                                               strides=2,
                                               padding='valid',
                                               data_format='channels_first',
                                               use_bias=True)
        self.cv2 = Conv(output_channel=self.c_, kernel_size=3)
        self.cv3 = Conv(output_channel=self.c2, kernel_size=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "number of protos": self.c_,
            "number of masks": self.c2,
            "conv1": self.cv1,
            "conv2": self.cv2,
            "conv3": self.cv3,
            "upsample": self.upsample,
        })
        return config

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
        self.proto = Proto(self.npr, self.nm)  # protos
        self.detect = Detect.call

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = [tf.keras.Sequential([
            Conv(output_channel=c4, kernel_size=3),
            Conv(output_channel=c4, kernel_size=3),
            tf.keras.layers.Conv2D(filters=self.nm,
                                   kernel_size=1,
                                   data_format='channels_first')
        ]) for x in ch]

    def get_config(self):
        config = super().get_config()
        config.update({
            "number of protos": self.npr,
            "number of masks": self.nm,
            "detect": self.detect,
            "proto": self.proto,
            "conv4": self.cv4,
        })
        return config

    def call(self, x):
        p = self.proto(x[0])  # mask protos
        bs = tf.shape(p)[0]  # batch size
        mc_parts = []
        for i in range(self.nl):
            conv_result = self.cv4[i](x[i])
            reshaped_result = tf.reshape(conv_result, (bs, self.nm, -1))
            mc_parts.append(reshaped_result)

        mc = tf.concat(mc_parts, axis=2)  # mask coefficients
        x = self.detect(self, x)

        return tf.concat([x, mc], axis=1), p


def intersection(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    return (x2 - x1) * (y2 - y1)


def union(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    return box1_area + box2_area - intersection(box1, box2)


def iou(box1, box2):
    return intersection(box1, box2) / union(box1, box2)


class_label = ["small rock", "big rock"]


class YOLOv8Seg_BaseModel(tf.keras.Model):
    # Yolov8 Segmentation Model, input channel must be a multiple of 32
    # Including Segmentation head with encoding/post-processing methode
    # Base model structure based on https://arxiv.org/abs/2304.00501
    def __init__(self, shape_in, nc=4):
        super(YOLOv8Seg_BaseModel, self).__init__()
        # Backbone
        self.shape_in = shape_in  # NCHW
        self.nc = nc  # number of classes
        self.inputs = keras.Input(shape=self.shape_in)
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
        self.upsample = layers.UpSampling2D(size=2, data_format='channels_first', interpolation='bilinear')
        # self.upsample = lambda x: tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 2), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        self.c2f5 = C2f(output_channel=512, repeat=3, shortcut=False)  # - c2=512 x w
        self.c2f6 = C2f(output_channel=256, repeat=3, shortcut=False)  # p3
        self.cv6 = Conv(output_channel=256, kernel_size=3, strides=2)  # p3
        self.c2f7 = C2f(output_channel=512, repeat=3, shortcut=False)  # p4 - c2=512 x w
        self.cv7 = Conv(output_channel=512, kernel_size=3, strides=2)  # p4 - c2=512 x w
        self.c2f8 = C2f(output_channel=512, repeat=3, shortcut=False)  # p5 - c2=512 x w x r
        self.segment_head = Segment(nc=nc, ch=[256, 512, 512])


def get_mask(img_shape, mask, box):
    img_height = max(img_shape)
    img_width = max(img_shape)
    mask = tf.reshape(mask, (img_width // 4, img_height // 4))
    mask = tf.sigmoid(mask)
    # mask = (mask > 0.5).astype('uint8') * 255  (if mask is numpy)
    mask = tf.cast(mask > 0.5, tf.uint8) * 255  # background = 0 (black), object = 255 (white)

    #  crop the mask
    x1, y1, x2, y2 = box  # float
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    mask_x1 = x1 // 4
    mask_y1 = y1 // 4
    mask_x2 = x2 // 4
    mask_y2 = y2 // 4
    mask = mask[mask_y1:mask_y2, mask_x1:mask_x2]

    # resize the cropped mask
    target_width = tf.abs(x2 - x1)
    target_height = tf.abs(y2 - y1)

    img_mask = Image.fromarray(mask.numpy(), "L")
    img_mask = img_mask.resize((target_width, target_height), Image.BILINEAR)
    mask = np.array(img_mask)
    return mask, (x1, y1, x2, y2)  # int


def post_process(x):
    detect, segment, img_shape = x
    nc = img_shape[3]
    img_height = max(img_shape)
    img_width = max(img_shape)

    detect = detect.transpose(1, 0)

    mi = 4 + nc  # Mask start index
    nm = segment.shape[0]  # Number of masks

    boxes = detect[:, :mi]
    masks = detect[:, mi:]
    segment_reshaped = segment.reshape(nm, -1)
    masks = masks @ segment_reshaped  # parse mask from segment with index from detect head
    boxes = tf.concat((boxes, masks), axis=1)
    # for each box: 0-4:xyxy, 4-(4+nc): class probabilities, (4+nc)- pixels of segmentation mask (as a single row)

    final_masks = []
    # filter each detected object based on their class probability
    objects = []
    for row in boxes:
        prob = tf.reduce_max(row[4:mi])
        if prob < 0.4:
            continue
        class_id = tf.argmax(row[4:mi])  # get index of class with best probability
        x1, y1, x2, y2 = row[:4]
        mask = get_mask(input_shape, row[mi:], (x1, y1, x2, y2))
        objects.append([x1, y1, x2, y2, class_id, prob, mask])

    # filter overlapped boxes with non-maximum suppression
    objects.sort(key=lambda x: x[5], reverse=True)
    filtered_objects = []
    while len(objects) > 0:
        filtered_objects.append(objects[0])
        objects = [box for box in objects if iou(box, objects[0]) < 0.7]

    # convert the object mask with coordinates and class id into a single mask
    final_mask = tf.zeros((img_height, img_width, nc), dtype=np.uint8)
    for obj in filtered_objects:
        mask = obj[6][0]
        x1, y1, x2, y2 = obj[6][1]
        class_id = tf.one_hot(obj[4], nc)
        final_mask[y1:y2, x1:x2, -1] = tf.where(mask > 0, class_id, 0)

    final_masks.append(final_mask)  # stack of masks in all batches
    outputs = tf.stack(final_masks, axis=0)
    return outputs


class MapLayer(layers.Layer):
    def call(self, input, img_shape):
        return tf.map_fn(post_process,
                         input,
                         fn_output_signature=tf.TensorSpec(shape=(img_shape[0], img_shape[1], img_shape[2]),
                                                           dtype=tf.float32))


def Yolov8_Seg(input_shape, nc=4):
    m = YOLOv8Seg_BaseModel(input_shape, nc=nc)
    img_hw = max(input_shape)
    img_shape = (img_hw, img_hw, nc)
    inputs = m.inputs
    inputs = tf.transpose(inputs, perm=[0, 3, 1, 2])  # If input format is channel_last, comment this line
    x1 = m.seq1(inputs)
    x2 = m.seq2(x1)
    x3 = m.seq3(x2)
    # Head
    x4 = m.c2f5(tf.concat([m.upsample(x3), x2], axis=1))
    xs1 = m.c2f6(tf.concat([m.upsample(x4), x1], axis=1))
    xs2 = m.c2f7(tf.concat([m.cv6(xs1), x4], axis=1))
    xs3 = m.c2f8(tf.concat([m.cv7(xs2), x3], axis=1))
    # Segmentation
    # ch = [xs1.shape[1], xs2.shape[1], xs3.shape[1]]
    seg_inputs = [xs1, xs2, xs3]
    # segment = m.segment_head(nc=self.nc, ch=ch)
    seg_outputs = m.segment_head(seg_inputs)

    # post-processing for each image in batch
    outputs = MapLayer()(seg_outputs, img_shape)

    yolo_model = keras.Model(inputs=inputs, outputs=outputs, name='YOLOv8-Seg')
    return yolo_model


if __name__ == "__main__":
    if True:
        input_shape = (800, 800, 1)
        # input_tensor = tf.keras.layers.Input(shape=input_shape)
        nclasses = 4
        model = Yolov8_Seg(input_shape, nc=nclasses)
        model.summary()
        # model.fit()
        # img = tf.convert_to_tensor(np.expand_dim(np.zeros((1024, 1024)), axis=-1)) #-> output: 1024x1024x4
        # Print model summary
        # model.summary()

        # Generate random input tensor
        batch_size = 4
        input_tensor = tf.random.normal((batch_size,) + input_shape)

        # Pass input_tensor through the model
        output = model(input_tensor)
        print(output)
