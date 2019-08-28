import tensorflow as tf

IMAGE_SIZE = 28
# 输入的通道数 灰度图 所以是1
NUM_CHANNELS = 1
# 第一层卷积核的大小 5 * 5
CONV1_SIZE = 5
# 第一层卷积核的数目
CONV1_KERNEL_NUM = 32
CONV2_SIZE = 5
CONV2_KERNEL_NUM = 64
FC_SIZE = 512
OUTPUT_NODE = 10


def get_weight(shape, regularizer):
    # 去掉偏离过大的正态分布随机数
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer is not None:
        # tf.contrib.layers.l2_regularizer计算l2正则化 算出一个数字
        # tf.add_to_collection 把数字放进一个名叫losses的列表里面
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bios(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


# x输入描述 w卷积核描述也就是有多少个weight
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


'''
6.lenet-5的网络结构
28*28*1(输入图片 28*28*1的灰度图） ---( 5*5*1*32 步长为1 conv1 )--> 28*28*32 ---( 2*2 步长为2 pooling)--> 14*14*32 

---( 5*5*32*64 步长为1 conv2) --> 14*14*64 ---( 2*2 步长为2 pooling )--> 7*7*64 ---( flatten )-->[1, 7*7*64]
'''


def forward(x, train, regularizer):
    # 第一层卷积
    # shape是对卷积核的描述，分别是行列分辨率，通道数，核数
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer)
    # 卷积神经网络偏置是和卷积核绑定的，所以这里的数量等于卷积核的数量
    conv1_b = get_bios([CONV1_KERNEL_NUM])
    # x是输入描述 分别是batch_size, 长，宽，通道数.
    conv1 = conv2d(x, conv1_w)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    pool1 = max_pool_2x2(relu1)

    # 第二层卷积
    # 第二层的通道数等于第一层的kernel数目
    conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer)
    conv2_b = get_bios([CONV2_KERNEL_NUM])
    conv2 = conv2d(pool1, conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = max_pool_2x2(relu2)

    # 把pool2的维度变成一个列表
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    # pool_shape[0]是一个batch中样本的个数
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    fc1_w = get_weight([nodes, FC_SIZE], regularizer)
    fc1_b = get_bios([FC_SIZE])
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    if train:
        fc1 = tf.nn.dropout(fc1, 0.5)

    fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)
    fc2_b = get_bios([OUTPUT_NODE])
    y = tf.matmul(fc1, fc2_w) + fc2_b
    return y
