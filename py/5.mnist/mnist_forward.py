import tensorflow as tf

# 28*28个像素点
INPUT_NODE = 784
# 输出10个数
OUTPUT_NODE = 10
# 隐藏层的节点个数
LAYER1_NODE = 500


# 随机生成权重w
def get_weight(shape, regularizer):
    # truncated_normal产生截断正态分布随机数，取值范围为 [ mean - 2 * stddev, mean + 2 * stddev ]
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    # 定义正则化
    if regularizer is not None:
        tf.add_to_collection("losses",  tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


# 生成偏执b
def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def forward(x, regularizer):
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2
    return y
