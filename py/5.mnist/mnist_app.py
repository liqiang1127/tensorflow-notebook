import tensorflow as tf
import numpy as np
from PIL import Image
import mnist_backward
import mnist_forward


def restore_model(test_pic_arr):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y = mnist_forward.forward(x, None)
        pre_value = tf.arg_max(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                pre_value = sess.run(pre_value, feed_dict={x: test_pic_arr})
                return pre_value
            else:
                print("No checkpoint file found")
                return -1


def pre_pic(pic_name):
    img = Image.open(pic_name)
    re_im = img.resize((28, 28), Image.ANTIALIAS)
    im_arr = np.array(re_im.convert('L'))
    threshold = 50
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if im_arr[i][j] < threshold:
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255

    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1.0 / 255.0)

    return img_ready


def application():
    test_num = input("input the num of the test pictures")
    for i in range(test_num):
        test_pic = input("the path of test picture:")
        test_pic_arr = pre_pic(test_pic)
        pre_value = restore_model(test_pic_arr)
        print("the prediction num is:", pre_value)


def main():
    application()


if __name__ == "__main__":
    main()
