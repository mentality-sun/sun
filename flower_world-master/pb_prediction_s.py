import tensorflow as tf
import  numpy as np
import PIL.Image as Image
from skimage import io, transform
import model

def recognize(jpg_path):
    img = Image.open('/home/zhang/input_data/tulips/3202130001.jpeg')
    imag = img.resize([60, 60])
    image = np.array(imag)
    x = tf.shape(image)
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        BATCH_SIZE = 1
        N_CLASSES = 4
        image = tf.cast(jpg_path, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 60, 60, 3])
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)
        logit = tf.nn.softmax(logit)
        x = tf.placeholder(tf.float32, shape=[60, 60, 3], name="input")
        with tf.Session() as sess:
            print("Reading checkpoints...")
            with open("/home/zhang/Downloads/model/expert-graph.pb", "rb") as f:
                output_graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(output_graph_def, name="")
                prediction = sess.run(logit, feed_dict={x: jpg_path})
                max_index = np.argmax(prediction)
                if max_index == 0:
                    result = ('这是玫瑰花的可能性为： %.6f' % prediction[:, 0])
                elif max_index == 1:
                    result = ('这是郁金香的可能性为： %.6f' % prediction[:, 1])
                elif max_index == 2:
                    result = ('这是蒲公英的可能性为： %.6f' % prediction[:, 2])
                else:
                    result = ('这是这是向日葵的可能性为： %.6f' % prediction[:, 3])
                # return result
                print(result)
if __name__ == '__main__':
    img = Image.open('/home/zhang/input_data/tulips/3202130001.jpeg')
    # plt.imshow(img)
    # plt.show()
    imag = img.resize([60, 60])

    image = np.array(imag)
    #print(image.shape)
    #print(image)
    x = tf.shape(image)
    #print(x)
    #
    recognize(image)
