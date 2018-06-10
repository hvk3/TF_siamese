from keras_vggface import VGGFace
from keras.layers import Dense, Flatten, Input, merge
from keras.regularizers import l2
from keras.models import Model
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

import argparse
import keras.backend as K
import keras.optimizers as optimizers
import os
import pdb
import tensorflow as tf

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    '--use_CNN',
    default=False,
    action='store_true'
)
parser.add_argument(
    '--use_siamese',
    default=False,
    action='store_true'
)


def allow_dynamic_growth():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess


def get_convnet(input_shape):
    model = VGGFace(
        include_top=False,
        pooling='max',
        input_shape=input_shape
    )
    last_layer = model.get_layer('pool5').output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(
        1024,
        activation='relu',
        name='fc6',
        kernel_regularizer=l2(0.01)
    )(x)
    my_model = Model(model.input, x)

    return my_model


def get_siamese_model(input_shape):
    # Reference : https://sorenbouma.github.io/blog/oneshot/
    # https://github.com/NVIDIA/keras/blob/master/examples/mnist_siamese_graph.py
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    convnet = get_convnet(input_shape)
    for i in xrange(len(convnet.layers[:-1])):
        convnet.layers[i].trainable = False
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)

    both = merge(
        [encoded_l, encoded_r],
        # mode=lambda x: K.abs(x[0] - x[1]),
        mode=lambda x: (x[0] - x[1]) ** 2 / (x[0] + x[1] + K.epsilon()),
        output_shape=lambda x: x[0]
    )
    # both = Lambda(
    #     euclidean_distance,
    #     output_shape=eucl_dist_output_shape
    # )([encoded_l, encoded_r])
    prediction = Dense(1, activation='sigmoid')(both)
    siamese_net = Model(
        inputs=[left_input, right_input],
        outputs=prediction
        # outputs=both
    )
    siamese_net.compile(
        loss="binary_crossentropy",
        # loss=contrastive_loss,
        metrics=['acc', 'mse'],
        optimizer=optimizers.Adam(lr=5e-4)
    )
    return siamese_net


if __name__ == '__main__':
    args = parser.parse_args()
    K.set_image_data_format('channels_last')
    K.set_session(allow_dynamic_growth())
    if (args.use_CNN):
        model = VGGFace(
            include_top=False,
            input_shape=(224, 224, 3)
        )
        save_to = 'VGGFace.pb'
    else:
        model = get_siamese_model((224, 224, 3))
        if (os.path.exists('siamese_net.h5')):
            model.load_weights('siamese_net.h5')
            save_to = 'siamese.pb'
    sess = K.get_session()
    num_output = 1
    pred = [None] * num_output
    pred_node_names = [None] * num_output
    for i in range(num_output):
        pred_node_names[i] = 'output_node' + str(i)
        pred[i] = tf.identity(model.outputs[i], name=pred_node_names[i])
    print('output nodes names are: ', pred_node_names)

    constant_graph = graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(),
        pred_node_names
    )
    graph_io.write_graph(
        constant_graph,
        '.',
        save_to,
        as_text=False
    )
