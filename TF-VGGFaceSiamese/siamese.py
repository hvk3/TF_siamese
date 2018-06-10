from tensorflow.python.tools import optimize_for_inference_lib

import argparse
import cv2
import numpy as np
import os
import pdb
import tensorflow as tf

parser = argparse.ArgumentParser(
    description='Utility for training and saving a Siamese network in TF',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    '--test_CNN',
    default=False,
    action='store_true'
)
parser.add_argument(
    '--test_siamese',
    default=False,
    action='store_true'
)
parser.add_argument(
    '--train_siamese',
    default=False,
    action='store_true'
)


def allow_dynamic_growth():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config


def load_samples(which):
    import cv2
    name_to_label = np.load('name_to_label.npy').item()
    images, labels = [[] for _ in xrange(2)]
    for dir_ in sorted(os.listdir(
        os.path.join(which)
    )):
        label = name_to_label[dir_]
        for img_path in os.listdir(os.path.join(which, dir_)):
            img_path = os.path.join(
                which,
                dir_,
                img_path
            )
            img = cv2.imread(img_path).astype('float64')
            img /= 255.
            images.append(img)
            labels.append(label)
    return images, labels


def get_pairs(samples, labels, nb_samples, batch_size=16):
    pairs, pair_labels = [[] for _ in xrange(2)]
    unique_labels = np.unique(labels)
    for unique_label in unique_labels:
        relevant_indices = []
        irrelevant_indices = []
        for i in xrange(len(labels)):
            if (labels[i] == unique_label):
                relevant_indices.append(i)
            else:
                irrelevant_indices.append(i)
        np.random.shuffle(relevant_indices)
        for i in xrange(nb_samples):
            idx_relevant_1 = np.random.choice(
                relevant_indices,
                replace=False
            )
            idx_relevant_2 = np.random.choice(
                relevant_indices,
                replace=False
            )
            # print idx_relevant_1, idx_relevant_2
            pairs.append(
                (samples[idx_relevant_1], samples[idx_relevant_2])
            )
            pair_labels.append(1)
        for i in xrange(nb_samples):
            idx_relevant_1 = np.random.choice(
                relevant_indices,
                replace=False
            )
            idx_irrelevant = np.random.choice(
                irrelevant_indices,
                replace=False
            )
            pairs.append(
                (samples[idx_relevant_1], samples[idx_irrelevant])
            )
            pair_labels.append(0)
    rng = np.random.get_state()
    np.random.shuffle(pairs)
    np.random.set_state(rng)
    np.random.shuffle(pair_labels)
    return pairs, pair_labels


def generator(pairs, pair_labels, batch_size):
    i = 0
    while True:
        batch_pairs = np.array(pairs[i * batch_size:(i + 1) * batch_size])
        batch_labels = pair_labels[
            i * batch_size:(i + 1) * batch_size
        ]
        batch_pair_labels = []
        for batch_label in batch_labels:
            if (batch_label == 0):
                batch_pair_labels.append([1, 0])
            else:
                batch_pair_labels.append([0, 1])
        i += 1
        i %= len(pairs) // batch_size
        yield [batch_pairs[:, 0], batch_pairs[:, 1]], batch_pair_labels


if __name__ == '__main__':
    args = parser.parse_args()
    graph_def = tf.GraphDef()
    if (args.test_siamese):
        load_from = 'siamese.pb'
    else:
        load_from = 'VGGFace.pb'
    with open(load_from, 'r') as f:
        graph_def.ParseFromString(f.read())
    frozen_graph_def = optimize_for_inference_lib.optimize_for_inference(
        graph_def,
        ["input_1"],
        ["output_node0"],
        tf.float32.as_datatype_enum
    )
    tf.import_graph_def(frozen_graph_def)
    config = allow_dynamic_growth()
    sess = tf.Session(config=config)
    frozen_graph = sess.graph
    if (args.test_siamese):
        pdb.set_trace()
        img_1 = np.array([cv2.imread('0.jpg')]).astype('float32')
        img_2 = np.array([cv2.imread('1.jpg')]).astype('float32')
        img_1 /= 255.
        img_2 /= 255.
        match_score = sess.run(
            frozen_graph.get_tensor_by_name('import/output_node0:0'),
            feed_dict={
                frozen_graph.get_tensor_by_name('import/input_1:0'): img_1,
                frozen_graph.get_tensor_by_name('import/input_2:0'): img_2
            }
        )
        print match_score
    if (args.test_CNN):
        img = np.array([cv2.imread('0.jpg')])
        img_vector = sess.run(
            frozen_graph.get_tensor_by_name('import/output_node0:0'),
            feed_dict={
                frozen_graph.get_tensor_by_name('import/input_1:0'): img
            }
        )
        print img_vector.shape
    if (args.train_siamese):
        img = np.array([cv2.imread('0.jpg')])
        img_vector = sess.run(
            frozen_graph.get_tensor_by_name('import/output_node0:0'),
            feed_dict={
                frozen_graph.get_tensor_by_name('import/input_1:0'): img
            }
        )
        img_vector = img_vector.flatten()
        flat_left = tf.placeholder("float32", [None, img_vector.shape[0]])
        flat_right = tf.placeholder("float32", [None, img_vector.shape[0]])
        left = tf.layers.dense(
            flat_left,
            units=1024,
            name='left',
            activation=tf.nn.relu
        )
        right = tf.layers.dense(
            flat_right,
            units=1024,
            name='right',
            activation=tf.nn.relu
        )
        merge = tf.abs(tf.subtract(left, right))
        prediction = tf.layers.dense(
            merge,
            units=2,
            name='similarity',
            activation=tf.nn.sigmoid
        )
        y = tf.placeholder("float32", [None, 2])
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=prediction,
                labels=y
            )
        )
        learning_rate = 1e-3
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate
        ).minimize(cost)

        init = tf.global_variables_initializer()

        # Train and save
        batch_size = 32
        train_samples, train_labels = load_samples('train')
        test_samples, test_labels = load_samples('test')
        validation_samples, validation_labels = load_samples('validation')

        train_pairs, train_pair_labels = get_pairs(
            train_samples,
            train_labels,
            125,
            batch_size
        )
        test_pairs, test_pair_labels = get_pairs(
            test_samples,
            test_labels,
            75,
            batch_size
        )
        validation_pairs, validation_pair_labels = get_pairs(
            validation_samples,
            validation_labels,
            50, batch_size
        )

        train_generator = generator(
            train_pairs,
            train_pair_labels,
            batch_size
        )
        validation_generator = generator(
            validation_pairs,
            validation_pair_labels,
            batch_size
        )
        test_generator = generator(
            test_pairs,
            test_pair_labels,
            batch_size
        )
        sess.run(init)
        prev_avg_cost = None
        import tqdm
        for epoch in xrange(100):
            avg_cost = 0.
            for j in tqdm.tqdm(range(len(train_pairs) // batch_size)):
                (batch_x_left, batch_x_right), batch_y = train_generator.next()
                batch_x_left_features = sess.run(
                    frozen_graph.get_tensor_by_name('import/output_node0:0'),
                    feed_dict={
                        frozen_graph.get_tensor_by_name(
                            'import/input_1:0'
                        ): batch_x_left
                    }
                )
                batch_x_right_features = sess.run(
                    frozen_graph.get_tensor_by_name('import/output_node0:0'),
                    feed_dict={
                        frozen_graph.get_tensor_by_name(
                            'import/input_1:0'
                        ): batch_x_right
                    }
                )
                batch_x_left_features = map(
                    lambda x: x.flatten(),
                    batch_x_left_features
                )
                batch_x_right_features = map(
                    lambda x: x.flatten(),
                    batch_x_right_features
                )
                # batch_y = np.expand_dims(batch_y, 1)
                _, c = sess.run(
                    [optimizer, cost],
                    feed_dict={
                        flat_left: batch_x_left_features,
                        flat_right: batch_x_right_features,
                        y: batch_y
                    }
                )
                avg_cost += c / batch_size
            if epoch % 1 == 0:
                print(
                    "Epoch:",
                    '%04d' % (epoch + 1),
                    "cost=",
                    "{:.9f}".format(avg_cost)
                )
            if (prev_avg_cost is None or prev_avg_cost > avg_cost):
                prev_avg_cost = avg_cost
                saver = tf.train.Saver()
                saver.save(sess, './siamese.ckpt')
