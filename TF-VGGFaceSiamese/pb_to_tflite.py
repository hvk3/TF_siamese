import tensorflow as tf
import pdb
with tf.Session() as sess:
    model_filename = 'siamese.pb'
    with open(model_filename, 'r') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
pdb.set_trace()
tflite = tf.contrib.lite.toco_convert(
    sess.graph_def,
    input_tensors=[
        sess.graph.get_tensor_by_name('import/input_1:0'),
        sess.graph.get_tensor_by_name('import/input_2:0')
    ],
    output_tensors=[
        sess.graph.get_tensor_by_name('import/output_node0:0')
    ]
)
