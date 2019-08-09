
import tensorflow as tf

def _reshape(in_tensor):
    shape = tf.shape(tf.get_default_graph().get_tensor_by_name('Inputs/MotFrames:0'), name='OrigShape')
    new_shape = tf.concat([shape[:2], tf.shape(in_tensor)[1:]], axis=0, name='NewShape')

    out_tensor = tf.reshape(tensor=in_tensor,
                            shape=new_shape,
                            name='Output')

    return out_tensor