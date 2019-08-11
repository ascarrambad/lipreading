
import tensorftensorflow as tf

def diff_frames(in_tensor):
    prev = tf.identity(in_tensor[:,:-1], name='PreviousFrames')
    next_ = tf.identity(in_tensor[:, 1:], name='NextFrames')
    out_tensor = tf.identity(next_ - prev, name='Output')

    return out_tensor