
import tensorflow as tf

def sobel_edges(in_tensor, arctan_or_norm=0, keep_channel=False):

    keep_channel = bool(int(keep_channel))
    arctan_or_norm = int(arctan_or_norm)

    new_shape = tf.concat([[-1], tf.shape(in_tensor)[2:]], axis=0, name='SobelShape')
    reshaped = tf.reshape(tensor=in_tensor, shape=new_shape, name='SobelReshaped')

    edge_maps = tf.identity(tf.image.sobel_edges(reshaped), name='SobelEdgeMaps')
    reshaped = tf.reshape(tensor=edge_maps, shape=[-1, 2], name='ArcTanReshaped')

    dy, dx = tf.unstack(reshaped, axis=-1)

    if arctan_or_norm == 0:
        gradient_maps = tf.atan2(dy, dx, name='SobelGradientEdgeMaps')
    else:
        gradient_maps = tf.sqrt(tf.square(dy) + tf.square(dx), name='SobelNormEdgeMaps')

    if keep_channel:
        out_tensor = tf.reshape(tensor=gradient_maps, shape=tf.shape(in_tensor), name='Output')
    else:
        out_tensor = tf.reshape(tensor=gradient_maps, shape=tf.shape(in_tensor)[:-1], name='Output')

    return out_tensor