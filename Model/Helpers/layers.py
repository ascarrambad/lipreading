
import numpy as np
import tensorflow as tf

from ..CustomLayers.GradientReversal import gradient_reversal

################################################################################
################################### LAYERS #####################################
################################################################################

# FC(FullyConnected)128t-64t
def _fullyconnected(in_tensor, num_hidden_units, init_std, activ_func):
    assert len(in_tensor.shape) == 2
    assert type(in_tensor.shape.as_list()[-1]) is int

    in_size = in_tensor.shape.as_list()[-1]

    W = _weight_variable([in_size, num_hidden_units], init_std, name='W')
    b = _weight_variable([num_hidden_units], init_std, name='b')

    out_tensor = tf.matmul(in_tensor, W) + b
    out_tensor = _add_activation_func(out_tensor, activ_func)

    return out_tensor

# CONV(Conv)12r!2-2
# in_tensor = [batch_size, height, width, channels]
def _conv2d(in_tensor, out_channels, init_std, activ_func, kernel, stride, depthwise=False):

    kernel = int(kernel)
    stride = int(stride)
    depthwise = bool(int(depthwise))

    in_channels = in_tensor.shape.as_list()[-1]
    filters = _weight_variable([kernel,kernel,in_channels,out_channels], init_std, name='Filters')

    if depthwise:
        out_tensor = tf.nn.depthwise_conv2d(input=in_tensor,
                                            filter=filters,
                                            strides=[1,stride,stride,1],
                                            padding='SAME',
                                            name='DepthwiseConvolution')
    else:
        out_tensor = tf.nn.conv2d(input=in_tensor,
                                  filter=filters,
                                  strides=[1,stride,stride,1],
                                  padding='SAME',
                                  name='Convolution')

    out_tensor = _add_activation_func(out_tensor, activ_func)

    return out_tensor

################################################################################
############################### SPECIAL LAYERS #################################
################################################################################

# *LSTM!64.64-0
def _lstm(in_tensor, num_hidden_units, out_hidden_state=True, zero_init=False, seq_len_tensor_name='Inputs/SeqLengths'):

    num_hidden_units = [int(x) for x in num_hidden_units.split('.')]
    out_hidden_state = bool(int(out_hidden_state))
    zero_init = bool(int(zero_init))

    cells = [tf.nn.rnn_cell.LSTMCell(num, state_is_tuple=True) for num in num_hidden_units]
    multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

    seq_len = tf.get_default_graph().get_tensor_by_name(seq_len_tensor_name + ':0')

    if zero_init:
        batch_size = in_tensor.shape[0]
        state = multi_cell.zero_state(batch_size, tf.float32)
        outputs_state = tf.nn.dynamic_rnn(multi_cell, in_tensor, sequence_length=seq_len, dtype=tf.float32, initial_state=state)
    else:
        outputs_state = tf.nn.dynamic_rnn(multi_cell, in_tensor, sequence_length=seq_len, dtype=tf.float32)

    out_tensor = outputs_state[1][-1].c if out_hidden_state else outputs_state[0]

    return tf.identity(out_tensor, name='Output')

# *ConvLSTM!2-12
def _convlstm(in_tensor, kernel, num_filters, conv_ndims=2, out_hidden_state=True, zero_init=False, seq_len_tensor_name='Inputs/SeqLengths'):

    kernel = int(kernel)
    num_filters = int(num_filters)
    conv_ndims = int(conv_ndims)
    out_hidden_state = bool(int(out_hidden_state))
    zero_init = bool(int(zero_init))

    in_shape = in_tensor.shape.as_list()[2:] # shape of tensor excluding batch_size as API required AND sequence_length
    cell = tf.contrib.rnn.ConvLSTMCell(conv_ndims=conv_ndims,
                                       input_shape=in_shape,
                                       output_channels=num_filters,
                                       kernel_shape=[kernel]*conv_ndims)

    seq_len = tf.get_default_graph().get_tensor_by_name(seq_len_tensor_name + ':0')

    if zero_init:
        batch_size = in_tensor.shape[0]
        state = cell.zero_state(batch_size, tf.float32)
        outputs_state = tf.nn.dynamic_rnn(cell, in_tensor, sequence_length=seq_len, dtype=tf.float32, initial_state=state)
    else:
        outputs_state = tf.nn.dynamic_rnn(cell, in_tensor, sequence_length=seq_len, dtype=tf.float32)

    out_tensor = outputs_state[1].c if out_hidden_state else outputs_state[0]

    return tf.identity(out_tensor, name='Output')

def _mask_seq(in_tensor, mask_tensor_name='Inputs/SeqLengths'):

    seq_idx = tf.identity(tf.get_default_graph().get_tensor_by_name(mask_tensor_name + ':0') - 1, name='SeqIndices')
    batch_idx = tf.range(tf.shape(seq_idx)[0], dtype=tf.int32, name='BatchIndices')

    idx = tf.stack([batch_idx, seq_idx], axis=1, name='Indices')

    return tf.gather_nd(params=in_tensor, indices=idx, name='Output')

# *ADVSPLIT(Adv)
def _adversarial_split(in_tensor, train_tensor_name='Inputs/Training'):

    at_index = tf.identity(tf.cast(tf.shape(in_tensor)[0]/2, dtype=tf.int32), name='HalfBatchLength')

    train_cond = tf.get_default_graph().get_tensor_by_name(train_tensor_name + ':0')

    all_features = lambda: in_tensor
    source_features = lambda: in_tensor[:at_index,]

    out_tensor = tf.cond(train_cond, source_features, all_features)

    return tf.identity(out_tensor, name='Output')

# *GRADFLIP(GReversal)
def _gradient_reversal(in_tensor, lambda_tensor_name='Inputs/Lambda'):

    lambda_tensor = tf.get_default_graph().get_tensor_by_name(lambda_tensor_name + ':0')

    out_tensor = gradient_reversal(in_tensor, lambda_tensor)

    return tf.identity(out_tensor, name='Output')

# *FLATFEAT(Flat)!3
_orig_shape = None # Shape to be recovered by orig_shape layer
def _flatfeatures(in_tensor, num_to_flat, reversed_=False):

    num_to_flat = int(num_to_flat)
    reversed_ = bool(int(reversed_))

    global _orig_shape

    assert len(in_tensor.shape) - num_to_flat >= 1

    if _orig_shape is None:
        _orig_shape = (reversed_, tf.shape(in_tensor)[:num_to_flat] if reversed_ else tf.shape(in_tensor)[-num_to_flat:])

    current_shape = in_tensor.shape.as_list()
    current_shape = [-1 if x is None else x for x in current_shape]

    if not reversed_:
        flat_feat = np.prod(current_shape[-num_to_flat:])
        new_shape = current_shape[:-num_to_flat] + [flat_feat]
    else:
        flat_feat = -1
        new_shape = [flat_feat] + current_shape[num_to_flat:]

    return tf.reshape(tensor=in_tensor,
                      shape=new_shape,
                      name='Output')

# *ORESHAPE
def _original_reshape(in_tensor):

    global _orig_shape

    assert _orig_shape is not None

    reversed_ = _orig_shape[0]
    flat_feat = _orig_shape[1]

    if not reversed_:
        new_shape = tf.concat([tf.shape(in_tensor)[:-1], flat_feat], axis=0)
    else:
        new_shape = tf.concat([flat_feat, tf.shape(in_tensor)[1:]], axis=0)

    out_tensor = tf.reshape(tensor=in_tensor,
                            shape=new_shape,
                            name='Output')

    _orig_shape = None

    return out_tensor



# *DP(Dropout)!0.5
def _dropout(in_tensor, keep_prob=0.5, train_tensor_name='Inputs/Training'):

    keep_prob = float(keep_prob)

    train_cond = tf.get_default_graph().get_tensor_by_name(train_tensor_name + ':0')

    train_prob = lambda: keep_prob
    test_prob = lambda: 1.0
    keep_prob = tf.cond(train_cond, train_prob, test_prob)

    dropout = tf.nn.dropout(x=in_tensor, keep_prob=keep_prob)

    return tf.identity(dropout, name='Output')

# *MP(MaxPooling)!2-2
def _max_pool(in_tensor, kernel, stride):

    kernel = int(kernel)
    stride = int(stride)

    return tf.nn.max_pool(value=in_tensor,
                          ksize=[1,kernel,kernel, 1],
                          strides=[1,stride,stride,1],
                          padding='SAME',
                          name='Output')

################################################################################
################################### HELPERS ####################################
################################################################################

layer_type = {
    'FC': _fullyconnected,
    'CONV': _conv2d,
    'LSTM': _lstm,
    'CONVLSTM': _convlstm,
    'MASKSEQ': _mask_seq,
    'ADVSPLIT': _adversarial_split,
    'GRADFLIP': _gradient_reversal,
    'FLATFEAT': _flatfeatures,
    'ORESHAPE': _original_reshape,
    'DP': _dropout,
    'MP': _max_pool,
}

def _add_activation_func(in_tensor, func):
    if func == 't':
        out = tf.nn.tanh(in_tensor, name='Output')
    elif func == 'r':
        out = tf.nn.relu(in_tensor, name='Output')
    elif func == 's':
        out = tf.nn.sigmoid(in_tensor, name='Output')
    elif func == 'i':
        out = tf.identity(in_tensor, name='Output')
    else:
        raise Exception('Nonlinearity %s not known' % func)
    return out

def _weight_variable(shape, init_std, name):
    with tf.name_scope(name):
        initial = tf.truncated_normal(shape, stddev=init_std, name='TruncatedNormal')

    return tf.Variable(initial, name=name)
