
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
    assert num_hidden_units > 0
    assert init_std >= 0

    ############################################################################

    in_size = in_tensor.shape.as_list()[-1]

    W = _weight_variable([in_size, num_hidden_units], init_std, name='W')
    b = _weight_variable([num_hidden_units], init_std, name='b')

    ############################################################################

    out_tensor = tf.matmul(in_tensor, W) + b
    out_tensor = _add_activation_func(out_tensor, activ_func)

    return out_tensor

# CONV(Conv)12r!2-2
# in_tensor = [batch_size, height, width, channels]
def _conv2d(in_tensor, out_channels, init_std, activ_func, kernel, stride=1, depthwise=False):

    assert len(in_tensor.shape) == 4
    assert out_channels > 0
    assert init_std >= 0

    kernel = int(kernel)
    stride = int(stride)
    depthwise = bool(int(depthwise))

    assert kernel > 0
    assert stride > 0

    ############################################################################

    in_channels = in_tensor.shape.as_list()[-1]

    filters = _weight_variable([kernel,kernel,in_channels,out_channels], init_std, name='Filters')

    ############################################################################

    if depthwise:
        conv = tf.nn.depthwise_conv2d(input=in_tensor,
                                      filter=filters,
                                      strides=[1,stride,stride,1],
                                      padding='SAME',
                                      name='DepthwiseConvolution')
    else:
        conv = tf.nn.conv2d(input=in_tensor,
                            filter=filters,
                            strides=[1,stride,stride,1],
                            padding='SAME',
                            name='Convolution')

    biases = _weight_variable([out_channels], init_std, name='Biases')
    out_tensor = tf.nn.bias_add(conv, biases, data_format='NHWC')

    out_tensor = _add_activation_func(out_tensor, activ_func)

    return out_tensor

# CONVTD(Conv)12r!2-2
# in_tensor = [batch_size, height, width, channels]
def _conv3d(in_tensor, out_channels, init_std, activ_func, kernel, kdepth, stride=1, sdepth=1):

    assert len(in_tensor.shape) == 5
    assert out_channels > 0
    assert init_std >= 0

    kernel = int(kernel)
    kdepth = int(kdepth)
    stride = int(stride)
    sdepth = int(sdepth)

    assert kernel > 0
    assert kdepth > 0
    assert stride > 0
    assert sdepth > 0

    ############################################################################

    in_channels = in_tensor.shape.as_list()[-1]

    filters = _weight_variable([kdepth,kernel,kernel,in_channels,out_channels], init_std, name='Filters')

    ############################################################################

    conv = tf.nn.conv3d(input=in_tensor,
                        filter=filters,
                        strides=[1,sdepth,stride,stride,1],
                        padding='SAME',
                        name='Convolution')

    biases = _weight_variable([out_channels], init_std, name='Biases')
    out_tensor = tf.nn.bias_add(conv, biases, data_format='NHWC')

    out_tensor = _add_activation_func(out_tensor, activ_func)

    return out_tensor

def _deconv2d(in_tensor, out_channels, init_std, activ_func, kernel, stride=1):

    assert len(in_tensor.shape) == 4
    assert out_channels > 0
    assert init_std >= 0

    kernel = int(kernel)
    stride = int(stride)

    assert kernel > 0
    assert stride > 0

    ############################################################################

    in_shape = in_tensor.shape.as_list()
    out_shape = tf.concat([[tf.shape(in_tensor)[0]], in_shape[1:-1]+[out_channels]], axis=0, name='OutShape')
    in_channels = in_shape[-1]

    filters = _weight_variable([kernel,kernel,out_channels,in_channels], init_std, name='Filters')

    ############################################################################

    deconv = tf.nn.conv2d_transpose(value=in_tensor,
                                    filter=filters,
                                    output_shape=out_shape,
                                    strides=[1,stride,stride,1],
                                    name='Deconvolution')

    biases = _weight_variable([out_channels], init_std, name='Biases')
    out_tensor = tf.nn.bias_add(deconv, biases, data_format='NHWC')

    out_tensor = _add_activation_func(out_tensor, activ_func)

    return out_tensor

################################################################################
############################### SPECIAL LAYERS #################################
################################################################################

# *PREDICT!mse
def _predict(in_tensor, error, trg_tensor):

    if error == 'sce':
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=trg_tensor,
                                                          logits=in_tensor,
                                                          name='SoftmaxCrossEntropy')
        hits = tf.equal(tf.argmax(in_tensor, axis=1), tf.argmax(trg_tensor, axis=1), name='Hits')
    elif error == 'mse':
        loss = tf.square(in_tensor - trg_tensor, name='MeanSquaredError')
        hits = tf.equal(in_tensor, trg_tensor, name='Hits')
    else:
        raise Exception('Error %s not known' % error)

    loss = tf.reduce_mean(loss, name='MeanLoss')

    accuracy = tf.reduce_mean(tf.cast(hits, tf.float32), name='Accuracy')

    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Accuracy', accuracy)

    return loss, hits, accuracy

# *LSTM!64.64-0
_lstm_arr = []
def _lstm(in_tensor, num_hidden_units, out_hidden_state=False, init_type=0, lstm_state_idx=0, extra_params):

    assert len(in_tensor.shape) >= 2

    num_hidden_units = [int(x) for x in num_hidden_units.split('.')]
    out_hidden_state = bool(int(out_hidden_state))
    init_type = int(init_type)
    lstm_state_idx = int(lstm_state_idx)

    assert [x > 0 for x in num_hidden_units]
    assert init_type in range(3)
    assert lstm_state_idx >= 0

    global _lstm_arr

    ############################################################################

    cells = [tf.nn.rnn_cell.LSTMCell(num, name='LSTMCell-%d'%num) for num in num_hidden_units]
    multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    seq_len = extra_params['SequenceLengthsTensor']

    ############################################################################

    if init_type == 0: # default
        outputs_state = tf.nn.dynamic_rnn(multi_cell, in_tensor, sequence_length=seq_len, dtype=tf.float32)
    elif init_type == 1: # previous lstm state init
        state = _lstm_arr[lstm_state_idx][1]
        outputs_state = tf.nn.dynamic_rnn(multi_cell, in_tensor, sequence_length=seq_len, initial_state=state, dtype=tf.float32)
    elif init_type == 2: # zero init
        batch_size = in_tensor.shape[0]
        state = multi_cell.zero_state(batch_size, tf.float32)
        outputs_state = tf.nn.dynamic_rnn(multi_cell, in_tensor, sequence_length=seq_len, initial_state=state, dtype=tf.float32)
    # elif init_type == 3: # in_tensor as c state init
    #     state = _lstm_arr[lstm_state_idx]
    #     state = tf.nn.rnn_cell.LSTMStateTuple(in_tensor[1])
    #     outputs_state = tf.nn.dynamic_rnn(multi_cell, in_tensor, sequence_length=seq_len, initial_state=state, dtype=tf.float32)

    _lstm_arr.append(outputs_state)

    out_tensor = outputs_state[1][0].c if out_hidden_state else outputs_state[0]

    return tf.identity(out_tensor, name='Output')

# *ConvLSTM!2-12
def _convlstm(in_tensor, out_channels, kernel, out_hidden_state=False, conv_ndims=2, zero_init=False, extra_params):

    assert len(in_tensor.shape) >= 5

    out_channels = int(out_channels)
    kernel = int(kernel)
    out_hidden_state = bool(int(out_hidden_state))
    conv_ndims = int(conv_ndims)
    zero_init = bool(int(zero_init))

    assert out_channels > 0
    assert kernel >= 2
    assert conv_ndims >= 2

    ############################################################################

    in_shape = in_tensor.shape.as_list()[2:] # shape of tensor excluding batch_size as API required AND sequence_length

    ############################################################################

    cell = tf.contrib.rnn.ConvLSTMCell(conv_ndims=conv_ndims,
                                       input_shape=in_shape,
                                       output_channels=out_channels,
                                       kernel_shape=[kernel]*conv_ndims)

    seq_len = extra_params['SequenceLengthsTensor']

    ############################################################################

    if zero_init:
        batch_size = in_tensor.shape[0]
        state = cell.zero_state(batch_size, tf.float32)
        outputs_state = tf.nn.dynamic_rnn(cell, in_tensor, sequence_length=seq_len, dtype=tf.float32, initial_state=state)
    else:
        outputs_state = tf.nn.dynamic_rnn(cell, in_tensor, sequence_length=seq_len, dtype=tf.float32)

    out_tensor = outputs_state[1].c if out_hidden_state else outputs_state[0]

    return tf.identity(out_tensor, name='Output')

# *MASKSEQ
def _mask_seq(in_tensor, extra_params):

    assert len(in_tensor.shape) >= 2
    assert extra_params['MaskIndicesTensor'] != None

    ############################################################################

    seq_idx = extra_params['MaskIndicesTensor']

    ############################################################################

    batch_idx = tf.range(tf.shape(seq_idx)[0], dtype=tf.int32, name='BatchIndices')

    idx = tf.stack([batch_idx, seq_idx], axis=1, name='Indices')

    return tf.gather_nd(params=in_tensor, indices=idx, name='Output')

# *GRADFLIP
def _gradient_reversal(in_tensor, extra_params):

    assert extra_params['LambdaTensor'] != None

    ############################################################################

    lambda_tensor = extra_params['LambdaTensor']

    ############################################################################

    out_tensor = gradient_reversal(in_tensor, lambda_tensor)

    return tf.identity(out_tensor, name='Output')

# *FLATFEAT!3
_orig_shape = [] # Shape to be recovered by orig_shape layer
def _flat_features(in_tensor, num_to_flat, reversed_=False):

    assert len(in_tensor.shape) - num_to_flat >= 1

    num_to_flat = int(num_to_flat)
    reversed_ = bool(int(reversed_))

    global _orig_shape

    ############################################################################

    current_shape = in_tensor.shape.as_list()
    current_shape = [-1 if x is None else x for x in current_shape]

    ############################################################################

    _orig_shape.append((reversed_, tf.identity(input=tf.shape(in_tensor)[:num_to_flat] if reversed_ else tf.shape(in_tensor)[-num_to_flat:],
                                               name='RemainingShape')))

    if not reversed_:
        flat_feat = np.prod(current_shape[-num_to_flat:])
        new_shape = current_shape[:-num_to_flat] + [flat_feat]
    else:
        flat_feat = -1
        new_shape = [flat_feat] + current_shape[num_to_flat:]

    return tf.reshape(tensor=in_tensor,
                      shape=new_shape,
                      name='Output')

# *UNDOFLAT
def _undo_flat_features(in_tensor, index=0):

    index = int(index)

    assert index >= 0

    global _orig_shape

    assert len(_orig_shape) > 0

    ############################################################################

    reversed_ = _orig_shape[index][0]
    flat_feat = _orig_shape[index][1]

    ############################################################################

    if not reversed_:
        new_shape = tf.concat([tf.shape(in_tensor)[:-1], flat_feat], axis=0, name='OrigShape')
    else:
        new_shape = tf.concat([flat_feat, tf.shape(in_tensor)[1:]], axis=0, name='OrigShape')

    out_tensor = tf.reshape(tensor=in_tensor,
                            shape=new_shape,
                            name='Output')

    return out_tensor

# *DP!0.5
def _dropout(in_tensor, keep_prob=0.5, extra_params):

    keep_prob = float(keep_prob)

    assert keep_prob > 0 and keep_prob <= 1
    assert extra_params['TrainingStatusTensor'] != None

    ############################################################################

    train_cond = extra_params['TrainingStatusTensor']

    train_prob = lambda: keep_prob
    test_prob = lambda: 1.0
    keep_prob = tf.cond(train_cond, train_prob, test_prob)

    ############################################################################

    dropout = tf.nn.dropout(x=in_tensor, keep_prob=keep_prob)

    return tf.identity(dropout, name='Output')

# *MP!2-2
def _max_pool2d(in_tensor, kernel, stride):

    kernel = int(kernel)
    stride = int(stride)

    assert kernel > 0
    assert stride > 0

    ############################################################################

    return tf.nn.max_pool(value=in_tensor,
                          ksize=[1,kernel,kernel, 1],
                          strides=[1,stride,stride,1],
                          padding='SAME',
                          name='Output')

# *MPTD!2-2
def _max_pool3d(in_tensor, kernel, kdepth, stride=1, sdepth=1)):

    kernel = int(kernel)
    kdepth = int(kdepth)
    stride = int(stride)
    sdepth = int(sdepth)

    assert kernel > 0
    assert kdepth > 0
    assert stride > 0
    assert sdepth > 0

    ############################################################################

    return tf.nn.max_pool3d(input=in_tensor,
                            ksize=[1,kdepth,kernel,kernel, 1],
                            strides=[1,sdepth,stride,stride,1],
                            padding='SAME',
                            name='Output')

# *UNP!2
def _unpooling(in_tensor, multiplier):

    multiplier = int(multiplier)

    assert multiplier > 0

    ############################################################################

    in_shape = in_tensor.shape.as_list()
    out_size = [x*multiplier for x in  in_shape[1:3]]
    out_size = tf.identity(out_size, name='OutSize')

    ############################################################################

    return tf.image.resize_bilinear(images=in_tensor,
                                    size=out_size,
                                    name='Output')

# *CONCAT!1
def _concatenate(in_tensors, axis):

    axis = int(axis)

    assert axis >= 0

    ############################################################################

    return tf.concat(in_tensors, axis=axis, name='Output')

# *SPLIT!1
def _split(in_tensor, axis, splits_num=2):

    axis = int(axis)
    splits_num = int(splits_num)

    assert axis >= 0
    assert splits_num >= 2

    ############################################################################

    splits = tf.split(value=in_tensors,
                      num_or_size_splits=splits_num,
                      axis=axis)

    return [tf.identity(x, name='Output-%d'%i) for i,x in enumerate(splits)]

# *STOPGRAD
def _stop_gradient(in_tensor):
    return tf.stop_gradient(in_tensor,
                            name='Output')

# *CUSTOM
def _custom(in_tensor, *args):
    assert args[-1]['CustomFunction'] != None

    out_tensor = args[-1]['CustomFunction'](in_tensor, *args[:-1])

    return tf.identity(out_tensor, name='Output')

# SOBEL
def _sobel_edges(in_tensor, arctan_or_norm=0, keep_channel=False):

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

# *DIFF
def _diff_frames(in_tensor):
    prev = tf.identity(in_tensor[:,:-1], name='PreviousFrames')
    next_ = tf.identity(in_tensor[:, 1:], name='NextFrames')
    out_tensor = tf.identity(next_ - prev, name='Output')

    return out_tensor

# *SCALE
def _scaler(in_tensor):
    min_ = tf.reduce_min(in_tensor)
    return tf.div(x=tf.subtract(in_tensor, min_),
                  y=tf.subtract(tf.reduce_max(in_tensor), min_),
                  name='Output')

# *RESGEN
_resids = []
def _residual_gen(in_tensors, axis):

    axis = int(axis)

    n_layers = len(in_tensors) // 2
    input_dyn = in_tensors[:n_layers]
    input_cont = in_tensors[n_layers:]

    for l in range(n_layers):
        with tf.variable_scope('ORESHAPE%d'%l):
            input_dyn[l] = _original_reshape(input_dyn[l], index=0)
        with tf.variable_scope('MASKSEQ%d'%l):
            input_dyn[l] = _mask_seq(input_dyn[l])

    global _resids

    for l in range(n_layers):
        input_ = tf.concat([input_dyn[l], input_cont[l]], axis=axis)
        out_dim = input_cont[l].shape.as_list()[-1]

        with tf.variable_scope('CONV%d_1'%l):
            res1 = _conv2d(input_, out_dim, 0.1, 'r', 3)
        with tf.variable_scope('CONV%d_2'%l):
            res2 = _conv2d(res1, out_dim, 0.1, 'i', 3)
        _resids.append(res2)

    return _resids

# *RESGET
def _residual_get(in_tensor, index):

    index = int(index)

    global _resids

    resid = _resids[index]
    out_tensor = tf.add(in_tensor, resid, name='Output')

    return out_tensor

################################################################################
################################### HELPERS ####################################
################################################################################

layer_type = {
    'FC': _fullyconnected,
    'CONV': _conv2d,
    'CONVTD': _conv3d,
    'DECONV': _deconv2d,
    ## SPECIAL LAYERS ##
    'PREDICT': _predict,
    'LSTM': _lstm,
    'CONVLSTM': _convlstm,
    'MASKSEQ': _mask_seq,
    'GRADFLIP': _gradient_reversal,
    'FLATFEAT': _flat_features,
    'UNDOFLAT': _undo_flat_features,
    'DP': _dropout,
    'MP': _max_pool2d,
    'MPTD': _max_pool3d,
    'UNP': _unpooling,
    'CONCAT': _concatenate,
    'SPLIT': _split,
    'STOPGRAD': _stop_gradient,
    'CUSTOM': _custom,
    'SOBEL': _sobel_edges,
    'DIFF': _diff_frames,
    'SCALE': _scaler,
    'RESGEN': _residual_gen,
    'RESGET': _residual_get,
}

def _add_activation_func(in_tensor, func):
    if func == 't':
        out = tf.nn.tanh(in_tensor, name='Output')
    elif func == 'r':
        out = tf.nn.relu(in_tensor, name='Output')
    elif func == 'lr':
        out = tf.nn.leaky_relu(in_tensor, name='Output')
    elif func == 's':
        out = tf.nn.sigmoid(in_tensor, name='Output')
    elif func == 'i':
        out = tf.identity(in_tensor, name='Output')
    else:
        raise Exception('Nonlinearity %s not known' % func)
    return out

def _weight_variable(shape, init_std, name, validate_shape=True):
    initial = tf.truncated_normal(shape, stddev=init_std, name='TruncatedNormal')
    var = tf.get_variable(initializer=initial, validate_shape=validate_shape, name=name)
    return var

