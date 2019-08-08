
import numpy as np
import tensorflow as tf

from Model.Helpers.layers import _flat_features, _undo_flat_features

def _gdl_loss(gen_frames, gt_frames, mask, alpha=1.):
  """
  Calculates the sum of GDL losses between the predicted and gt frames.
  @param gen_frames: The predicted frames at each scale.
  @param gt_frames: The ground truth frames at each scale
  @param alpha: The power to which each gradient term is raised.
  @return: The GDL loss.
  """
  # create filters [-1, 1] and [[1],[-1]]
  # for diffing to the left and down respectively.
  pos = tf.constant(np.identity(1), dtype=tf.float32)
  neg = -1 * pos
  # [-1, 1]
  filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)
  # [[1],[-1]]
  filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])
  strides = [1, 1, 1, 1]  # stride of (1, 1)
  padding = 'SAME'

  gen_dx = tf.abs(tf.nn.conv2d(gen_frames, filter_x, strides, padding=padding))
  gen_dy = tf.abs(tf.nn.conv2d(gen_frames, filter_y, strides, padding=padding))
  gt_dx = tf.abs(tf.nn.conv2d(gt_frames, filter_x, strides, padding=padding))
  gt_dy = tf.abs(tf.nn.conv2d(gt_frames, filter_y, strides, padding=padding))

  grad_diff_x = tf.abs(gt_dx - gen_dx)
  grad_diff_y = tf.abs(gt_dy - gen_dy)

  gdl_loss = (grad_diff_x ** alpha + grad_diff_y ** alpha)
  with tf.variable_scope('UNDOFLAT'):
      gdl_loss = _undo_flat_features(gdl_loss, index=4)
  gdl_loss = tf.identity(tf.reduce_sum(tf.reduce_sum(gdl_loss, [2,3,4]) * mask) / (tf.reduce_sum(mask)*20*40), name='Output')

  # condense into one tensor and avg
  return gdl_loss

def imgloss(in_tensor, trg_tensor):

    in_tensor = in_tensor[:,:-1,:,:,:]

    loss = tf.square(in_tensor - trg_tensor, name='PLoss')
    hits = tf.cast(tf.equal(in_tensor, trg_tensor), dtype=tf.float32, name='Hits')

    seq_lens = tf.get_default_graph().get_tensor_by_name('Inputs/SeqLengths:0')

    mask = tf.sequence_mask(lengths=seq_lens,
                            maxlen=tf.reduce_max(seq_lens),
                            dtype=tf.float32,
                            name='LossMask')

    loss = tf.identity(tf.reduce_sum(tf.reduce_sum(loss, [2,3,4]) * mask) / (tf.reduce_sum(mask)*20*40), name='PLossAVG')
    accuracy = tf.identity(tf.reduce_sum(tf.reduce_sum(hits, [2,3,4]) * mask) / (tf.reduce_sum(mask)*20*40), name='AccuracyAVG')

    with tf.variable_scope('FLATFEAT-PRED'):
        in_flat = _flat_features(in_tensor, 2, True)
    with tf.variable_scope('FLATFEAT-TRG'):
        trg_flat = _flat_features(trg_tensor, 2, True)

    with tf.variable_scope('GdlLoss'):
        gdl_loss = _gdl_loss(in_flat, trg_flat, mask)

    img_loss = tf.identity(gdl_loss + loss, name='ImgLoss')

    tf.summary.scalar('PLoss', loss)
    tf.summary.scalar('GdlLoss', gdl_loss)
    tf.summary.scalar('ImgLoss', img_loss)

    tf.summary.image('PredFrame',in_flat)
    tf.summary.image('TrgFrame',trg_flat)

    return [loss, gdl_loss, img_loss], hits, accuracy