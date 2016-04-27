#!/usr/bin/env python3

import numpy as np
import sys
import os
import random

import tensorflow as tf
from rntn_load_data import *

# pylint: disable=g-bad-name

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.003, 'Initial learning rate.')
flags.DEFINE_float('lamda', 0.5, 'Regularization parameter.')
flags.DEFINE_float('u_range', 0.0001, 'Range for uniform random weights.')
flags.DEFINE_integer('max_steps', 5000, 'Number of steps to run trainer.')
flags.DEFINE_float('cost_threshold', 0.5, 'Stop training if cost falls below this level.')
flags.DEFINE_integer('batch_size', 3, 'Training batch size.')
flags.DEFINE_integer('wvs', 8, 'Word vector size.')
flags.DEFINE_integer('n_labels', 5, 'Number of sentiment categories.')
flags.DEFINE_string('data_dir', '../rntn_data/toydata', 'Training data directory.')
flags.DEFINE_integer('max_sentence_length', 150, 'Maximum length sentence we can process.')
flags.DEFINE_boolean('log_device_placement', False, 'Log device placement')

FTYPE = np.float32

ACT_LEN = FLAGS.max_sentence_length * FLAGS.batch_size

i_is_leaf = 0
i_is_root = 1
i_left = 2
i_right = 3
i_parent = 4
i_idx = 5
i_phrase_id = 6
N_INFOS = 7


def node_info(node):
  info = np.array([1 if node.is_leaf else 0,
                   1 if node.is_root else 0,
                   0, 0, 0, 0, 0]).astype(np.int32).reshape([N_INFOS, 1])
  info[i_phrase_id] = node.phrase_id
  if not node.is_leaf:
    info[i_left] = node.left.idx
    info[i_right] = node.right.idx
  if not node.is_root:
    info[i_parent] = node.parent.idx
  return info


def fill_feed_dict(p_bounds, p_nodes, p_labels, sentences):
  a_nodes = np.array([node_info(node) for s in sentences for node in s])
  a_labels = np.array(
      [node.sentiment for s in sentences for node in s]).astype(FTYPE).reshape(
      [n_nodes, FLAGS.n_labels, 1])
  i_node = 0
  a_bounds = []
  for s in sentences:
    a_bounds.append([i_node, len(s)])
    i_node += len(s)
  feed_dict = {
    p_bounds: np.array(a_bounds, dtype=np.int32),
    p_nodes: a_nodes,
    p_labels: a_labels,
  }
  return feed_dict


def attach_info_to_nodes(sents, ddict):
  for s in sents:
    for node in s:
      entry = ddict[node.phrase]
      node.phrase_id = entry.phrase_id
      node.sentiment = np.array(np.mat(entry.sentiment_1hot), FTYPE).T


rntn = tf.VariableScope(False, name='rntn')

ONE = tf.ones([1, 1], name='fONE')
ZERO = tf.zeros([1, 1], name='fZERO')
iONE = tf.constant(1, name='iONE')
iZERO = tf.constant(0, name='iZERO')
NEG1 = tf.constant(-1, name='iNEG1')
true = tf.constant(True, tf.bool, name='TRUE')
false = tf.constant(False, tf.bool, name='FALSE')


def normal_weight_variable(shape, name=None):
  return tf.Variable(tf.random_normal(shape, 0, FLAGS.u_range), name=name)


def weight_variable(shape, name=None):
  return tf.Variable(tf.random_uniform(shape, -FLAGS.u_range, FLAGS.u_range), name=name)


def bias_variable(shape, name=None):
  return tf.Variable(tf.zeros(shape, dtype=FTYPE), name=name)


with tf.name_scope('weights'):
  V = weight_variable([2 * FLAGS.wvs, 2 * FLAGS.wvs, FLAGS.wvs], name='V')
  W = weight_variable([FLAGS.wvs, 2 * FLAGS.wvs], name='W')
  Ws = weight_variable([FLAGS.n_labels, FLAGS.wvs], name='Ws')

with tf.name_scope('weights/bias'):
  Wsb = bias_variable([FLAGS.n_labels, 1], name='Wsb')
  Wb = bias_variable([FLAGS.wvs, 1], name='Wb')

(sentences, dsdict, trains, valids, tests) = load_dataset(FLAGS.data_dir)
vocab_size = len(dsdict)
attach_info_to_nodes(sentences, dsdict)

n_nodes = sum(len(s) for s in sentences)

############## DEBUG #########
# a_nodes = np.array([node_info(node) for s in sentences for node in s])
# a_labels = np.array(
#         [node.sentiment for s in sentences for node in s]).astype(FTYPE).reshape(
#         [n_nodes, FLAGS.n_labels, 1])
# i_node = 0
# a_bounds = []
# for s in sentences:
#     a_bounds.append([i_node, len(s)])
#     i_node += len(s)
#
# bounds = tf.Variable(a_bounds,  name='bounds')
# nodes = tf.Variable(a_nodes, name='nodes')
# labels = tf.Variable(a_labels, name='labels')
############## DEBUG #########

#with tf.device('/cpu:0'):
bounds = tf.placeholder(tf.int32, shape=[len(sentences), 2], name='bounds')
nodes = tf.placeholder(tf.int32, shape=[n_nodes, N_INFOS, 1], name='nodes')
labels = tf.placeholder(tf.float32, shape=[n_nodes, FLAGS.n_labels, 1], name='labels')

f_dict = fill_feed_dict(bounds, nodes, labels, sentences)

words = weight_variable([len(dsdict), FLAGS.wvs, 1], name='words')

act_init = tf.constant(0.0, FTYPE, [ACT_LEN, FLAGS.wvs, 1], name='act_init')
with tf.device('/cpu:0'):
  activations = tf.Variable(tf.zeros([ACT_LEN, FLAGS.wvs, 1], FTYPE), name='activations')


def word_vec(i_node):
  with tf.op_scope([i_node, nodes], 'word_vec') as scope:
    phrase_id = tf.reshape(tf.slice(nodes, tf.pack([i_node, i_phrase_id, 0]), [1, 1, 1]), [],
                         name='phrase_id')
    wv = tf.slice(words, tf.pack([phrase_id, 0, 0]), [1, -1, -1], name='words_slice')
    return tf.reshape(wv, [FLAGS.wvs, 1], name=scope)

def is_leaf(i_node):
  with tf.op_scope([i_node, nodes], 'is_leaf') as scope:
    is_leaf_field = tf.pack([i_node, i_is_leaf, 0], name='is_leaf_field')
    n_slice = tf.slice(nodes, is_leaf_field, [1, 1, -1], name='node')
    result = tf.reshape(n_slice, [], name=scope)
  return result

def is_root(i_node):
  with tf.name_scope('node_info'):
    return tf.reshape(tf.slice(nodes, tf.pack([i_node, i_is_root, 0]), [1, 1, -1]), [],
      name='is_root')

def idx(i_node):
  with tf.name_scope('node_info'):
    return tf.reshape(tf.slice(nodes, tf.pack([i_node, i_idx, 0]), [1, 1, -1]), [],
                    name='idx')


def parent(i_node):
  with tf.name_scope('node_info'):
    return tf.reshape(tf.slice(nodes, tf.pack([i_node, i_parent, 0]), [1, 1, -1]), [],
                    name='parent')

def get_left(i_node, name=None):
  with tf.op_scope([i_node, nodes], name, 'get_left') as scope:
    return tf.reshape(tf.slice(nodes, tf.pack([i_node, i_left, 0]), [1, 1, -1]), [],
                      name=scope)

def get_right(i_node, name=None):
  with tf.op_scope([i_node, nodes], name, 'get_right') as scope:
    return tf.reshape(tf.slice(nodes, tf.pack([i_node, i_right, 0]), [1, 1, 1]), [],
                      name=scope)


def get_activation(i_node, acts, offset, name=None):
  with tf.op_scope([i_node, acts, offset], name, 'get_activation') as scope:
    return tf.reshape(
          tf.slice(acts, tf.pack([i_node+offset, 0, 0]), [1, -1, -1], name='activation'),
          [FLAGS.wvs, 1], name=scope)


def left_activation(i_node, acts, offset, name=None):
  with tf.op_scope([i_node, acts, offset], name, 'left_activation') as scope:
    return get_activation(get_left(i_node), acts, offset, name=scope)


def right_activation(i_node, acts, offset, name=None):
  with tf.op_scope([i_node, acts, offset], name, 'right_activation') as scope:
    return get_activation(get_right(i_node), acts, offset, name=scope)


def get_bounds(i_s):
  rec = tf.reshape(tf.slice(bounds, tf.pack([i_s, 0]), [1, 2], name='bounds_slice'),
                   [2], name='get_bounds')
  return rec[0], rec[1]


def rntn_tensor_forward(a, b, V, name=None):
  with tf.op_scope([a, b, V], name, 'TensorForward') as scope:
    wvs = FLAGS.wvs
    a = tf.convert_to_tensor(a, dtype=tf.float32, name='a')
    b = tf.convert_to_tensor(b, dtype=tf.float32, name='b')
    V = tf.convert_to_tensor(V, dtype=tf.float32, name='V')
    ab = tf.concat(0, (a, b), name='ab')
    return tf.matmul(
      tf.transpose(
        tf.reshape(
          tf.matmul(
            tf.transpose(ab, name='ab.T'),
            tf.reshape(V, [wvs * 2, wvs * wvs * 2], name='inter/V_flattened'),
              name='inter/abTxV'),
          [wvs * 2, wvs], name='inter/prod/reshape'),
          name='inter/prod/transpose'),
      ab, name=scope)

def std_forward(a, weights, bias_weights, name=None):
  with tf.op_scope([a, W, Wb], name, 'std_forward') as scope:
    a = tf.convert_to_tensor(a, dtype=tf.float32, name='input')
    weights = tf.convert_to_tensor(weights, dtype=tf.float32, name='weights')
    bias_weights = tf.convert_to_tensor(bias_weights, dtype=tf.float32, name='bias_weights')
    biased = tf.concat(1, (weights, bias_weights), name='biased')
    return tf.matmul(biased, a, name=scope)


def fwd_hidden(a, b):
  wvs = FLAGS.wvs
  ab = tf.concat(0, (a, b), name='ab')
  ab1 = tf.concat(0, (ab, ONE), name='ab1')
  # below works for tensor 2d x d x2d
  # tfinter = tf.reshape(tf.matmul(tf.transpose(ab),
  #                                tf.reshape(V, [wvs*2, wvs*wvs*2])),
  #                      [wvs, wvs*2])
  # below works for tensor 2d x 2d x d
  # inter = tf.transpose(
  #     tf.reshape(tf.matmul(tf.transpose(ab, name='ab.T'),
  #                          tf.reshape(V, [wvs * 2, wvs * wvs * 2]), name='V_reshaped'),
  #                [wvs * 2, wvs]), name='inter')
  # h = tf.matmul(inter, ab, name='h')
  h = rntn_tensor_forward(a, b, V, name='tensor_forward')
  #W_biased = tf.concat(1, (W, Wb), name='W_biased')
  #std_forward = tf.matmul(W_biased, ab1, name='std_forward')
  #return tf.add(h, std_forward, name='fwd_hidden')
  return tf.add(h, std_forward(ab1, W, Wb), name='fwd_hidden')


def get_node_info(i):
  with tf.op_scope([i, nodes], 'node_info') as scope:
    return tf.reshape(tf.slice(nodes, tf.pack([i, 0, 0]), [1, N_INFOS, -1],
                               name='node_info_slice'), [N_INFOS], name=scope)


def forward_node(i_node, acts, offset):
  def f_leaf():
    return word_vec(i_node)

  def f_nonleaf():
    a_left = left_activation(i_node, acts, offset)
    a_right = right_activation(i_node, acts, offset)
    return fwd_hidden(a_left, a_right)

  bool_is_leaf = tf.equal(is_leaf(i_node), 1, name='bool_is_leaf')
  return f_act(tf.cond(bool_is_leaf, f_leaf, f_nonleaf, name='cond_leaf_nonleaf'))


def f_act(x):
  return tf.tanh(x, name='f_act_tanh')


def forward_prop_nodes(i_start, size, acts, offset):
  # Note: In the corpus that we've seen, parse trees are always ordered such that
  # iteration forward through the list will be in bottom-up order.
  # Conversely, iteration in reverse is always top-down.
  # This enables a simple iterative algorithm. If this were not the case,
  # putting the nodes in order by a postorder traversal would fix it.
  def fwd_continue(*parms):
    (_, sz, cur, _) = parms
    return tf.less(cur, sz, name='cur_le_size')

  def forward_prop(*parms):
    (i0, sz, cur, act) = parms
    with tf.device('/gpu:0'):
      gact = act
      gcur = cur
      next_idx = i0 + gcur
    node_out = tf.reshape(forward_node(next_idx, act, offset), [1, FLAGS.wvs, 1], name='node_out')
    tf.scatter_add(gact, tf.pack([gcur]), node_out, name='act_update')
    act = gact
    return [i0, sz, cur + iONE, act]

  with tf.device('/cpu:0'):
    i_start = tf.convert_to_tensor(i_start, dtype=tf.int32, name='i_start')
    size = tf.convert_to_tensor(size, dtype=tf.int32, name='size')
    iZ = tf.convert_to_tensor(0, dtype=tf.int32, name='ZERO')

  while_parms = [i_start, size, iZ, acts]
  wresult = tf.while_loop(fwd_continue, forward_prop, while_parms, parallel_iterations=1,
                          name='forward_prop_while')
  (_, _, _, result) = wresult
  return tf.slice(result, [0, 0, 0], tf.pack([size, -1, -1]), name='fwd_prop_nodes')


def forward_sentence(i_s, acts, offset):
  i_start, i_size = get_bounds(i_s)
  return forward_prop_nodes(i_start, i_size, acts, offset)


def activation_to_sm(a):
  return tf.transpose(tf.nn.softmax(
      tf.transpose(
          activation_to_logits(a), name='act_to_softmax')))


def activations_to_sm(acts):
  return tf.map_fn(activation_to_sm, acts, name='acts_to_softmax')


def sm_rated(i_s):
  i_start, i_size = get_bounds(i_s)
  return tf.squeeze(tf.slice(labels,
                             tf.pack([i_start, 0, 0]),
                             tf.pack([i_size, -1, -1]), name='labels_slice'), [2],
                    name='sentence_labels')


def rated(i_s):
  return tf.argmax(sm_rated(i_s), 1, name='sentence_rating')


def predict_sentence(i_s, acts, offset=0):
  i_start, i_size = get_bounds(i_s)
  fwd_acts = forward_prop_nodes(i_start, i_size, acts, offset)
  return tf.argmax(activations_to_sm(fwd_acts), 1, name='predict_sentence')


def activation_to_logits(a):
  Ws_biased = tf.concat(1, (Ws, Wsb), name='Ws_biased')
  a1 = tf.concat(0, (a, ONE), name='a1')
  return tf.matmul(Ws_biased, a1, name='act_to_logits')


def logits(acts):
  logits = tf.map_fn(activation_to_logits, acts, name='logits')
  return tf.squeeze(logits, [2])


def sentence_logits(start, size, acts, offset):
  fwd_acts = forward_prop_nodes(start, size, acts, offset)
  return logits(fwd_acts)


def cost1(i_s, acts, out_ptr):
  start, size = get_bounds(i_s)
  s_labels = sm_rated(i_s)
  s_logits = sentence_logits(start, size, acts, out_ptr)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      s_logits, s_labels, name='c1xentropy')
  return tf.reduce_sum(cross_entropy)


def cost1_batch(i_s, acts, out_ptr):
  start, size = get_bounds(i_s)
  s_labels = sm_rated(i_s)
  predicts = sentence_logits(start, size, acts, out_ptr)
  result = tf.nn.softmax_cross_entropy_with_logits(
      predicts, s_labels, name='c1bxentropy')
  return result, size


# def cost_batch(indices, acts):
#     logits = cost_batch_sub(indices, acts)
#     labs = tf.squeeze(batch_labels(indices), [2])
#     ce = tf.nn.softmax_cross_entropy_with_logits(logits, labs)
#     loss = tf.reduce_mean(ce)
#     return loss


def calc_loss(logs, labs):
  ce = tf.nn.softmax_cross_entropy_with_logits(logs, labs)
  ce_mean = tf.reduce_mean(ce)
  with tf.op_scope([V, W, Ws, Wb, Wsb, words], 'regularization') as scope:
    regularizers = tf.square(tf.nn.l2_loss(W) + tf.nn.l2_loss(Wb) +
                    tf.nn.l2_loss(Ws) + tf.nn.l2_loss(Wsb) +
                    tf.nn.l2_loss(V) + tf.nn.l2_loss(words))
  loss = ce_mean + regularizers * FLAGS.lamda
  return loss


def batch_labels(indices):
  inits = tf.zeros([1, FLAGS.n_labels, 1])

  def bl_cond(*parms):
    i, idxs, _ = parms
    return tf.less(i, tf.size(idxs))

  def bl_body(*parms):
    i, idxs, labs = parms
    # really?
    i_s = tf.reshape(tf.slice(idxs, tf.pack([i]), [1]), [])
    start, size = get_bounds(i_s)
    i_labels = tf.slice(labels,
                        tf.pack([start, 0, 0]),
                        tf.pack([size, -1, -1]), name='labels_slice')
    new_labels = tf.cond(tf.equal(i, iZERO),
                         lambda: i_labels,
                         lambda: tf.concat(0, [labs, i_labels]))
    return i + iONE, idxs, new_labels
  with tf.device('/cpu:0'):
    iZ = tf.convert_to_tensor(0, dtype=tf.int32)
  while_parms = [iZ, indices, inits]
  _, _, results = tf.while_loop(bl_cond, bl_body, while_parms, name='batch_labels')
  return tf.squeeze(results, [2])


def batch_logits(indices, acts):
  init_outs = tf.zeros([1, FLAGS.wvs, 1])

  def logits_continue(*parms):
    cur, idxs, _, _, _ = parms
    return tf.less(cur, tf.size(idxs), name='batch_done')

  def logits_batch_body(*parms):
    i, idxs, ptr, css, act = parms
    i_s = tf.reshape(tf.slice(idxs, tf.pack([i]), [1]), [])
    start, size = get_bounds(i_s)
    outs = forward_prop_nodes(start, size, acts, ptr)
    new_css = tf.cond(tf.equal(i, iZERO),
                      lambda: outs,
                      lambda: tf.concat(0, [css, outs]))
    return i + iONE, indices, ptr + size, new_css, acts
  with tf.device('/cpu:0'):
    iZ =  tf.convert_to_tensor(0, dtype=tf.int32)
  zero_activations(acts)
  while_parms = [iZ, indices, iZ, init_outs, acts]
  _, _, _, outs, _ = tf.while_loop(logits_continue, logits_batch_body, while_parms,
                                   parallel_iterations=1, name='batch_logits')
  lumpy_logits = tf.map_fn(activation_to_logits, outs, name='raw_logits')
  logits = tf.squeeze(lumpy_logits, [2], name='logits')
  return logits

# def batch_correct(indices, acts):
#   n_correct = tf.convert_to_tensor(0.0, tf.float32)
#   def correct_continue(*parms):
#     cur, idxs, _, _ = parms
#     return tf.less(cur, tf.size(idxs))
#   def correct_body(*parms):
#     def f_correct():
#       correct(i_s, acts)
#     i, idxs, act, crx = parms
#     i_s = tf.reshape(tf.slice(idxs, tf.pack([i]), [1]), [])
#     rightness = tf.cast(tf.cond(
#         correct(i_s, acts),
#         lambda: iONE,
#         lambda: iZERO), tf.float32)
#     crx = tf.add(crx, rightness)
#     return i+iONE, idxs, act, crx
#   while_parms = [iZERO, indices, acts, n_correct]
#   _, _, _, n_correct = tf.while_loop(correct_continue, correct_body, while_parms)
#   return tf.div(n_correct, tf.cast(tf.size(indices), tf.float32))

def accuracy(logs, labs):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logs: Logits tensor, float - [batch_size, NUM_CLASSES].
    labs: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  labs = tf.argmax(labs, 1)
  correct = tf.nn.in_top_k(logs, labs, 1)
  # Return the number of true entries.
  return tf.reduce_mean(tf.cast(correct, tf.float32))

# def percent_correct(indices, acts):
#   n_correct = 0
#   for i in indices:
#     n_correct += 1 if correct(tf.convert_to_tensor(i), acts).eval() else 0
#   corrects = np.sum([1 if correct(tf.convert_to_tensor())])

def correct(i_s, acts):
  return tf.equal(rated(i_s), predict_sentence(i_s, acts))


# def train_step(indices, acts):
#   return opt.minimize(cost_batch(indices, acts))


# def accuracy(i_s, acts):
#   i_s = tf.convert_to_tensor(i_s, tf.int32)
#   return tf.reduce_mean(tf.cast(batch_correct(i_s, acts), tf.float32))


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
c_proto = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=FLAGS.log_device_placement)


def zero_activations(acts):
  acts = tf.zeros_like(acts)
  return acts


# random producer for training batches
sess = tf.InteractiveSession(config=c_proto)
# Start input enqueue threads.
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess=sess, coord=coord)

#with tf.device('/cpu:0'):
i_ss = tf.placeholder(tf.int32, name='i_ss')
#valid_ss = tf.placeholder(tf.int32, name='valid_ss')

setup_done = False


def debug_init():
  tf.histogram_summary('activations', activations)
  f_dict[i_ss] = random.sample(range(len(trains)), FLAGS.batch_size)
  logits = batch_logits(i_ss, activations.ref())
  labs = batch_labels(i_ss)
  loss = calc_loss(logits, labs)
  tf.scalar_summary('cost_summary', loss)
  writer = tf.train.SummaryWriter(
      '/Users/rgobbel/src/pymisc/rntn_tf/tf_logs', sess.graph)
  merged = tf.merge_all_summaries()
  sess.run(tf.initialize_all_variables())
  return logits, labs, loss, merged, writer


ro = tf.RunOptions(trace_level='FULL_TRACE')


def run_training(cost_threshold=FLAGS.cost_threshold, max_steps=FLAGS.max_steps):
  global setup_done
  cost_value = 1e9
  accuracy_value = 0.0
  # if setup_done is False:
  setup_done = True
  opt = tf.train.AdamOptimizer()
  # try:
  #opt = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
  i_trains = [s.idx for s in trains]
  i_valids = [s.idx for s in valids]
  i_tests = [s.idx for s in tests]
  i_all = [s.idx for s in sentences]
  logits = batch_logits(i_ss, activations.ref())
  labs = batch_labels(i_ss)
  loss = calc_loss(logits, labs)
  i_ss_accuracy = accuracy(logits, labs)
  #v_labs = batch_labels(valid_ss)
  #v_logits = batch_logits(valid_ss, activations.ref())
  #v_loss = calc_loss(v_logits, v_labs)
  #train_accuracy = accuracy(logits, labs)
  #valid_accuracy = accuracy(v_logits, v_labs)
  # test_accuracy = accuracy(i_tests, activations.ref())
  train_op = opt.minimize(loss)
  #tf.histogram_summary('activations', activations)
  tf.histogram_summary('samples', i_ss)
  tf.scalar_summary('loss', loss)
  #tf.scalar_summary('training accuracy', train_accuracy)
  tf.scalar_summary('validation accuracy', i_ss_accuracy)
  # tf.scalar_summary('test accuracy', test_accuracy)
  merged = tf.merge_all_summaries()
  sess.run(tf.initialize_all_variables())
  writer = tf.train.SummaryWriter(
      '/Users/rgobbel/src/pymisc/rntn_tf/tf_logs', sess.graph)
  # except Exception as exc:
  #     print('Exception: {0}'.format(exc))
  # setup_done = False
  f_dict[i_ss] = random.sample(i_trains, FLAGS.batch_size)
  _, cost_value = sess.run([train_op, loss], feed_dict=f_dict)
  #f_dict[valid_ss] = i_valids
  _ = sess.run(zero_activations(activations.ref()), feed_dict=f_dict)
  print('starting')
  accuracy_value = sess.run([i_ss_accuracy], feed_dict=f_dict)
  for step in range(max_steps):
    #_ = sess.run(zero_activations(activations.ref()), feed_dict=f_dict)
    f_dict[i_ss] = random.sample(i_trains, FLAGS.batch_size)
    #logits = batch_logits(i_ss, activations.ref())
    #labs = batch_labels(i_ss)
    _, _, cost_value, _ = sess.run([tf.pack([i_ss]), train_op, loss], feed_dict=f_dict)
    #_ = sess.run(zero_activations(activations.ref()), feed_dict=f_dict)
    f_dict[i_ss] = i_valids
    _, valid_accuracy_value = sess.run([loss, i_ss_accuracy], feed_dict=f_dict)
    (summ,) = sess.run([merged], feed_dict=f_dict)
    # summ = sess.run([merged], feed_dict=f_dict)
    writer.add_summary(summ, step)
    writer.flush()
    print('.', end='', flush=True)
    if cost_value < cost_threshold:
      return step, cost_value, valid_accuracy_value
  return max_steps, cost_value, valid_accuracy_value


def seval(expr):
  return sess.run(expr, feed_dict=f_dict)

#run_training()
