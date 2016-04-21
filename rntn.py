"""Recursive Neural Tensor Network.

Usage:
  rntn check-grad [options]
  rntn train [options]
  rntn accuracy (training | validation | test) [--print-tests=(True|False)] [options]
  rntn -h | --help

Options:
  -h --help                   Show this message.
  --log-name=<str>            Base name for log file.
  --config-name=<str>         Configuration file name.
  --train-limit=<n>           Limit size of training set.
  --valid-limit=<n>           Limit size of validation set.
  --test-limit=<n>            Limit size of test set.
  --data-from <dir>           Load data from named directory.
  --params-from <file>        Load net parameters from named file.
  --seed=<n>                  Random number seed.
  --word-vector-size=<n>      Size of word vectors. NOTE: Will not work with pickled parameters!
  --max-epochs=<n>            Number of batch training epochs to run.
  --learning-rate=<n>         Learning rate.
  --wvlr=<n>                  Separate fixed learning rate for word vectors.
  --lambda=<n>                Regularization parameter.
  --batch-size=<n>            Training batch size.
  --epsilon=<n>               Increment for gradient checking.
  --report-interval=<n>       Step number and cost will be printed every <interval> epochs.
  --checkpoint-interval=<n>   A checkpoint file will be written every <interval> epochs.
  --checkpoint-base=<str>     Base for checkpoint file names.
  --checkpoint-dir=<str>      Directory for writing checkpoint files. [default: checkpoints]
  --ckpt-compress=<boolean>   Whether or not to compress checkpoint files. [default: False]
  --validate-interval=<n>     Run accuracy test on validation set every <n> epochs.
  --check-training=<bool>     Include training set in accuracy test. [default: False]
  --cost-threshold=<n>        Stop training if cost drops below this value.
  --weight-initial-range=<n>  Initial range for weight initialization.
  --disable-tensor=<boolean>  Run as RNN without tensor [default: False]
  --use-adamax=<boolean>      Use Adamax adaptive optimizer. [default: True]
  --print-tests=<boolean>     Print accuracy test result. [default: True]
"""

from __future__ import print_function
from rntn_data import *
import numpy as np
import pickle
import gzip
import docopt
import time
import datetime as dt
import logging as lg
import random
import collections

FTYPE = np.float64

default_log_name = 'rntn.log'
default_config_name = 'rntn.ini'
default_learning_rate = 0.002
default_wvlr = 0.002
default_lambda = 1.0
u_range = 0.0001
default_max_epochs = 2000
default_batch_size = 1
default_epsilon = 1e-7
default_seed = 1
default_cost_threshold = None
default_word_vec_size = 10 # make this number small for easier debugging
n_labels = 5
#default_data_dir = 'sstdata'
default_data_dir = 'toydata'

default_train_limit = None
default_test_limit = None
default_valid_limit = None

g_disable_tensor = False


v1 = np.ones([1, 1], FTYPE)


def report(msg):
    print(msg)
    #lg.getLogger('rntn').info(msg)


def weight_variable(shape):
    return np.random.rand(*shape).astype(FTYPE) * 2 * u_range - u_range


def bias_variable(shape):
    return np.random.rand(*shape).astype(FTYPE) * 2 * u_range - u_range
    #return np.zeros(shape, np.float32)


def attach_info_to_nodes(sents, ddict, wvecs):
    for s in sents:
        for node in s:
            node_entry = ddict[node.phrase]
            node.phrase_id = node_entry.phrase_id
            node.sentiment = np.array(np.mat(node_entry.sentiment_1hot), FTYPE).T
            node.word_vec = wvecs[node.phrase_id,:,:]


def dump_params(filename, params, dir=None, compress=False):
    if compress:
        extension = 'pklz'
    else:
        extension = 'pickle'
    if dir is not None:
        filename = dir + '/' + filename
    full_filename = '{}.{}'.format(filename, extension)
    if compress:
        with gzip.open(full_filename, 'wb') as f:
            pickle.dump(params, f)
    else:
        with open(full_filename, 'wb') as f:
            pickle.dump(params, f)


def load_params(filename):
    if filename[-4:] == 'pklz':
        with gzip.open(filename, 'rb') as f:
            return np.load(f)
    else:
        with open(filename, 'rb') as f:
            return np.load(f)


def reset_params():
    global word_vec_size, vocab_size, dsdict, sentences, gL
    global gW, gWs, gWb, gV, gParams, default_seed
    global ggl_hist, gam_params

    np.random.seed(default_seed)

    gam_params = None
    ggl_hist = None

    gL = weight_variable([vocab_size, word_vec_size, 1])

    attach_info_to_nodes(sentences, dsdict, gL)

    gWs = weight_variable([n_labels, word_vec_size])
    gWsb = bias_variable([n_labels, 1])

    gW = weight_variable([word_vec_size, 2*word_vec_size])
    gWb = bias_variable([word_vec_size, 1])

    gV = weight_variable([word_vec_size, 2*word_vec_size, 2*word_vec_size])

    gParams = [gV, gW, gWs, gWb, gWsb, gL]


def unroll_params(params):
    total_size = sum(a.size for a in params)
    unrolled = np.zeros(total_size, FTYPE)
    shapes = []
    i = 0
    for param in params:
        shapes.append(param.shape)
        rav = np.ravel(param)
        unrolled[i:i+len(rav)] = rav
        i += len(rav)
    return unrolled, shapes


def reroll_params(unrolled, shapes):
    rerolled = []
    i = 0
    for shape in shapes:
        this_len = np.prod(shape)
        rerolled.append(np.reshape(unrolled[i:i+this_len], shape))
        i += this_len
    return rerolled


def check_gradients(data, params, epsilon=default_epsilon, lamda=default_lambda,
                    u=u_range, sin_init=False):
    unrolled, shapes = np.copy(unroll_params(params))
    max_diff = 0.0
    max_i = 0
    if sin_init:
        for i in range(len(unrolled)):
            unrolled[i] = np.sin(i+1) * u
        params = reroll_params(unrolled, shapes)
    numgrad = np.zeros(len(unrolled), np.float32)
    diffs = np.zeros_like(unrolled)
    # if g_disable_tensor:
    #     params[0] = np.zeros_like(params[0])
    names = ['V', 'W', 'Ws', 'W_bias', 'Ws_bias', 'L']
    dwords = wvdict(params[1].shape[0])
    grad_words = wvdict(params[1].shape[0])
    num_grad_words = wvdict(params[1].shape[0])
    np.random.seed(default_seed)
    for s in sentences:
        for n in s.nodes:
            fake_sentiment = np.zeros([n_labels, 1])
            fake_sentiment[random.randrange(n_labels)] = 1.0
            n.sentiment = fake_sentiment
    (acost, agrads) = fwd_back_batch(params[:-1], params[-1], dwords, grad_words, data, lamda)
    unagrads, _ = unroll_params(agrads)
    gwords = np.concatenate([grad_words[gw] for gw in grad_words], axis=1)
    agrads.append(gwords)
    for i in range(len(unrolled)):
        # Note: this approach avoids numerical instabilities with very small quantities
        p0 = unrolled[i]
        unrolled[i] = p0 + epsilon
        rparams = reroll_params(unrolled, shapes)
        dwords.clear()
        np.random.seed(default_seed)
        (pcost, _) = fwd_back_batch(rparams[:-1], rparams[-1], dwords,
                                         num_grad_words, data, lamda)
        unrolled[i]  = p0 - epsilon
        rparams = reroll_params(unrolled, shapes)
        dwords.clear()
        np.random.seed(default_seed)
        (mcost, _) = fwd_back_batch(rparams[:-1], rparams[-1], dwords,
                                         num_grad_words, data, lamda)
        unrolled[i] = p0
        numgrad[i] = (pcost - mcost) / (2 * epsilon)
        if i < len(unagrads):
            diffs[i] = numgrad[i] - unagrads[i]
            if np.abs(diffs[i]) > max_diff:
                 max_diff = numgrad[i]
                 max_i = i
    gwords = np.concatenate([grad_words[gw] for gw in grad_words], axis=1)
    numgrads = reroll_params(numgrad, shapes)[:-1]
    numgrads.append(gwords)
    print('Cost = {}, epsilon={}'.format(acost, epsilon))
    print('Differences:')
    print('{0:>8}{1:>{width}}{2:>{width}}{3:>{width}}{4:>{width}}'.format(
            'Value', 'Numeric', 'Analytical','Delta', 'Ratio', width=25))
    for s, gnum, ganal in zip(names, numgrads, agrads):
        gn = np.sum(gnum)
        ga = np.sum(ganal)
        print('{0:>8}:{1:>{width}.{prec}f}{2:>{width}.{prec}f}{3:>{width}.{prec}f}{4:>{width}.{'
           'prec}f}'.
              format(
                s, gn, ga, gn - ga, gn/ga,
                width=25, prec=20))
    print('Largest difference was {:.{prec}f}, at index {}'.
          format(max_diff, max_i, prec=15))
    return reroll_params(diffs, shapes)[:-1]
    #return list(zip(numgrads, agrads))


def tanh_prime(x):
    return 1.0 -  np.tanh(x)**2


def softmax(x):
    """Compute softmax values for each sets of scores in x (numerically stable version)."""
    e_x = np.exp(x - x.max())
    return e_x / e_x.sum()


def softmax_prime(x_sm_value):
    return x_sm_value * (1.0 - x_sm_value)


def predict_cost(prediction, target):
    return 1 - np.log(prediction[np.argmax(target)])


def forward_prop(a, b, V, W, Wb):
    W_biased = np.concatenate([W, Wb], axis=1)
    ab = np.concatenate([a, b])
    ab1 = np.concatenate([a, b, v1])
    h = np.dot(np.dot(ab.T, V), ab)[0]
    std_forward = np.dot(W_biased, ab1)
    return h + std_forward

def forward_logits(Ws, Wsb, a):
    Ws_biased = np.concatenate((Ws, Wsb), axis=1)
    return np.dot(Ws_biased,
                  np.concatenate((a, v1)))

def forward_predict(Ws, Wsb, a):
  return softmax(forward_logits(Ws, Wsb, a))


def forward_prop_sentence(s, V, W, Ws, Wb, Wsb, L):
    predictions = []
    targets = []
    outputs = []
    word_vecs = []
    # Note: In the corpus that we've seen, parse trees are always ordered such that
    # iteration forward through the list will be in bottom-up order.
    # Conversely, iteration in reverse is always top-down.
    # This is extremely convenient. If this were not the case, a topological sort would fix it.
    for node in s:
        if node.is_leaf:
            activation = L[node.phrase_id]
            word_vecs.append(activation)
        else:
            activation = forward_prop(outputs[node.left.idx], outputs[node.right.idx], V, W, Wb)
        output = np.tanh(activation)
        target = node.sentiment
        prediction = forward_predict(Ws, Wsb, output)
        predictions.append(prediction)
        targets.append(target)
        outputs.append(output)
    return predictions, targets, outputs, word_vecs


def get_words(ss):
    """Return nodes with a phrase of length 1, unique by phrase."""
    return list({node.phrase:node
                 for s in ss
                 for node in s.nodes
                 if len(node.phrase.split(' ')) == 1}.values())


def pos_neg_words(words):
    return [word for word in words
            for rating in [np.argmax(word.sentiment)
                           for word in words]
            if (rating < 2 or rating > 2)]


def predict_sentence(s, params):
    (predictions, _, _, _) = forward_prop_sentence(s, *params)
    return np.argmax(predictions[-1])


def pos_neg_neutral_predict(sentence, params):
    (predictions, _, _, _) = forward_prop_sentence(sentence, *params)
    overall_sentiment = np.argmax(predictions[-1])
    if overall_sentiment < 2:
        return '-'
    elif overall_sentiment == 2:
        return 'o'
    else:
        return '+'


def pos_neg_neutral_rated(sentence):
    overall_sentiment = np.argmax(sentence.root.sentiment)
    if overall_sentiment < 2:
        return '-'
    elif overall_sentiment == 2:
        return 'o'
    else:
        return '+'


def pos_neg(ss):
    return [s for s in ss if pos_neg_neutral_rated(s) != 'o']


def pos_neg_accuracy(ss, params):
    n_correct = sum([(1.0 if (pos_neg_neutral_predict(s, params) ==
                              pos_neg_neutral_rated(s)) else 0.0)
                     for s in ss])
    return n_correct/len(ss)


def regularization_all(lamda, m, params, word_vecs):
    th_sq_norm = np.sum(np.square(np.linalg.norm(theta)) for theta in params)
    L_sq_norm = np.sum(np.sum(np.square(np.linalg.norm(word_vecs))))
    return (lamda/(2*m)) * (th_sq_norm + L_sq_norm)


def aggregate_cost(targets, predictions, lamda, m, params, word_vecs):
    err_components = np.sum(predict_cost(y, t)
                                for (t, y) in zip(targets, predictions))[0]
    reg = regularization_all(lamda, m, params, word_vecs)
    return err_components/m + reg


def regularization(lamda, m, theta):
    return lamda/(2*m) * np.sum(np.square(theta))


def E_theta(targets, predictions, lamda, m, theta):
    err_components = (np.sum(sum(t * np.log(y) for (t, y) in zip(targets, predictions))))
    reg = regularization(lamda, m, theta)
    return np.sum(err_components) + reg


def delta_sm_i(prediction, target, Ws, node_out):
    err_predict = np.dot(Ws.T, prediction - target)
    f_prime_node = 1 - node_out ** 2
    delta_sm = err_predict * f_prime_node
    return delta_sm


def delta_down_i(W, delta_com, V, inputs):
    grad_inputs = 1.0 - inputs**2
    delta_down_w = np.dot(W.T, delta_com)
    VVT = V + np.transpose(V, (0, 2, 1))
    pr1 = (delta_com.T * np.dot(VVT, inputs).T).T
#    pr1 = delta_com * np.dot(VVT, inputs)
    S = np.sum(pr1, axis=0)
    return (delta_down_w + S) * grad_inputs


def wvdict(word_vector_size):
    return collections.defaultdict(
            lambda: np.zeros([word_vector_size, 1], dtype=FTYPE))


def backprop_sentence(s, dwords, V, W, Ws, Wb, Wsb,
                      predictions, targets, outputs):
    s_top_down = reversed(list(zip(
            s.nodes, predictions, targets, outputs)))
    delta_down = [np.zeros([word_vec_size, 1], FTYPE)] * len(s)
    delta_v = np.zeros_like(V, FTYPE)
    delta_w = np.zeros_like(W, FTYPE)
    delta_ws = np.zeros_like(Ws, FTYPE)
    delta_wb = np.zeros_like(Wb, FTYPE)
    delta_wsb = np.zeros_like(Wsb, FTYPE)
    # reverse index goes top-down
    for (node, prediction, target, output) in s_top_down:
        delta_sm = delta_sm_i(prediction, target, Ws, output)
        delta_ws += np.dot(prediction - target, output.T)
        delta_wsb += (prediction - target)

        if node.is_root:
            delta_complete = delta_sm
        else:
            delta_complete = delta_sm + delta_down[node.idx]

        if node.is_leaf:
            dwords[node.phrase_id] += delta_complete
        else:
            inp = np.concatenate((outputs[node.left.idx],
                                  outputs[node.right.idx]), axis = 0)
            dd = delta_down_i(W, delta_complete, V, inp)
            delta_down[node.left.idx] = dd[:word_vec_size]
            delta_down[node.right.idx] = dd[word_vec_size:]
            delta_w += np.dot(delta_complete, inp.T)
            delta_wb += delta_complete
            input_inputT = np.dot(inp, inp.T)
            delta_v += np.squeeze(np.tensordot(delta_complete, input_inputT, axes=0), 1)
    deltas = [delta_v, delta_w, delta_ws, delta_wb, delta_wsb]
    return deltas


def fwd_back_batch(params, L, dwords, grad_words, examples, lamda):
    (V, W, Ws, Wb, Wsb) = params
    m = sum(len(s) for s in examples)
    d_batch = [np.zeros_like(theta, FTYPE) for theta in params]
    batch_scale = 1.0/m
    reg_scale = lamda/m
    dwords.clear()
    predictions = []
    targets = []
    word_vecs = []
    for s in examples:
        s_preds, s_targets, outs, s_words = forward_prop_sentence(s, V, W, Ws, Wb, Wsb, L)
        predictions += s_preds
        word_vecs += s_words
        targets += s_targets
        deltas = backprop_sentence(s, dwords, V, W, Ws, Wb, Wsb, s_preds, s_targets, outs)
        for j in range(len(deltas)):
            d_batch[j] += deltas[j]
    word_vecs = np.array([dwords[key] for key in dwords])
    total_cost = aggregate_cost(targets, predictions, lamda, m, params, np.array(word_vecs))
    for key in dwords:
        grad_words[key] += dwords[key] * batch_scale + L[key] * reg_scale
    grads = [d_theta * batch_scale  + theta * reg_scale
              for (d_theta, theta) in zip(params, d_batch)]
    return total_cost, grads


def train(params, trains, valids=None, gl_hist=None, am_params=None, max_epochs=100,
          batch_size=None, lr=default_learning_rate, lamda=default_lambda, use_adamax=True,
          report_interval=1,
          checkpoint_interval=None, cp_file_base=None, ckpt_compress=False, checkpoint_dir=None,
          valid_interval=None, check_trains=False, check_valids=True, print_test_results=True,
          cost_threshold=None, wvlr=default_wvlr):
    if cp_file_base is None: cp_file_base = 'rntn_checkpoint'
    if batch_size is None:
        batch_size = len(trains)
    else:
        batch_size = min(batch_size, len(trains))
    if max_epochs is None: max_epochs = default_max_epochs
    (V, W, Ws, Wb, Wsb, L) = params
    # if g_disable_tensor:
    #     V = np.zeros_like(V, FTYPE)
    tparams = params[:-1]
    gr_epsilon = 1e-6
    msg1 = 'Batch size={}, max epochs={}'.format(batch_size, max_epochs)
    msg2 = ', cost threshold={}, learning rate={}, wvlr={}, lambda={}'.format(
            cost_threshold, lr, wvlr, lamda)
    report(msg1+msg2)
    report('Training set size={}, validation set size={}, adamax={}, tensor={}'.format(
            len(trains), len(valids), use_adamax, not g_disable_tensor))
    if use_adamax:
        report('Word vector history preloaded={}, optimizer params preloaded={}'.format(
            gl_hist is not None, am_params is not None
        ))
    report('Report_interval={}, validation interval={}, checkpoint interval={}, checkpoint base={}'.format(
        report_interval, valid_interval, checkpoint_interval, cp_file_base)
    )
    m = batch_size
    if am_params is None: # Adamax parameters
        # zero init of first moment
        opt_m = [np.zeros_like(p, FTYPE) for p in tparams]
        # zero init of exponentially weighted infinity norm
        opt_u = [np.zeros_like(p, FTYPE) for p in tparams]
        #beta1 = 0.9
        #beta2 = 0.999
        am_params = (opt_m, opt_u)
    else:
        (opt_m, opt_u) = am_params
    if gl_hist is None:
        gl_hist = np.ones_like(L, FTYPE) * gr_epsilon
    last_epoch = 0
    dwords = wvdict(W.shape[0])
    grad_words = wvdict(W.shape[0])
    for i in range(max_epochs):
        # if g_disable_tensor:
        #     V = np.zeros_like(V, FTYPE)
        last_epoch = i
        grad_words.clear()
        batch = [trains[i] for i in random.sample(range(len(trains)), m)]
        cost, grads = fwd_back_batch(
                tparams, L, dwords, grad_words, batch, lamda)
        if use_adamax:
            opt_m, opt_u, tparams = adamax(grads, tparams, opt_m, opt_u, i+1, lr)
        else: # good old-fashioned SGD
            for p, grad in zip(params, grads):
                p -= lr * grad
        # Inline Adamax weight update
        # for g, p, m_t, u_t in zip(grads, params, opt_m, opt_u):
        #     m_t = (beta1 * m_t) + (1 - beta1) * g
        #     u_t = np.fmax(np.ones_like(g) * beta2 * u_t, np.absolute(g))
        #     p -= (lr / (1 - np.power(beta1, i + 1))) * m_t/(u_t + gr_epsilon)

        # Word embedding vector is dealt with separately.
        # Adagrad for word vector update. This should in effect give more weight to
        # infrequently-occurring words.
        for key in grad_words:
            gl_hist[key] += grad_words[key]**2
            L[key] -= (wvlr * grad_words[key]) / np.sqrt(gl_hist[key])
        if i % report_interval == 0:
            report('Epoch {}, cost = {}'.format(i+1, cost))
        if valid_interval is not None and i % valid_interval == 0:
            if check_trains:
                check_set_accuracy(trains, params, print_test_results, 'Training')
            if check_valids and valids is not None:
                check_set_accuracy(valids, params, print_test_results, 'Validation')
        if checkpoint_interval is not None and i % checkpoint_interval == 0:
            strnow = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
            dump_params('{}_{}_{}'.format(cp_file_base, strnow, i),
                        params+[gl_hist, am_params], dir=checkpoint_dir, compress=ckpt_compress)
        if cost_threshold is not None and cost < cost_threshold:
            report('Stopping with cost={} (threshold={})'.format(cost, cost_threshold))
            break
    report('Stopping after {} epochs'.format(last_epoch))
    strnow = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    dumpable = [V, W, Ws, Wb, Wsb, L, gl_hist, am_params]
    dump_params('{}_{}_final'.format(cp_file_base, strnow), dumpable, dir=checkpoint_dir, compress=ckpt_compress)
    return params, gl_hist, am_params


# def Adam(grads, params, m, u, i, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
#     i_t = i + 1.0
#     fix1 = 1.0 - (1.0 - b1)**i_t
#     fix2 = 1.0 - (1.0 - b2)**i_t
#     lr_t = lr * (np.sqrt(fix2) / fix1)
#     for p, g in zip(params, grads):
#         m = 0.0
#         v = 0.0
#         m_t = (b1 * g) + ((1.0 - b1) * m)
#         v_t = (b2 * T.sqr(g)) + ((1.0 - b2) * v)
#         g_t = m_t / (T.sqrt(v_t) + e)
#         p_t = p - (lr_t * g_t)
#         updates.append((m, m_t))
#         updates.append((v, v_t))
#         updates.append((p, p_t))
#     return m, u, params


def adamax(grads, params, ms, us, i, lr=0.0002, b1=0.9, b2=0.999, epsilon=1e-8):
    lr_t = lr / (1 - np.power(b1, i))
    for g, p, m, u in zip(grads, params, ms, us):
        m = (b1 * m) + (1 - b1) * g
        u = np.fmax(np.ones_like(g, FTYPE) * b2 * u, np.absolute(g))
        p -= lr_t * m/(u+epsilon)
    return ms, us, params


def main(str_args):
    global word_vec_size, vocab_size, dsdict, sentences, gL
    global gW, gWs, gWb, gWsb, gV, gParams, default_seed, u_range
    global ggl_hist, gam_params, g_disable_tensor, gLog

    report('Starting at {}'.format(dt.datetime.now().strftime('%c')))
    try:
        args = docopt.docopt(__doc__, argv=str_args)
    except docopt.DocoptExit as docexit: # custom-modified version of docopt. I would not use docopt again.
        print(docexit)
        exit(1)

    def cint(arg):
        return int(arg)

    def cfloat(arg):
        return float(arg)

    def cstr(arg):
        return arg

    def cbool(arg):
        return eval(arg)

    def eval_arg(argstr, default=None, converter=cstr):
        arg = args[argstr]
        if arg is not None:
            return converter(arg)
        else:
            return default

    log_name = eval_arg('--log-name', default_log_name)
    config_name = eval_arg('--config-name', default_config_name)
    train_limit = eval_arg('--train-limit', default_train_limit, cint)
    valid_limit = eval_arg('--valid-limit', default_valid_limit, cint)
    test_limit = eval_arg('--test-limit', default_test_limit, cint)
    data_dir = eval_arg('--data-from', default_data_dir)
    default_seed = eval_arg('--seed', default_seed, cint)
    if args['--seed']:
        np.random.seed(default_seed)
    word_vec_size = eval_arg('--word-vector-size', default_word_vec_size, cint)
    max_epochs = eval_arg('--max-epochs', default_max_epochs, cint)
    learning_rate = eval_arg('--learning-rate', default_learning_rate, cfloat)
    wvlr = eval_arg('--wvlr', default_wvlr, cfloat)
    lambda_reg = eval_arg('--lambda', default_lambda, cfloat)
    batch_size = eval_arg('--batch-size', default_batch_size, cint)
    epsilon = eval_arg('--epsilon', default_epsilon, cfloat)
    report_interval = eval_arg('--report-interval', 1, cint)
    checkpoint_interval = eval_arg('--checkpoint-interval', None, cint)
    checkpoint_base = eval_arg('--checkpoint-base', 'rntn_checkpoint')
    checkpoint_dir = eval_arg('--checkpoint-dir', None)
    compress_checkpoints = eval_arg('--ckpt-compress', False, cbool)
    cost_threshold = eval_arg('--cost-threshold', default_cost_threshold, cfloat)
    u_range = eval_arg('--weight-initial-range', 0.0001, cfloat)
    use_adamax = eval_arg('--use-adamax', True, cbool)
    g_disable_tensor = eval_arg('--disable-tensor', False, cbool)
    valid_interval = eval_arg('--validate-interval', None, cint)
    print_test_results = eval_arg('--print-tests', True, cbool)
    check_trains = eval_arg('--check-training', False, cbool)

    gLog = lg.getLogger('rntn')
    lg.basicConfig(level=lg.DEBUG, format='%(message)s')
    fh = lg.FileHandler(log_name)
    fh.setLevel(lg.DEBUG)
    #ch = lg.StreamHandler()
    gLog.addHandler(fh)
    #gLog.addHandler(ch)
    (sentences, dsdict, trains, valids, tests) = \
        load_dataset(data_dir, train_limit=train_limit,
                     test_limit=test_limit, valid_limit=valid_limit)
    vocab_size = len(dsdict)
    report('Vocabulary size from {} is {} items.'.format(data_dir, vocab_size))
    if args['--params-from']:
        (gV, gW, gWs, gWb, gWsb, gL, ggl_hist, gam_params) = load_params(args['--params-from'])
        gParams = [gV, gW, gWs, gWb, gWsb, gL]
    else:
        # Tensor layer
        gV = weight_variable([word_vec_size, 2*word_vec_size, 2*word_vec_size])
        # Weights for forward activation
        gW = weight_variable([word_vec_size, 2*word_vec_size])
        # Weights for softmax
        gWs = weight_variable([n_labels, word_vec_size])
        # Bias for forward activation
        gWb = bias_variable([word_vec_size, 1])
        # Bias for softmax
        gWsb = bias_variable([n_labels, 1])
        # Word embedding vector
        gL = weight_variable([vocab_size, word_vec_size, 1])

        gParams = [gV, gW, gWs, gWb, gWsb, gL]
        # Adaptive optimization parameters
        ggl_hist = None
        gam_params = None

    attach_info_to_nodes(sentences, dsdict, gL)
    if args['train']:
        train(gParams, trains, valids,
              gl_hist=ggl_hist, am_params=gam_params, max_epochs=max_epochs,
              batch_size=batch_size, lr=learning_rate, lamda=lambda_reg,
              use_adamax=use_adamax, report_interval=report_interval,
              checkpoint_interval=checkpoint_interval, ckpt_compress=compress_checkpoints,
              cp_file_base=checkpoint_base, cost_threshold=cost_threshold,
              wvlr=wvlr,
              valid_interval=valid_interval, check_trains=check_trains,
              checkpoint_dir=checkpoint_dir,
              print_test_results=print_test_results)
        sys.exit(0)
    elif args['check-grad']:
        check_gradients([sentences[0]], gParams, epsilon, lambda_reg)
        sys.exit(0)
    elif args['accuracy']:
        if args['training']:
            ss = trains
            set_name = 'Training'
        elif args['validation']:
            ss = valids
            set_name = 'Validation'
        elif args['test']:
            ss = tests
            set_name = 'Test'
        else:
            ss = sentences
            set_name = 'All'
        check_set_accuracy(ss, gParams, print_results=print_test_results, set_name=set_name)
        sys.exit(0)


def check_set_accuracy(ss, params, print_results=True, set_name=None):
    (V, W, Ws, Wb, Wsb, L) = params
    # if g_disable_tensor:
    #     V = np.zeros_like(V)
    vwords = pos_neg_words(get_words(ss))
    pos_negs = pos_neg(ss)
    accuracy = pos_neg_accuracy(pos_negs, params)
    words_correct = sum(
            1 if np.argmax(forward_predict(Ws, Wsb, L[word.phrase_id])) ==
                 np.argmax(word.sentiment)
            else 0 for word in vwords)
    word_acccuracy = words_correct / len(vwords)
    if print_results:
        if set_name:
            print(set_name+': ', end='')
        print('positive/negative accuracy: word: {:.2%}, sentence: {:.2%}'.format(
                word_acccuracy, accuracy))
    return accuracy, word_acccuracy


def init_for_debug(data_dir):
    global word_vec_size, vocab_size, dsdict, sentences, trains, valids, tests
    global gW, gWs, gWb, gWsb, gV, gL, gParams, default_seed, u_range
    global ggl_hist, gam_params

    np.random.seed(default_seed)
    word_vec_size = default_word_vec_size
    (sentences, dsdict, trains, valids, tests) = load_dataset(data_dir)
    vocab_size = len(dsdict)
    # Tensor layer
    gV = weight_variable([word_vec_size, 2*word_vec_size, 2*word_vec_size])
    # Weights for forward activation
    gW = weight_variable([word_vec_size, 2*word_vec_size])
    # Weights for softmax
    gWs = weight_variable([n_labels, word_vec_size])
    # Bias for forward activation
    gWb = bias_variable([word_vec_size, 1])
    # Bias for softmax
    gWsb = bias_variable([n_labels, 1])
    # Word embedding vector
    gL = weight_variable([vocab_size, word_vec_size, 1])

    gParams = [gV, gW, gWs, gWb, gWsb, gL]
    attach_info_to_nodes(sentences, dsdict, gL)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
