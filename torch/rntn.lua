require 'torch'
require 'cutorch'
require 'optim '
require 'nn'
require 'zipmap'
require 'phrase_tree'
require 'rntn_dictionary'

torch.manualSeed(1)

torch.setdefaulttensortype('torch.FloatTensor')

options = {
    default_log_name = 'rntn.log',
    default_config_name = 'rntn.ini',
    default_learning_rate = 0.002,
    default_wvlr = 0.002,
    default_lambda = 1.0,
    u_range = 0.0001,
    default_max_epochs = 2000,
    default_batch_size = 1,
    default_epsilon = 1e-7,
    default_seed = 1,
    default_cost_threshold = nil,
    wvs = 10, -- make this number small for easier debugging
    n_labels = 5,
    --default_data_dir = 'sstdata'
    default_data_dir = 'toydata',

    default_train_limit = nil,
    default_test_limit = nil,
    default_valid_limit = nil,

    g_disable_tensor = false

}

Dict0 = {}
Dict0.mt = {}
function Dict0.new(o)
    setmetatable(o, Dict0.mt)
    return o
end
Dict0.__newindex = function(table, key, update)
    rawset(table, key, value)
    return value
end

ONE = torch.Tensor(1, 1):fill(1)

function tx(mat)
    torch.transpose(mat, 1, 2)
end

function predict_cost(logit, target)
    local _, target_index = torch.max(target, 1)
    return 1 - nn.LogSoftMax(logit[target_index])
end

function forward_tensor(ab, V)
    local abx = ab:expand(2*options.wvs, options.wvs)
    local pr1 = torch.bmm(abx:tx(), V)
    torch.bmm(pr1, abx)
end

function forward_prop(a, b, V, W, Wb)
    local ab = torch.cat(a, b, 1)
    local ab1 = torch.cat(a, b, ONE)
    local W_biased = torch.cat(W, Wb)
    local h = forward_tensor(ab, V)
    local std_forward = torch.mm(W_biased, ab1)
    return h + std_forward
end

function forward_logits(Ws, Wsb, a)
    local Ws_biased = torch.cat(Ws, Wsb)
    torch.mm(Ws_biased, torch.cat(a, ONE, 1))
end

function forward_predicts(Ws, Wsb, a)
    forward_logits(Ws, Wsb, a)
end

function forward_prop_sentence(s, V, W, Ws, Wb, Wsb, L)
    local outputs = {}
    local predictions = {}
    local targets = {}
    local costs = {}
    local activations = {}
    local word_vecs = {}
    for node in s do
        if node.is_leaf then
            local activation = L[node.phrase_id]
        else
            local activation = forward_prop(
                outputs[node.left.idx],
                outputs[node.right.idx],
                V, W, Wb)
        end
        local output = nn.Tanh(activation)
        local target = node.sentiment
        local prediction = forward_softmax(Ws, Wsb, output)
        local cost = predict_cost()
    end
    return predictions, targets, outputs, word_vecs
end

function regularization(lambda, m, params, word_vecs)
    local norms = torch.norm(params)^2
    local l_norms = torch.sum(torch.square(torch.norm(word_vecs)))
    return lambda / (2*m) * (norms + l_norms)
end

function aggregate_cost(targets, predictions, lambda, m, params, word_vecs)
    local err_components = 0
    for y, t in targets, predictions do
        err_components = err_components + torch.sum(predict_cost(y, y))
    end
    return err_components + regularization(lambda, m, params, word_vecs)
end

function delta_sm_i(prediction, target, Ws, node_out)
    local err_predict = torch.mm(Ws:tx(), prediction - target)
    local f_prime_node = 1 - node_out^2
    local delta_sm = err_predict * f_prime_node
    return delta_sm
end

function delta_down_i(W, delta_com, V, inputs)
    local mm = torch.mm
    local grad_inputs = 1.0 - inputs^2
    local delta_down_w = torch.mm(W:tx(), delta_com)
    local VVT = V + V:transpose(1, 3, 2)
    local pr1 = (delta_com:tx() * mm(VVT, inputs)):tx(1,2)
    local S = torch.sum(pr1)
    return (delta_down_w + S) * grad_inputs
end

function backprop_sentence(s, dwords, V, W, Ws, Wb, Wsb, predictions, targets, outputs)
    local wvs = options.wvs
    local delta_down = {}
    local delta_v = torch.zeros(V:size())
    local delta_w = torch.zeros(W:size())
    local delta_ws = torch.zeros(Ws:size())
    local delta_wb = torch.zeros(Wb:size())
    local delta_wsb = torch.zeros(Wsb:size())
    for i = 1, s.length do
        local idx = s.length - i
        local prediction = predictions[idx]
        local target = targets[idx]
        local output = outputs[idx]
        local delta_sm = delta_sm_i(prediction, target, Ws, output)
        delta_ws[idx] = torch.add(
            delta_ws[idx],
            torch.mm(prediction - target, output:transpose(1, 2)))
        delta_wsb[idx] = prediction - target
        if node.is_root then
            local delta_complete = delta_sm
        else
            local delta_complete = delta_sm + delta_down[node.idx]
        end
        if node.is_leaf then
            dwords[node.phrase_id] = node.phrase_id + delta_complete
        else
            local inp = torch.cat(outputs[node.left.idx], outputs[node.right.idx])
            local dd = delta_down_i(W, delta_complete, V, inp)
            delta_down[node.left.idx] = dd[{{1,3},{1,wvs}}]
            delta_down[node.right.idx] = dd[{{1,wvs+1},{1,2*wvs}}]
            delta_w =
        end
    end
end

function fwd_back_batch(params, L, dwords, grad_words, examples, lambda)
    local deltas = {}
    local m = 0
    for s in examples do
        m = m + s.length
    end
    local batch_scale = 1/m
    local reg_scale = lambda/m
    local predictions = {}
    local targets = {}
    local grads = {}
    for s in examples do
        local s_preds, s_targets, outs, s_words = forward_prop_sentence(s, params, L)
        predictions = predictions + s_preds
        targets = targets + s_targets
        deltas = backprop_sentence(s, dwords, params, s_preds, s_targets, outs)
        for j in 1, deltas.length do
            d_batch[j] = d_batch[j] + deltas[j]
        end
    end
    for k, v in pairs(dwords) do
        grad_words[k] = grad_words[k] + dwords[k] * batch_scale + L[k] * reg_scale
    end
    for dth in zip(params, d_batch) do
        local d_theta = dth.param
        local theta = dth.d_b
        grads = grads + d_theta * batch_scale + theta * reg_scale
    end

end

adamax_config = {
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8
}
