import tensorflow as tf
from tensorflow import Tensor as ts
from tensorflow.python.ops import rnn
import my_rnn



eps = 1e-6
def cosine_distance(y1,y2):
    # y1 [....,a, 1, d]
    # y2 [....,1, b, d]
#     cosine_numerator = T.sum(y1*y2, axis=-1)
    cosine_numerator = tf.reduce_sum(tf.multiply(y1, y2), axis=-1)
#     y1_norm = T.sqrt(T.maximum(T.sum(T.sqr(y1), axis=-1), eps)) #be careful while using T.sqrt(), like in the cases of Euclidean distance, cosine similarity, for the gradient of T.sqrt() at 0 is undefined, we should add an Eps or use T.maximum(original, eps) in the sqrt.
    y1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1), axis=-1), eps)) 
    y2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y2), axis=-1), eps)) 
    return cosine_numerator / y1_norm / y2_norm

def cal_relevancy_matrix(in_premise_repres, in_hypothesis_repres):
    in_premise_repres_tmp = tf.expand_dims(in_premise_repres, 1) # [batch_size, 1, premise_len, dim]
    in_hypothesis_repres_tmp = tf.expand_dims(in_hypothesis_repres, 2) # [batch_size, hypothesis_len, 1, dim]
    relevancy_matrix = cosine_distance(in_premise_repres_tmp,in_hypothesis_repres_tmp) # [batch_size, hypothesis_len, premise_len]
    return relevancy_matrix

def mask_relevancy_matrix(relevancy_matrix, premise_mask, hypothesis_mask):
    # relevancy_matrix: [batch_size, hypothesis_len, premise_len]
    # premise_mask: [batch_size, premise_len]
    # hypothesis_mask: [batch_size, passsage_len]
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(premise_mask, 1))
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(hypothesis_mask, 2))
    return relevancy_matrix

def cal_cosine_weighted_premise_representation(premise_representation, cosine_matrix, normalize=False):
    # premise_representation: [batch_size, premise_len, dim]
    # cosine_matrix: [batch_size, hypothesis_len, premise_len]
    if normalize: cosine_matrix = tf.nn.softmax(cosine_matrix)
    expanded_cosine_matrix = tf.expand_dims(cosine_matrix, axis=-1) # [batch_size, hypothesis_len, premise_len, 'x']
    weighted_premise_words = tf.expand_dims(premise_representation, axis=1) # [batch_size, 'x', premise_len, dim]
    weighted_premise_words = tf.reduce_sum(tf.multiply(weighted_premise_words, expanded_cosine_matrix), axis=2)# [batch_size, hypothesis_len, dim]
    if not normalize:
        weighted_premise_words = tf.div(weighted_premise_words, tf.expand_dims(tf.add(tf.reduce_sum(cosine_matrix, axis=-1),eps),axis=-1))
    return weighted_premise_words # [batch_size, hypothesis_len, dim]


def cal_cosine_weighted_image_representation(premise_representation, cosine_matrix, normalize=False):
    # premise_representation: [batch_size, premise_len, dim]
    # cosine_matrix: [batch_size, hypothesis_len, premise_len]
    if normalize: cosine_matrix = tf.nn.softmax(cosine_matrix)
    expanded_cosine_matrix = tf.expand_dims(cosine_matrix, axis=-1) # [batch_size, hypothesis_len, premise_len, 'x']
    weighted_premise_words = tf.expand_dims(premise_representation, axis=1) # [batch_size, 'x', premise_len, dim]
    weighted_premise_words = tf.reduce_sum(tf.multiply(weighted_premise_words, expanded_cosine_matrix), axis=2)# [batch_size, hypothesis_len, dim]
    if not normalize:
        weighted_premise_words = tf.div(weighted_premise_words, tf.expand_dims(tf.add(tf.reduce_sum(cosine_matrix, axis=-1),eps),axis=-1))
    return weighted_premise_words # [batch_size, hypothesis_len, dim]



def multi_perspective_expand_for_3D(in_tensor, decompose_params):
    in_tensor = tf.expand_dims(in_tensor, axis=2) #[batch_size, hypothesis_len, 'x', dim]
    decompose_params = tf.expand_dims(tf.expand_dims(decompose_params, axis=0), axis=0) # [1, 1, decompse_dim, dim]
    return tf.multiply(in_tensor, decompose_params)#[batch_size, hypothesis_len, decompse_dim, dim]

def multi_perspective_expand_for_2D(in_tensor, decompose_params):
    in_tensor = tf.expand_dims(in_tensor, axis=1) #[batch_size, 'x', dim]
    decompose_params = tf.expand_dims(decompose_params, axis=0) # [1, decompse_dim, dim]
    return tf.multiply(in_tensor, decompose_params) # [batch_size, decompse_dim, dim]

def multi_perspective_expand_for_1D(in_tensor, decompose_params):
    in_tensor = tf.expand_dims(in_tensor, axis=0) #['x', dim]
    return tf.multiply(in_tensor, decompose_params) # [decompse_dim, dim]

def cal_full_matching(hypothesis_representation, full_premise_representation, decompose_params):
    # hypothesis_representation: [batch_size, hypothesis_len, dim]
    # full_premise_representation: [batch_size, dim]
    # decompose_params: [decompose_dim, dim]
    def singel_instance(x):
        p = x[0]
        q = x[1]
        # p: [pasasge_len, dim], q: [dim]
        p = multi_perspective_expand_for_2D(p, decompose_params) # [pasasge_len, decompose_dim, dim]
        q = multi_perspective_expand_for_1D(q, decompose_params) # [decompose_dim, dim]
        q = tf.expand_dims(q, 0) # [1, decompose_dim, dim]
        return cosine_distance(p, q) # [hypothesis_len, decompose]
    elems = (hypothesis_representation, full_premise_representation)
    return tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, hypothesis_len, decompse_dim]
   

# match each time step against full image representation
# for now full_img_repres = average(all_features)
def cal_full_matching_img(hypothesis_representation, full_img_representation, w_decompose_params, img_decompose_params):
    # hypothesis_representation: [batch_size, hypothesis_len, dim]
    # full_img_representation: [batch_size, full_img_dim]
    # decompose_params: [decompose_dim, dim]
    # img_decompose_params: [decompose_dim, img_dim] 
    def singel_instance(x):
        p = x[0]
        q = x[1]
        # p: [pasasge_len, dim], q: [dim]
        p = multi_perspective_expand_for_2D(p, w_decompose_params) # [pasasge_len, decompose_dim, dim]
        q = multi_perspective_expand_for_1D(q, img_decompose_params) # [decompose_dim, dim]
        q = tf.expand_dims(q, 0) # [1, decompose_dim, dim]
        return cosine_distance(p, q) # [hypothesis_len, decompose]
    elems = (hypothesis_representation, full_img_representation)
    return tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, hypothesis_len, decompse_dim]




#match each time step against every image features
def cal_maxpooling_matching_img(hypothesis_rep, img_features, w_decompose_params, img_decompose_params):
    # hypothesis_representation: [batch_size, hypothesis_len, dim]
    # img_features: [batch_size, features_num, img_dim]
    # w_decompose_params: [decompose_dim, dim]
    # img_decompose_params: [decompose_dim, img_dim]
    def singel_instance(x):
        p = x[0]
        q = x[1]
        # p: [pasasge_len, dim], q: [premise_len, dim]
        p = multi_perspective_expand_for_2D(p, w_decompose_params) # [pasasge_len, decompose_dim, dim]
        q = multi_perspective_expand_for_2D(q, img_decompose_params) # [features_num, decompose_dim, dim]
        p = tf.expand_dims(p, 1) # [pasasge_len, 1, decompose_dim, dim]
        q = tf.expand_dims(q, 0) # [1, features_num, decompose_dim, dim]
        return cosine_distance(p, q) # [hypothesis_len, features_num, decompose]
    elems = (hypothesis_rep, img_features)
    matching_matrix = tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, hypothesis_len, features_num, decompse_dim]
    return tf.concat([tf.reduce_max(matching_matrix, axis=2), tf.reduce_mean(matching_matrix, axis=2)], 2)# [batch_size, hypothesis_len, 2*decompse_dim]


def cal_maxpooling_matching(hypothesis_rep, premise_rep, decompose_params):
    # hypothesis_representation: [batch_size, hypothesis_len, dim]
    # qusetion_representation: [batch_size, premise_len, dim]
    # decompose_params: [decompose_dim, dim]
    
    def singel_instance(x):
        p = x[0]
        q = x[1]
        # p: [pasasge_len, dim], q: [premise_len, dim]
        p = multi_perspective_expand_for_2D(p, decompose_params) # [pasasge_len, decompose_dim, dim]
        q = multi_perspective_expand_for_2D(q, decompose_params) # [premise_len, decompose_dim, dim]
        p = tf.expand_dims(p, 1) # [pasasge_len, 1, decompose_dim, dim]
        q = tf.expand_dims(q, 0) # [1, premise_len, decompose_dim, dim]
        return cosine_distance(p, q) # [hypothesis_len, premise_len, decompose]
    elems = (hypothesis_rep, premise_rep)
    matching_matrix = tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, hypothesis_len, premise_len, decompse_dim]
    return tf.concat([tf.reduce_max(matching_matrix, axis=2), tf.reduce_mean(matching_matrix, axis=2)], 2)# [batch_size, hypothesis_len, 2*decompse_dim]

def cal_maxpooling_matching_for_word(hypothesis_rep, premise_rep, decompose_params):
    # hypothesis_representation: [batch_size, hypothesis_len, dim]
    # qusetion_representation: [batch_size, premise_len, dim]
    # decompose_params: [decompose_dim, dim]
    
    def singel_instance(x):
        p = x[0]
        q = x[1]
        q = multi_perspective_expand_for_2D(q, decompose_params) # [premise_len, decompose_dim, dim]
        # p: [pasasge_len, dim], q: [premise_len, dim]
        def single_instance_2(y):
            # y: [dim]
            y = multi_perspective_expand_for_1D(y, decompose_params) #[decompose_dim, dim]
            y = tf.expand_dims(y, 0) # [1, decompose_dim, dim]
            matching_matrix = cosine_distance(y, q)#[premise_len, decompose_dim]
            return tf.concat([tf.reduce_max(matching_matrix, axis=0), tf.reduce_mean(matching_matrix, axis=0)], 0) #[2*decompose_dim]
        return tf.map_fn(single_instance_2, p, dtype=tf.float32) # [hypothesis_len, 2*decompse_dim]
    elems = (hypothesis_rep, premise_rep)
    return tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, hypothesis_len, 2*decompse_dim]


def cal_attentive_matching(hypothesis_rep, att_premise_rep, decompose_params):
    # hypothesis_rep: [batch_size, hypothesis_len, dim]
    # att_premise_rep: [batch_size, hypothesis_len, dim]
    def singel_instance(x):
        p = x[0]
        q = x[1]
        # p: [pasasge_len, dim], q: [pasasge_len, dim]
        p = multi_perspective_expand_for_2D(p, decompose_params) # [pasasge_len, decompose_dim, dim]
        q = multi_perspective_expand_for_2D(q, decompose_params) # [pasasge_len, decompose_dim, dim]
        return cosine_distance(p, q) # [pasasge_len, decompose_dim]

    elems = (hypothesis_rep, att_premise_rep)
    return tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, hypothesis_len, decompse_dim]

def cal_attentive_matching_img(hypothesis_rep, att_image_rep, w_decompose_params, i_decompose_params):
    # hypothesis_rep: [batch_size, hypothesis_len, dim]
    # att_image_rep: [batch_size, hypothesis_len, dim]
    def singel_instance(x):
        p = x[0]
        q = x[1]
        # p: [pasasge_len, dim], q: [pasasge_len, dim]
        p = multi_perspective_expand_for_2D(p, w_decompose_params) # [pasasge_len, decompose_dim, dim]
        q = multi_perspective_expand_for_2D(q, i_decompose_params) # [pasasge_len, decompose_dim, dim]
        return cosine_distance(p, q) # [pasasge_len, decompose_dim]

    elems = (hypothesis_rep, att_image_rep)
    return tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, hypothesis_len, decompse_dim]
            #if with_max_attentive_match:
                # forward max attentive-matching
            #    max_att_fw = cal_max_premise_representation(premise_context_representation_fw, forward_relevancy_matrix)
            #    fw_max_att_decomp_params = tf.get_variable("fw_max_att_decomp_params", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
            #    fw_max_attentive_rep = cal_attentive_matching(hypothesis_context_representation_fw, max_att_fw, fw_max_att_decomp_params)
            #    all_hypthesis_matching_representations.append(fw_max_attentive_rep)
            #    dim += MP_dim

                # backward max attentive-matching
            #    max_att_bw = cal_max_premise_representation(premise_context_representation_bw, backward_relevancy_matrix)
            #    bw_max_att_decomp_params = tf.get_variable("bw_max_att_decomp_params", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
            #    bw_max_attentive_rep = cal_attentive_matching(hypothesis_context_representation_bw, max_att_bw, bw_max_att_decomp_params)
            #    all_hypthesis_matching_representations.append(bw_max_attentive_rep)
            #    dim += MP_dim


def cross_entropy(logits, truth, mask):
    # logits: [batch_size, hypothesis_len]
    # truth: [batch_size, hypothesis_len]
    # mask: [batch_size, hypothesis_len]

#     xdev = x - x.max()
#     return xdev - T.log(T.sum(T.exp(xdev)))
    logits = tf.multiply(logits, mask)
    xdev = tf.subtract(logits, tf.expand_dims(tf.reduce_max(logits, 1), -1))
    log_predictions = tf.subtract(xdev, tf.expand_dims(tf.log(tf.reduce_sum(tf.exp(xdev),-1)),-1))
#     return -T.sum(targets * log_predictions)
    result = tf.multiply(tf.multiply(truth, log_predictions), mask) # [batch_size, hypothesis_len]
    return tf.multiply(-1.0,tf.reduce_sum(result, -1)) # [batch_size]
    
def highway_layer(in_val, output_size, scope=None):
    # in_val: [batch_size, hypothesis_len, dim]
    input_shape = tf.shape(in_val)
    batch_size = input_shape[0]
    hypothesis_len = input_shape[1]
#     feat_dim = input_shape[2]
    in_val = tf.reshape(in_val, [batch_size * hypothesis_len, output_size])
    with tf.variable_scope(scope or "highway_layer"):
        highway_w = tf.get_variable("highway_w", [output_size, output_size], dtype=tf.float32)
        highway_b = tf.get_variable("highway_b", [output_size], dtype=tf.float32)
        full_w = tf.get_variable("full_w", [output_size, output_size], dtype=tf.float32)
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        trans = tf.nn.tanh(tf.nn.xw_plus_b(in_val, full_w, full_b))
        gate = tf.nn.sigmoid(tf.nn.xw_plus_b(in_val, highway_w, highway_b))
        outputs = tf.add(tf.multiply(trans, gate), tf.multiply(in_val, tf.subtract(1.0, gate)), "y")
    outputs = tf.reshape(outputs, [batch_size, hypothesis_len, output_size])
    return outputs

def multi_highway_layer(in_val, output_size, num_layers, scope=None):
    scope_name = 'highway_layer'
    if scope is not None: scope_name = scope
    for i in xrange(num_layers):
        cur_scope_name = scope_name + "-{}".format(i)
        in_val = highway_layer(in_val, output_size, scope=cur_scope_name)
    return in_val

def cal_max_premise_representation(premise_representation, cosine_matrix):
    # premise_representation: [batch_size, premise_len, dim]
    # cosine_matrix: [batch_size, hypothesis_len, premise_len]
    premise_index = tf.arg_max(cosine_matrix, 2) # [batch_size, hypothesis_len]
    def singel_instance(x):
        q = x[0]
        c = x[1]
        return tf.gather(q, c)
    elems = (premise_representation, premise_index)
    return tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, hypothesis_len, dim]

def cal_linear_decomposition_representation(hypothesis_representation, hypothesis_lengths, cosine_matrix,is_training, 
                                            lex_decompsition_dim, dropout_rate):
    # hypothesis_representation: [batch_size, hypothesis_len, dim]
    # cosine_matrix: [batch_size, hypothesis_len, premise_len]
    hypothesis_similarity = tf.reduce_max(cosine_matrix, 2)# [batch_size, hypothesis_len]
    similar_weights = tf.expand_dims(hypothesis_similarity, -1) # [batch_size, hypothesis_len, 1]
    dissimilar_weights = tf.subtract(1.0, similar_weights)
    similar_component = tf.multiply(hypothesis_representation, similar_weights)
    dissimilar_component = tf.multiply(hypothesis_representation, dissimilar_weights)
    all_component = tf.concat([similar_component, dissimilar_component], 2)
    if lex_decompsition_dim==-1:
        return all_component
    with tf.variable_scope('lex_decomposition'):
        lex_lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(lex_decompsition_dim)
        lex_lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(lex_decompsition_dim)
        if is_training:
            lex_lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(lex_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
            lex_lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(lex_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
        lex_lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([lex_lstm_cell_fw])
        lex_lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([lex_lstm_cell_bw])

        (lex_features_fw, lex_features_bw), _ = rnn.bidirectional_dynamic_rnn(
                    lex_lstm_cell_fw, lex_lstm_cell_bw, all_component, dtype=tf.float32, sequence_length=hypothesis_lengths)

        lex_features = tf.concat([lex_features_fw, lex_features_bw], 2)
    return lex_features



## image_features bw and fw are the same. This way we can switch between image and text
def match_hypothesis_with_image(image_features_fw, image_features_bw, full_img_rep_fw, full_img_rep_bw, image_mask, 
                                hypothesis_context_representation_fw, hypothesis_context_representation_bw, hypothesis_mask,
                                MP_dim, context_lstm_dim, scope=None, img_dim=100,
                                with_img_full_match=True, with_img_maxpool_match=True, with_img_attentive_match=True,
                                with_img_max_attentive_match=True):

    all_image_aware_representatins = []
    dim = 0
    with tf.variable_scope(scope or "match_hypothesis_with_image"):

        hypothesis_context_representation_fw = tf.multiply(hypothesis_context_representation_fw, tf.expand_dims(hypothesis_mask,-1))
        hypothesis_context_representation_bw = tf.multiply(hypothesis_context_representation_bw, tf.expand_dims(hypothesis_mask,-1))


        img_forward_relevancy_matrix = cal_relevancy_matrix(image_features_fw, hypothesis_context_representation_fw)
        img_forward_relevancy_matrix = mask_relevancy_matrix(img_forward_relevancy_matrix, image_mask, hypothesis_mask)

        img_backward_relevancy_matrix = cal_relevancy_matrix(image_features_bw, hypothesis_context_representation_bw)
        img_backward_relevancy_matrix = mask_relevancy_matrix(img_backward_relevancy_matrix, image_mask, hypothesis_mask)
        

        if MP_dim > 0:
            if with_img_full_match:
                    #def cal_full_matching_img(hypothesis_representation, full_img_representation, w_decompose_params, img_decompose_params):
                    # forward Full-Matching: hypothesis_context_representation_fw vs premise_context_representation_fw[-1]
                    fw_w_decomp_params = tf.get_variable("forward_word_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                    fw_full_img_decomp_params = tf.get_variable("forward_full_img_matching_decomp", shape=[MP_dim, img_dim], dtype=tf.float32)
                    fw_full_img_match_rep = cal_full_matching_img(hypothesis_context_representation_fw, full_img_rep_fw, fw_w_decomp_params, fw_full_img_decomp_params) #[batch_size, hypothesis_len, decompse_dim]
                    all_image_aware_representatins.append(fw_full_img_match_rep)
                    dim += MP_dim

                    # backward Full-Matching: hypothesis_context_representation_bw vs premise_context_representation_bw[0]
                    bw_w_decomp_params = tf.get_variable("backward_word_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                    bw_full_img_decomp_params = tf.get_variable("backward_full_img_matching_decomp", shape=[MP_dim, img_dim], dtype=tf.float32)
                    bw_full_img_match_rep = cal_full_matching_img(hypothesis_context_representation_bw, full_img_rep_bw, bw_w_decomp_params, bw_full_img_decomp_params) #[batch_size, hypothesis_len, decompse_dim]
                    all_image_aware_representatins.append(bw_full_img_match_rep)
                    dim += MP_dim

            if with_img_maxpool_match:
                    #def cal_maxpooling_matching_img(hypothesis_rep, img_features, w_decompose_params, img_decompose_params):
                    # forward Maxpooling-Matching
                    fw_w_decomp_params = tf.get_variable("forward_word_max_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                    fw_max_img_decomp_params = tf.get_variable("forward_max_img_matching_decomp", shape=[MP_dim, img_dim], dtype=tf.float32)
                    fw_maxpooling_rep = cal_maxpooling_matching_img(hypothesis_context_representation_fw, image_features_fw, fw_w_decomp_params, fw_max_img_decomp_params)
                    all_image_aware_representatins.append(fw_maxpooling_rep)
                    dim += 2*MP_dim

                    # backward Maxpooling-Matching
                    bw_w_decomp_params = tf.get_variable("backward_word_max_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                    bw_max_img_decomp_params = tf.get_variable("backward_max_img_matching_decomp", shape=[MP_dim, img_dim], dtype=tf.float32)
                    bw_maxpooling_rep = cal_maxpooling_matching_img(hypothesis_context_representation_bw, image_features_bw, bw_w_decomp_params, bw_max_img_decomp_params)
                    all_image_aware_representatins.append(bw_maxpooling_rep)
                    dim += 2*MP_dim
            if with_img_attentive_match:
                # forward attentive-matching
                # forward weighted premise representation: [batch_size, premise_len, hypothesis_len] [batch_size, premise_len, context_lstm_dim]
                att_image_fw_contexts = cal_cosine_weighted_premise_representation(image_features_fw, img_forward_relevancy_matrix)
                fw_w_attentive_decomp_params = tf.get_variable("forward_word_attentive_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                fw_i_attentive_decomp_params = tf.get_variable("forward_img_attentive_matching_decomp", shape=[MP_dim, img_dim], dtype=tf.float32)
                fw_attentive_rep = cal_attentive_matching_img(hypothesis_context_representation_fw, att_image_fw_contexts, fw_w_attentive_decomp_params, fw_i_attentive_decomp_params)
                all_image_aware_representatins.append(fw_attentive_rep)
                dim += MP_dim

                # backward attentive-matching
                # backward weighted premise representation
                att_image_bw_contexts = cal_cosine_weighted_premise_representation(image_features_bw, img_backward_relevancy_matrix)
                bw_w_attentive_decomp_params = tf.get_variable("backward_word_attentive_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                bw_i_attentive_decomp_params = tf.get_variable("backward_img_attentive_matching_decomp", shape=[MP_dim, img_dim], dtype=tf.float32)
                bw_attentive_rep = cal_attentive_matching_img(hypothesis_context_representation_bw, att_image_bw_contexts, bw_w_attentive_decomp_params, bw_i_attentive_decomp_params)
                all_image_aware_representatins.append(bw_attentive_rep)
                dim += MP_dim

            if True:
                # forward max attentive-matching
                max_att_fw = cal_max_premise_representation(image_features_fw, img_forward_relevancy_matrix)
                fw_w_max_att_decomp_params = tf.get_variable("fw_word_max_att_decomp_params", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                fw_i_max_att_decomp_params = tf.get_variable("fw_img_max_att_decomp_params", shape=[MP_dim, img_dim], dtype=tf.float32) 
                fw_max_attentive_rep = cal_attentive_matching_img(hypothesis_context_representation_fw, max_att_fw, fw_w_max_att_decomp_params, fw_i_max_att_decomp_params)
                all_image_aware_representatins.append(fw_max_attentive_rep)
                dim += MP_dim

                # backward max attentive-matching
                max_att_bw = cal_max_premise_representation(image_features_bw, img_backward_relevancy_matrix)
                bw_w_max_att_decomp_params = tf.get_variable("bw_word_max_att_decomp_params", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                bw_i_max_att_decomp_params = tf.get_variable("bw_img_max_att_decomp_params", shape=[MP_dim, img_dim], dtype=tf.float32)
                bw_max_attentive_rep = cal_attentive_matching_img(hypothesis_context_representation_bw, max_att_bw, bw_w_max_att_decomp_params, bw_i_max_att_decomp_params)
                all_image_aware_representatins.append(bw_max_attentive_rep)
                dim += MP_dim




        all_image_aware_representatins.append(tf.reduce_max(img_forward_relevancy_matrix, axis=2,keep_dims=True))
        all_image_aware_representatins.append(tf.reduce_mean(img_forward_relevancy_matrix, axis=2,keep_dims=True))
        all_image_aware_representatins.append(tf.reduce_max(img_backward_relevancy_matrix, axis=2,keep_dims=True))
        all_image_aware_representatins.append(tf.reduce_mean(img_backward_relevancy_matrix, axis=2,keep_dims=True))
        dim += 4
    return (all_image_aware_representatins, dim)





def match_hypothesis_with_premise(hypothesis_context_representation_fw, hypothesis_context_representation_bw, mask,
                                premise_context_representation_fw, premise_context_representation_bw,premise_mask,
                                MP_dim, context_lstm_dim, scope=None,
                                with_full_match=True, with_maxpool_match=True, with_attentive_match=True, with_max_attentive_match=True):

    all_hypthesis_matching_representations = []
    dim = 0
    with tf.variable_scope(scope or "match_hypothesis_with_premise"):
        fw_premise_full_rep = premise_context_representation_fw[:,-1,:]
        bw_premise_full_rep = premise_context_representation_bw[:,0,:]

        premise_context_representation_fw = tf.multiply(premise_context_representation_fw, tf.expand_dims(premise_mask,-1))
        premise_context_representation_bw = tf.multiply(premise_context_representation_bw, tf.expand_dims(premise_mask,-1))
        hypothesis_context_representation_fw = tf.multiply(hypothesis_context_representation_fw, tf.expand_dims(mask,-1))
        hypothesis_context_representation_bw = tf.multiply(hypothesis_context_representation_bw, tf.expand_dims(mask,-1))

        forward_relevancy_matrix = cal_relevancy_matrix(premise_context_representation_fw, hypothesis_context_representation_fw)
        forward_relevancy_matrix = mask_relevancy_matrix(forward_relevancy_matrix, premise_mask, mask)

        backward_relevancy_matrix = cal_relevancy_matrix(premise_context_representation_bw, hypothesis_context_representation_bw)
        backward_relevancy_matrix = mask_relevancy_matrix(backward_relevancy_matrix, premise_mask, mask)

        
        if MP_dim > 0:
            if with_full_match:
                # forward Full-Matching: hypothesis_context_representation_fw vs premise_context_representation_fw[-1]
                fw_full_decomp_params = tf.get_variable("forward_full_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                fw_full_match_rep = cal_full_matching(hypothesis_context_representation_fw, fw_premise_full_rep, fw_full_decomp_params) #[batch_size, hypothesis_len, decompse_dim])
                all_hypthesis_matching_representations.append(fw_full_match_rep)
                dim += MP_dim

                # backward Full-Matching: hypothesis_context_representation_bw vs premise_context_representation_bw[0]
                bw_full_decomp_params = tf.get_variable("backward_full_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                bw_full_match_rep = cal_full_matching(hypothesis_context_representation_bw, bw_premise_full_rep, bw_full_decomp_params)
                all_hypthesis_matching_representations.append(bw_full_match_rep)
                dim += MP_dim


            if with_maxpool_match:
                # forward Maxpooling-Matching
                fw_maxpooling_decomp_params = tf.get_variable("forward_maxpooling_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                fw_maxpooling_rep = cal_maxpooling_matching(hypothesis_context_representation_fw, premise_context_representation_fw, fw_maxpooling_decomp_params)
                all_hypthesis_matching_representations.append(fw_maxpooling_rep)
                dim += 2*MP_dim
                # backward Maxpooling-Matching
                bw_maxpooling_decomp_params = tf.get_variable("backward_maxpooling_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                bw_maxpooling_rep = cal_maxpooling_matching(hypothesis_context_representation_bw, premise_context_representation_bw, bw_maxpooling_decomp_params)
                all_hypthesis_matching_representations.append(bw_maxpooling_rep)
                dim += 2*MP_dim
            
            if with_attentive_match:
                # forward attentive-matching
                # forward weighted premise representation: [batch_size, premise_len, hypothesis_len] [batch_size, premise_len, context_lstm_dim]
                att_premise_fw_contexts = cal_cosine_weighted_premise_representation(premise_context_representation_fw, forward_relevancy_matrix)
                fw_attentive_decomp_params = tf.get_variable("forward_attentive_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                fw_attentive_rep = cal_attentive_matching(hypothesis_context_representation_fw, att_premise_fw_contexts, fw_attentive_decomp_params)
                all_hypthesis_matching_representations.append(fw_attentive_rep)
                dim += MP_dim

                # backward attentive-matching
                # backward weighted premise representation
                att_premise_bw_contexts = cal_cosine_weighted_premise_representation(premise_context_representation_bw, backward_relevancy_matrix)
                bw_attentive_decomp_params = tf.get_variable("backward_attentive_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                bw_attentive_rep = cal_attentive_matching(hypothesis_context_representation_bw, att_premise_bw_contexts, bw_attentive_decomp_params)
                all_hypthesis_matching_representations.append(bw_attentive_rep)
                dim += MP_dim
            
            if with_max_attentive_match:
                # forward max attentive-matching
                max_att_fw = cal_max_premise_representation(premise_context_representation_fw, forward_relevancy_matrix)
                fw_max_att_decomp_params = tf.get_variable("fw_max_att_decomp_params", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                fw_max_attentive_rep = cal_attentive_matching(hypothesis_context_representation_fw, max_att_fw, fw_max_att_decomp_params)
                all_hypthesis_matching_representations.append(fw_max_attentive_rep)
                dim += MP_dim

                # backward max attentive-matching
                max_att_bw = cal_max_premise_representation(premise_context_representation_bw, backward_relevancy_matrix)
                bw_max_att_decomp_params = tf.get_variable("bw_max_att_decomp_params", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                bw_max_attentive_rep = cal_attentive_matching(hypothesis_context_representation_bw, max_att_bw, bw_max_att_decomp_params)
                all_hypthesis_matching_representations.append(bw_max_attentive_rep)
                dim += MP_dim


        all_hypthesis_matching_representations.append(tf.reduce_max(forward_relevancy_matrix, axis=2,keep_dims=True))
        all_hypthesis_matching_representations.append(tf.reduce_mean(forward_relevancy_matrix, axis=2,keep_dims=True))
        all_hypthesis_matching_representations.append(tf.reduce_max(backward_relevancy_matrix, axis=2,keep_dims=True))
        all_hypthesis_matching_representations.append(tf.reduce_mean(backward_relevancy_matrix, axis=2,keep_dims=True))
        dim += 4
    return (all_hypthesis_matching_representations, dim)
        

def gather_along_second_axis1(data, indices):
    '''
    data has shape: [batch_size, sentence_length, word_dim]
    indices is list of index we want to gather
    1. add -1 and -2 to sentence ==> [batch_size, sentence_length + 2, word_dim]
    2. increase each indices by 2
    3. gather according to indices
    '''
    #add -1 
    flat_indices = tf.tile(indices[None, :], [tf.shape(data)[0], 1])
    batch_offset = tf.range(0, tf.shape(data)[0]) * tf.shape(data)[1]
    flat_indices = tf.reshape(flat_indices + batch_offset[:, None], [-1])
    flat_data = tf.reshape(data, tf.concat([[-1], tf.shape(data)[2:]], 0))
    result_shape = tf.concat([[tf.shape(data)[0], -1], tf.shape(data)[2:]], 0)
    result = tf.reshape(tf.gather(flat_data, flat_indices), result_shape)
    shape = data.shape[:1].concatenate(indices.shape[:1])
    result.set_shape(shape.concatenate(data.shape[2:]))
    return result

def tile_repeat(n, repTime):
    '''
    create something like 111..122..2333..33 ..... n..nn 
    one particular number appears repTime consecutively 
    '''
    #print n, repTime
    idx = tf.range(n)
    idx = tf.reshape(idx, [-1, 1])    # Convert to a n x 1 matrix.
    idx = tf.tile(idx, [1, repTime])  # Create multiple columns, each column has one number repeats repTime 
    y = tf.reshape(idx, [-1])
    return y

def gather_along_second_axis(x, idx):
    ''' 
    x has shape: [batch_size, sentence_length, word_dim]
    idx has shape: [batch_size, num_indices]
    Basically, in each batch, get words from sentence having index specified in idx
    However, since tensorflow does not fully support indexing,
    gather only work for the first axis. We have to reshape the input data, gather then reshape again
    '''
    idx1= tf.reshape(idx, [-1]) # [batch_size*num_indices]
    idx_flattened = tile_repeat(tf.shape(idx)[0], tf.shape(idx)[1]) * tf.shape(x)[1] + idx1
    y = tf.gather(tf.reshape(x, [-1,tf.shape(x)[2]]),  # flatten input
                idx_flattened)
    y = tf.reshape(y, tf.shape(x)) 
    return y





def bilateral_match_func(image_features, in_premise_repres, in_hypothesis_repres, in_premise_dep_cons, in_hypothesis_dep_cons,
                        premise_lengths, hypothesis_lengths, premise_mask, hypothesis_mask, MP_dim, input_dim, 
                        with_filter_layer, context_layer_num, context_lstm_dim,is_training,dropout_rate,
                        with_match_highway,aggregation_layer_num, aggregation_lstm_dim,highway_layer_num,
                        with_aggregation_highway,with_lex_decomposition,lex_decompsition_dim,
                        with_full_match=True, with_maxpool_match=True, with_attentive_match=True, with_max_attentive_match=True,
                        with_left_match=True, with_right_match=True, with_mean_aggregation=True, with_dep=False, with_image=False, 
                        image_with_hypothesis_only=False, with_img_full_match=False, with_img_maxpool_match=False, 
                        with_img_attentive_match=True, with_img_max_attentive_match=True, image_context_layer=True, img_dim=100):

    only_image = False
    if not only_image:
        cosine_matrix = cal_relevancy_matrix(in_premise_repres, in_hypothesis_repres) # [batch_size, hypothesis_len, premise_len]
        cosine_matrix = mask_relevancy_matrix(cosine_matrix, premise_mask, hypothesis_mask)
        cosine_matrix_transpose = tf.transpose(cosine_matrix, perm=[0,2,1])# [batch_size, premise_len, hypothesis_len]

    # ====word level matching======
    hypthesis_matching_representations = []
    premise_aware_dim = 0
    premise_matching_representations = []
    hypothesis_aware_dim = 0

    # max and mean pooling at word level
    if not only_image:
        hypthesis_matching_representations.append(tf.reduce_max(cosine_matrix, axis=2,keep_dims=True)) # [batch_size, hypothesis_length, 1]
        hypthesis_matching_representations.append(tf.reduce_mean(cosine_matrix, axis=2,keep_dims=True))# [batch_size, hypothesis_length, 1]
        premise_aware_dim += 2
        premise_matching_representations.append(tf.reduce_max(cosine_matrix_transpose, axis=2,keep_dims=True))# [batch_size, premise_len, 1]
        premise_matching_representations.append(tf.reduce_mean(cosine_matrix_transpose, axis=2,keep_dims=True))# [batch_size, premise_len, 1]
        hypothesis_aware_dim += 2
    

    if not only_image:#MP_dim>0:
        if with_max_attentive_match:
            # max_att word level
            prem_max_att = cal_max_premise_representation(in_premise_repres, cosine_matrix)# [batch_size, hypothesis_len, dim]
            prem_max_att_decomp_params = tf.get_variable("prem_word_max_att_decomp_params", shape=[MP_dim, input_dim], dtype=tf.float32)
            prem_max_attentive_rep = cal_attentive_matching(in_hypothesis_repres, prem_max_att, prem_max_att_decomp_params)# [batch_size, hypothesis_len, decompse_dim]
            hypthesis_matching_representations.append(prem_max_attentive_rep)
            premise_aware_dim += MP_dim

            hypo_max_att = cal_max_premise_representation(in_hypothesis_repres, cosine_matrix_transpose)# [batch_size, premise_len, dim]
            hypo_max_att_decomp_params = tf.get_variable("hypo_word_max_att_decomp_params", shape=[MP_dim, input_dim], dtype=tf.float32)
            hypo_max_attentive_rep = cal_attentive_matching(in_premise_repres, hypo_max_att, hypo_max_att_decomp_params)# [batch_size, premise_len, decompse_dim]
            premise_matching_representations.append(hypo_max_attentive_rep)
            hypothesis_aware_dim += MP_dim
    #print('context_layer_num: ', context_layer_num)
    in_premise_previous = [in_premise_repres]
    in_hypothesis_previous = [in_hypothesis_repres]
    with tf.variable_scope('context_MP_matching'):
        for i in xrange(context_layer_num): # support multiple context layer
            with tf.variable_scope('layer-{}'.format(i)):
                with tf.variable_scope('context_represent'):
                    if i > 0:
                        in_premise_repres = tf.concat(in_premise_previous, 2)
                        in_hypothesis_repres = tf.concat(in_hypothesis_previous, 2)
                    # parameters
                    context_lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(context_lstm_dim)
                    context_lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(context_lstm_dim)
                    if is_training:
                        context_lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(context_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
                        context_lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(context_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
                    context_lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([context_lstm_cell_fw])
                    context_lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([context_lstm_cell_bw])



                    # premise representation
                    (premise_context_representation_fw, premise_context_representation_bw), _ = my_rnn.bidirectional_dynamic_rnn(
                                        context_lstm_cell_fw, context_lstm_cell_bw, in_premise_repres, dtype=tf.float32, 
                                        sequence_length=premise_lengths) # [batch_size, premise_len, context_lstm_dim]
                    in_premise_repres = tf.concat([premise_context_representation_fw, premise_context_representation_bw], 2)
                   




                    # hypothesis representation
                    tf.get_variable_scope().reuse_variables()
                    (hypothesis_context_representation_fw, hypothesis_context_representation_bw), _ = my_rnn.bidirectional_dynamic_rnn(
                                        context_lstm_cell_fw, context_lstm_cell_bw, in_hypothesis_repres, dtype=tf.float32, 
                                        sequence_length=hypothesis_lengths) # [batch_size, hypothesis_len, context_lstm_dim]
                    in_hypothesis_repres = tf.concat([hypothesis_context_representation_fw, hypothesis_context_representation_bw], 2)
                   
                    #concat all previous output
                    in_premise_previous.append(in_premise_repres)
                    in_hypothesis_previous.append(in_hypothesis_repres)
                
        


        full_img_rep = None
        image_aware_representatins = []
        image_aware_dim = 0
        if with_image:
            if image_context_layer:
                with tf.variable_scope("image_context_representation"):
                    image_lstm_cell = tf.contrib.rnn.GRUCell(img_dim)

                    #dropout
                    if is_training: image_lstm_cell = tf.contrib.rnn.DropoutWrapper(image_lstm_cell, output_keep_prob=(1 - dropout_rate))

                    image_lstm_cell = tf.contrib.rnn.MultiRNNCell([image_lstm_cell])
                    image_outputs = my_rnn.dynamic_rnn(image_lstm_cell, image_features,
                                        sequence_length=None, dtype=tf.float32)[0]

                    #get outputs
                    full_img_rep = image_outputs[:,-1,:]
                    image_features = image_outputs 
            else:
                full_img_rep = tf.reduce_mean(image_features, axis=1) # [batch_size, feat_dim] 
            image_mask = tf.sequence_mask(tf.tile([49], [tf.shape(image_features)[0]]), 49, dtype=tf.float32)

                
                
        # Multi-perspective matching
        if not only_image:
            with tf.variable_scope('left_MP_matching'):
                (matching_vectors, matching_dim) = match_hypothesis_with_premise(hypothesis_context_representation_fw, 
                                hypothesis_context_representation_bw, hypothesis_mask,
                                premise_context_representation_fw, premise_context_representation_bw,premise_mask,
                                MP_dim, context_lstm_dim, scope=None,
                                with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                                with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match)
                hypthesis_matching_representations.extend(matching_vectors)
                premise_aware_dim += matching_dim
                
            with tf.variable_scope('right_MP_matching'):
                (matching_vectors, matching_dim) = match_hypothesis_with_premise(premise_context_representation_fw, 
                                premise_context_representation_bw, premise_mask,
                                hypothesis_context_representation_fw, hypothesis_context_representation_bw, hypothesis_mask,
                                MP_dim, context_lstm_dim, scope=None,
                                with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                                with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match)


                premise_matching_representations.extend(matching_vectors)
                hypothesis_aware_dim += matching_dim
        
        # image matching
        if with_image:
            with tf.variable_scope('hypothesis_image_matching'):
                (matching_vectors, matching_dim) = match_hypothesis_with_image(image_features,image_features, full_img_rep, full_img_rep, image_mask, 
                                hypothesis_context_representation_fw, hypothesis_context_representation_bw, hypothesis_mask,
                                MP_dim, context_lstm_dim, scope=None,img_dim=100,
                                with_img_full_match=with_img_full_match, with_img_max_attentive_match=with_img_max_attentive_match,
                                with_img_maxpool_match=with_img_maxpool_match, with_img_attentive_match=with_img_attentive_match)
            
                hypthesis_matching_representations.extend(matching_vectors)
                premise_aware_dim += matching_dim

            with tf.variable_scope('image_hypo_matching'):
                (matching_vectors, matching_dim) = match_hypothesis_with_image( hypothesis_context_representation_fw, hypothesis_context_representation_bw, 
                                hypothesis_context_representation_fw[:,-1,:], hypothesis_context_representation_bw[:,0,:], hypothesis_mask,
                                image_features, image_features, image_mask,
                                MP_dim, context_lstm_dim, scope=None,
                                with_img_full_match=with_img_full_match, with_img_max_attentive_match=with_img_max_attentive_match,
                                with_img_maxpool_match=with_img_maxpool_match, with_img_attentive_match=with_img_attentive_match, img_dim=100)
            
                image_aware_representatins.extend(matching_vectors)
                image_aware_dim += matching_dim

            if not image_with_hypothesis_only:
                with tf.variable_scope('premise_image_matching'):
                #with tf.variable_scope('hypothesis_image_matching', reuse=True):
                    (matching_vectors, matching_dim) = match_hypothesis_with_image(image_features,image_features, full_img_rep, full_img_rep, image_mask,
                                premise_context_representation_fw, premise_context_representation_bw, premise_mask,
                                MP_dim, context_lstm_dim, scope=None,img_dim=100,
                                with_img_full_match=with_img_full_match, with_img_max_attentive_match=with_img_max_attentive_match, 
                                with_img_maxpool_match=with_img_maxpool_match, with_img_attentive_match=with_img_attentive_match)

                    premise_matching_representations.extend(matching_vectors)
                    hypothesis_aware_dim += matching_dim

                with tf.variable_scope('image_premise_matching'):
                #with tf.variable_scope('image_hypo_matching', reuse=True):
                    (matching_vectors, matching_dim) = match_hypothesis_with_image( premise_context_representation_fw, premise_context_representation_bw,
                                premise_context_representation_fw[:,-1,:], premise_context_representation_bw[:,0,:], premise_mask,
                                image_features, image_features, image_mask,
                                MP_dim, context_lstm_dim, scope=None,
                                with_img_full_match=with_img_full_match, with_img_max_attentive_match=with_img_max_attentive_match,
                                with_img_maxpool_match=with_img_maxpool_match, with_img_attentive_match=with_img_attentive_match, img_dim=100)

                    image_aware_representatins.extend(matching_vectors)
                    image_aware_dim += matching_dim


        
    hypthesis_matching_representations = tf.concat(hypthesis_matching_representations, 2) # [batch_size, hypothesis_len, premise_aware_dim]
    if not only_image and not image_with_hypothesis_only:
        premise_matching_representations = tf.concat(premise_matching_representations, 2) # [batch_size, premise_len, premise_aware_dim]
    #image_aware_representatins = None
    if with_image: image_aware_representatins = tf.concat(image_aware_representatins, 2)
    if is_training:
        hypthesis_matching_representations = tf.nn.dropout(hypthesis_matching_representations, (1 - dropout_rate))
        if not only_image and not image_with_hypothesis_only:
            premise_matching_representations = tf.nn.dropout(premise_matching_representations, (1 - dropout_rate))
        if with_image: image_aware_representatins = tf.nn.dropout(image_aware_representatins, (1-dropout_rate))
    else:
        hypthesis_matching_representations = tf.multiply(hypthesis_matching_representations, (1 - dropout_rate))
        if not only_image and not image_with_hypothesis_only:
            premise_matching_representations = tf.multiply(premise_matching_representations, (1 - dropout_rate))
        if with_image: image_aware_representatins = tf.multiply(image_aware_representatins, (1-dropout_rate))
        
    # ======Highway layer======
    if with_match_highway:
        with tf.variable_scope("left_matching_highway"):
            hypthesis_matching_representations = multi_highway_layer(hypthesis_matching_representations, premise_aware_dim,highway_layer_num)
        if not only_image and not image_with_hypothesis_only:
            with tf.variable_scope("right_matching_highway"):
                premise_matching_representations = multi_highway_layer(premise_matching_representations, hypothesis_aware_dim,highway_layer_num)
        if with_image:
            with tf.variable_scope("image_matching_highway"):
                image_aware_representatins = multi_highway_layer(image_aware_representatins, image_aware_dim, highway_layer_num)
    #========Aggregation Layer======
    aggregation_representation = []
    aggregation_dim = 0
    
    
    if with_mean_aggregation:
        print('with_mean_aggregation: ', with_mean_aggregation)
        aggregation_representation.append(tf.reduce_max(hypthesis_matching_representations, axis=1))
        aggregation_dim += premise_aware_dim
        if not only_image and not image_with_hypothesis_only:
            aggregation_representation.append(tf.reduce_max(premise_matching_representations, axis=1))
            aggregation_dim += hypothesis_aware_dim
        if with_image: 
            aggregation_representation.append(tf.reduce_max(image_aware_representatins, axis=1))
            aggregation_dim += image_aware_dim

        aggregation_representation = tf.concat(aggregation_representation, 1)#[batch_size, 2 * aggregation_dim]
        return (aggregation_representation, aggregation_dim)

    hypo_aggregation_input = hypthesis_matching_representations
    prem_aggregation_input = premise_matching_representations
    img_aggregation_input = image_aware_representatins
    with tf.variable_scope('aggregation_layer'):
        for i in xrange(aggregation_layer_num): # support multiple aggregation layer
            with tf.variable_scope('left_layer-{}'.format(i)):
                aggregation_lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(aggregation_lstm_dim)
                aggregation_lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(aggregation_lstm_dim)
                if is_training:
                    aggregation_lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(aggregation_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
                    aggregation_lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(aggregation_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
                aggregation_lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([aggregation_lstm_cell_fw])
                aggregation_lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([aggregation_lstm_cell_bw])

                cur_aggregation_representation, _ = my_rnn.bidirectional_dynamic_rnn(
                        aggregation_lstm_cell_fw, aggregation_lstm_cell_bw, hypo_aggregation_input, 
                        dtype=tf.float32, sequence_length=hypothesis_lengths)

                fw_rep = cur_aggregation_representation[0][:,-1,:]
                bw_rep = cur_aggregation_representation[1][:,0,:]
                aggregation_representation.append(fw_rep)
                aggregation_representation.append(bw_rep)
                aggregation_dim += 2* aggregation_lstm_dim
                hypo_aggregation_input = tf.concat(cur_aggregation_representation, 2)# [batch_size, hypothesis_len, 2*aggregation_lstm_dim]

            if not only_image and not image_with_hypothesis_only:
                with tf.variable_scope('right_layer-{}'.format(i)):
                    aggregation_lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(aggregation_lstm_dim)
                    aggregation_lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(aggregation_lstm_dim)
                    if is_training:
                        aggregation_lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(aggregation_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
                        aggregation_lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(aggregation_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
                    aggregation_lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([aggregation_lstm_cell_fw])
                    aggregation_lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([aggregation_lstm_cell_bw])

                    cur_aggregation_representation, _ = my_rnn.bidirectional_dynamic_rnn(
                        aggregation_lstm_cell_fw, aggregation_lstm_cell_bw, prem_aggregation_input, 
                        dtype=tf.float32, sequence_length=premise_lengths)

                    fw_rep = cur_aggregation_representation[0][:,-1,:]
                    bw_rep = cur_aggregation_representation[1][:,0,:]
                    aggregation_representation.append(fw_rep)
                    aggregation_representation.append(bw_rep)
                    aggregation_dim += 2* aggregation_lstm_dim
                    prem_aggregation_input = tf.concat(cur_aggregation_representation, 2)# [batch_size, hypothesis_len, 2*aggregation_lstm_dim]
            if with_image:
                with tf.variable_scope('image_layer-{}'.format(i)):
                    aggregation_lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(aggregation_lstm_dim)
                    aggregation_lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(aggregation_lstm_dim)
                    if is_training:
                        aggregation_lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(aggregation_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
                        aggregation_lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(aggregation_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
                    aggregation_lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([aggregation_lstm_cell_fw])
                    aggregation_lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([aggregation_lstm_cell_bw])

                    cur_aggregation_representation, _ = my_rnn.bidirectional_dynamic_rnn(
                        aggregation_lstm_cell_fw, aggregation_lstm_cell_bw, img_aggregation_input,
                        dtype=tf.float32)

                    fw_rep = cur_aggregation_representation[0][:,-1,:]
                    bw_rep = cur_aggregation_representation[1][:,0,:]
                    aggregation_representation.append(fw_rep)
                    aggregation_representation.append(bw_rep)
                    aggregation_dim += 2* aggregation_lstm_dim
                    img_aggregation_input = tf.concat(cur_aggregation_representation, 2)# [batch_size, hypothesis_len, 2*aggregation_lstm_dim]

    #
    aggregation_representation = tf.concat(aggregation_representation, 1) # [batch_size, aggregation_dim]

    # ======Highway layer======
    if with_aggregation_highway:
        with tf.variable_scope("aggregation_highway"):
            agg_shape = tf.shape(aggregation_representation)
            batch_size = agg_shape[0]
            aggregation_representation = tf.reshape(aggregation_representation, [1, batch_size, aggregation_dim])
            aggregation_representation = multi_highway_layer(aggregation_representation, aggregation_dim, highway_layer_num)
            aggregation_representation = tf.reshape(aggregation_representation, [batch_size, aggregation_dim])
    
    return (aggregation_representation, aggregation_dim)

