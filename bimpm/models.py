import tensorflow as tf
import my_rnn
import match_utils


class ModelGraph(object):
    def __init__(self, num_classes, word_vocab=None, char_vocab=None, POS_vocab=None, NER_vocab=None,
                 dropout_rate=0.5, learning_rate=0.001, optimize_type='adam',lambda_l2=1e-5, 
                 with_word=True, with_char=True, with_POS=True, with_NER=True, 
                 char_lstm_dim=20, context_lstm_dim=100, aggregation_lstm_dim=200, is_training=True,filter_layer_threshold=0.2,
                 MP_dim=50, context_layer_num=1,aggregation_layer_num=1, fix_word_vec=False,with_filter_layer=True, with_highway=False,
                 with_lex_features=False,lex_dim=100,word_level_MP_dim=-1,sep_endpoint=False,end_model_combine=False,with_match_highway=False,
                 with_aggregation_highway=False,highway_layer_num=1,with_lex_decomposition=False, lex_decompsition_dim=-1,
                 with_left_match=True, with_right_match=True,
                 with_full_match=True, with_maxpool_match=True, with_attentive_match=True, with_max_attentive_match=True, with_dep=True, with_image=False,
                 image_with_hypothesis_only=False, with_img_full_match=True, with_img_maxpool_match=False, with_img_attentive_match=True, 
                 with_img_max_attentive_match=True, image_context_layer=True, img_dim=100):

        # ======word representation layer======
        in_question_repres = [] # premise
        in_question_dep_cons = [] # premise dependency connections
        in_passage_repres = [] # hypothesis
        in_passage_dep_cons = [] # hypothesis dependency connections
        self.question_lengths = tf.placeholder(tf.int32, [None])
        self.passage_lengths = tf.placeholder(tf.int32, [None])
        self.truth = tf.placeholder(tf.int32, [None]) # [batch_size]
        input_dim = 0
        # word embedding
        if with_word and word_vocab is not None: 
            self.in_question_words = tf.placeholder(tf.int32, [None, None]) # [batch_size, question_len]
            self.in_passage_words  = tf.placeholder(tf.int32, [None, None]) # [batch_size, passage_len]
            #self.emb_init          = tf.placeholder(tf.float32, shape=word_vocab.word_vecs.shape)
            #self.word_embedding = tf.get_variable("word_embedding", shape=[word_vocab.size()+1, word_vocab.word_dim], initializer=tf.constant(word_vocab.word_vecs), dtype=tf.float32)
            word_vec_trainable = True
            cur_device = '/gpu:0'
            if fix_word_vec: 
                word_vec_trainable = False
                cur_device = '/cpu:0'
            with tf.device(cur_device):
                self.word_embedding = tf.get_variable("word_embedding", trainable=word_vec_trainable, 
                                                  initializer=tf.constant(word_vocab.word_vecs), dtype=tf.float32)
                #self.word_embedding = tf.Variable(self.emb_init, name="word_embedding",trainable=word_vec_trainable, dtype=tf.float32)
            in_question_word_repres = tf.nn.embedding_lookup(self.word_embedding, self.in_question_words) # [batch_size, question_len, word_dim]
            in_passage_word_repres = tf.nn.embedding_lookup(self.word_embedding, self.in_passage_words) # [batch_size, passage_len, word_dim]
            #print (in_question_word_repres)
            in_question_repres.append(in_question_word_repres)
            in_passage_repres.append(in_passage_word_repres)
            
            input_shape = tf.shape(self.in_question_words)
            batch_size = input_shape[0]
            question_len = input_shape[1]
            input_shape = tf.shape(self.in_passage_words)
            passage_len = input_shape[1]
            input_dim += word_vocab.word_dim
        
        if with_dep:
            self.in_question_dependency = tf.placeholder(tf.float32, [None, None, word_vocab.parser.typesize]) # [batch_size, question_len, dep_dim]
            self.in_passage_dependency = tf.placeholder(tf.float32, [None, None, word_vocab.parser.typesize]) # [batch_size, passage_len, dep_dim]
            self.in_question_dep_con = tf.placeholder(tf.int32, [None, None]) # [batch_size, question_len]
            self.in_passage_dep_con = tf.placeholder(tf.int32, [None, None]) # [batch_size, passage_len]
            #dependency representation is the same as data input
            in_question_dep_repres = self.in_question_dependency
            in_passage_dep_repres = self.in_passage_dependency
            
            #in_question_repres.append(in_question_dep_repres)
            #in_passage_repres.append(in_passage_dep_repres)
            
            #input_dim += word_vocab.parser.typesize # dependency_dim
            # embedding dependency later here
            
            with tf.variable_scope('dep_lstm'):
                # lstm cell
                dep_lstm_cell = tf.contrib.rnn.BasicLSTMCell(20)
                # dropout
                if is_training: dep_lstm_cell = tf.contrib.rnn.DropoutWrapper(dep_lstm_cell, output_keep_prob=(1 - dropout_rate))
                dep_lstm_cell = tf.contrib.rnn.MultiRNNCell([dep_lstm_cell])

                # question_representation
                question_dep_outputs = my_rnn.dynamic_rnn(dep_lstm_cell, in_question_dep_repres,
                        sequence_length=self.question_lengths,dtype=tf.float32)[0] # [batch_size, question_len, 20]
                #question_dep_outputs = question_dep_outputs[:,-1,:]
                #question_char_outputs = tf.reshape(question_char_outputs, [batch_size, question_len, char_lstm_dim])

                tf.get_variable_scope().reuse_variables()
                # passage representation
                passage_dep_outputs = my_rnn.dynamic_rnn(dep_lstm_cell, in_passage_dep_repres,
                        sequence_length=self.passage_lengths,dtype=tf.float32)[0] # [batch_size, q_char_len, 20]
                #passage_char_outputs = passage_char_outputs[:,-1,:]
                #passage_char_outputs = tf.reshape(passage_char_outputs, [batch_size, passage_len, char_lstm_dim]) 
            
            in_question_repres.append(question_dep_outputs)
            in_passage_repres.append(passage_dep_outputs)
            input_dim += 20
            #get dependency connections, do smth here? otherwise just pass self.in_question_dep_con to matching function
            in_question_dep_cons = self.in_question_dep_con
            in_passage_dep_cons = self.in_passage_dep_con
        
        out_image_feats = None
        if with_image:
            self.image_feats = tf.placeholder(tf.float32,[None, 49, 512]) #[batch_size, in_feats_dim]
            image_feats = tf.reshape(self.image_feats,[-1, 512])
            #now resize it to context_lstm_dim
            w_0 = tf.get_variable("image_w_0", [512, context_lstm_dim], dtype=tf.float32)
            b_0 = tf.get_variable("image_b_0", [context_lstm_dim], dtype=tf.float32)
            out_image_feats= tf.matmul(image_feats, w_0) + b_0 # [batch_size, 300]
            if is_training:
                out_image_feats = tf.nn.dropout(out_image_feats, (1 - dropout_rate))
            else:
                out_image_feats = tf.multiply(out_image_feats, (1 - dropout_rate))
            out_image_feats = tf.reshape(out_image_feats, [-1, 49, context_lstm_dim])

        if with_POS and POS_vocab is not None: 
            self.in_question_POSs = tf.placeholder(tf.int32, [None, None]) # [batch_size, question_len]
            self.in_passage_POSs = tf.placeholder(tf.int32, [None, None]) # [batch_size, passage_len]
            #self.POS_embedding = tf.get_variable("POS_embedding", shape=[POS_vocab.size()+1, POS_vocab.word_dim], initializer=tf.constant(POS_vocab.word_vecs), dtype=tf.float32)
            self.POS_embedding = tf.get_variable("POS_embedding", initializer=tf.constant(POS_vocab.word_vecs), dtype=tf.float32)

            in_question_POS_repres = tf.nn.embedding_lookup(self.POS_embedding, self.in_question_POSs) # [batch_size, question_len, POS_dim]
            in_passage_POS_repres = tf.nn.embedding_lookup(self.POS_embedding, self.in_passage_POSs) # [batch_size, passage_len, POS_dim]
            in_question_repres.append(in_question_POS_repres)
            in_passage_repres.append(in_passage_POS_repres)

            input_shape = tf.shape(self.in_question_POSs)
            batch_size = input_shape[0]
            question_len = input_shape[1]
            input_shape = tf.shape(self.in_passage_POSs)
            passage_len = input_shape[1]
            input_dim += POS_vocab.word_dim

        if with_NER and NER_vocab is not None: 
            self.in_question_NERs = tf.placeholder(tf.int32, [None, None]) # [batch_size, question_len]
            self.in_passage_NERs = tf.placeholder(tf.int32, [None, None]) # [batch_size, passage_len]
            #self.NER_embedding = tf.get_variable("NER_embedding", shape=[NER_vocab.size()+1, NER_vocab.word_dim], initializer=tf.constant(NER_vocab.word_vecs), dtype=tf.float32)
            self.NER_embedding = tf.get_variable("NER_embedding", initializer=tf.constant(NER_vocab.word_vecs), dtype=tf.float32)

            in_question_NER_repres = tf.nn.embedding_lookup(self.NER_embedding, self.in_question_NERs) # [batch_size, question_len, NER_dim]
            in_passage_NER_repres = tf.nn.embedding_lookup(self.NER_embedding, self.in_passage_NERs) # [batch_size, passage_len, NER_dim]
            in_question_repres.append(in_question_NER_repres)
            in_passage_repres.append(in_passage_NER_repres)

            input_shape = tf.shape(self.in_question_NERs)
            batch_size = input_shape[0]
            question_len = input_shape[1]
            input_shape = tf.shape(self.in_passage_NERs)
            passage_len = input_shape[1]
            input_dim += NER_vocab.word_dim

        if with_char and char_vocab is not None: 
            self.question_char_lengths = tf.placeholder(tf.int32, [None,None]) # [batch_size, question_len]
            self.passage_char_lengths = tf.placeholder(tf.int32, [None,None]) # [batch_size, passage_len]
            self.in_question_chars = tf.placeholder(tf.int32, [None, None, None]) # [batch_size, question_len, q_char_len]
            self.in_passage_chars = tf.placeholder(tf.int32, [None, None, None]) # [batch_size, passage_len, p_char_len]
            input_shape = tf.shape(self.in_question_chars)
            batch_size = input_shape[0]
            question_len = input_shape[1]
            q_char_len = input_shape[2]
            input_shape = tf.shape(self.in_passage_chars)
            passage_len = input_shape[1]
            p_char_len = input_shape[2]
            char_dim = char_vocab.word_dim
#             self.char_embedding = tf.get_variable("char_embedding", shape=[char_vocab.size()+1, char_vocab.word_dim], initializer=tf.constant(char_vocab.word_vecs), dtype=tf.float32)
            self.char_embedding = tf.get_variable("char_embedding", initializer=tf.constant(char_vocab.word_vecs), dtype=tf.float32)

            in_question_char_repres = tf.nn.embedding_lookup(self.char_embedding, self.in_question_chars) # [batch_size, question_len, q_char_len, char_dim]
            in_question_char_repres = tf.reshape(in_question_char_repres, shape=[-1, q_char_len, char_dim])
            question_char_lengths = tf.reshape(self.question_char_lengths, [-1])
            in_passage_char_repres = tf.nn.embedding_lookup(self.char_embedding, self.in_passage_chars) # [batch_size, passage_len, p_char_len, char_dim]
            in_passage_char_repres = tf.reshape(in_passage_char_repres, shape=[-1, p_char_len, char_dim])
            passage_char_lengths = tf.reshape(self.passage_char_lengths, [-1])
            with tf.variable_scope('char_lstm'):
                # lstm cell
                char_lstm_cell = tf.contrib.rnn.BasicLSTMCell(char_lstm_dim)
                # dropout
                if is_training: char_lstm_cell = tf.contrib.rnn.DropoutWrapper(char_lstm_cell, output_keep_prob=(1 - dropout_rate))
                char_lstm_cell = tf.contrib.rnn.MultiRNNCell([char_lstm_cell])

                # question_representation
                question_char_outputs = my_rnn.dynamic_rnn(char_lstm_cell, in_question_char_repres, 
                        sequence_length=question_char_lengths,dtype=tf.float32)[0] # [batch_size*question_len, q_char_len, char_lstm_dim]
                question_char_outputs = question_char_outputs[:,-1,:]
                question_char_outputs = tf.reshape(question_char_outputs, [batch_size, question_len, char_lstm_dim])
             
                tf.get_variable_scope().reuse_variables()
                # passage representation
                passage_char_outputs = my_rnn.dynamic_rnn(char_lstm_cell, in_passage_char_repres, 
                        sequence_length=passage_char_lengths,dtype=tf.float32)[0] # [batch_size*question_len, q_char_len, char_lstm_dim]
                passage_char_outputs = passage_char_outputs[:,-1,:]
                passage_char_outputs = tf.reshape(passage_char_outputs, [batch_size, passage_len, char_lstm_dim])
                
            in_question_repres.append(question_char_outputs)
            in_passage_repres.append(passage_char_outputs)

            input_dim += char_lstm_dim
        #print('\n\n\n')
        #print (in_question_repres)
        #print('\n\n\n')
        in_question_repres = tf.concat(in_question_repres, 2) # [batch_size, question_len, dim]
        in_passage_repres = tf.concat(in_passage_repres, 2) # [batch_size, passage_len, dim]

        if is_training:
            in_question_repres = tf.nn.dropout(in_question_repres, (1 - dropout_rate))
            in_passage_repres = tf.nn.dropout(in_passage_repres, (1 - dropout_rate))
        else:
            in_question_repres = tf.multiply(in_question_repres, (1 - dropout_rate))
            in_passage_repres = tf.multiply(in_passage_repres, (1 - dropout_rate))
        


        mask = tf.sequence_mask(self.passage_lengths, passage_len, dtype=tf.float32) # [batch_size, passage_len]
        question_mask = tf.sequence_mask(self.question_lengths, question_len, dtype=tf.float32) # [batch_size, question_len]

        # ======Highway layer======
        if with_highway:
            with tf.variable_scope("input_highway"):
                in_question_repres = match_utils.multi_highway_layer(in_question_repres, input_dim, highway_layer_num)
                tf.get_variable_scope().reuse_variables()
                in_passage_repres = match_utils.multi_highway_layer(in_passage_repres, input_dim, highway_layer_num)
        
        # ========Bilateral Matching=====
        (match_representation, match_dim) = match_utils.bilateral_match_func(out_image_feats, 
                        in_question_repres, in_passage_repres, in_question_dep_cons, in_passage_dep_cons,
                        self.question_lengths, self.passage_lengths, question_mask, mask, MP_dim, input_dim, 
                        with_filter_layer, context_layer_num, context_lstm_dim,is_training,dropout_rate,
                        with_match_highway,aggregation_layer_num, aggregation_lstm_dim,highway_layer_num,
                        with_aggregation_highway,with_lex_decomposition,lex_decompsition_dim,
                        with_full_match, with_maxpool_match, with_attentive_match, with_max_attentive_match,
                        with_left_match, with_right_match, with_dep=False, with_image=with_image, with_mean_aggregation=True,
                        image_with_hypothesis_only=image_with_hypothesis_only,with_img_attentive_match=with_img_attentive_match, 
                        with_img_full_match=with_img_full_match, with_img_maxpool_match=with_img_maxpool_match, with_img_max_attentive_match=with_img_max_attentive_match, 
                        image_context_layer=image_context_layer, img_dim=100)

        #========Prediction Layer=========
        w_0 = tf.get_variable("w_0", [match_dim, match_dim/2], dtype=tf.float32)
        b_0 = tf.get_variable("b_0", [match_dim/2], dtype=tf.float32)
        w_1 = tf.get_variable("w_1", [match_dim/2, num_classes],dtype=tf.float32)
        b_1 = tf.get_variable("b_1", [num_classes],dtype=tf.float32)

        logits = tf.matmul(match_representation, w_0) + b_0
        logits = tf.tanh(logits)
        if is_training:
            logits = tf.nn.dropout(logits, (1 - dropout_rate))
        else:
            logits = tf.multiply(logits, (1 - dropout_rate))
        logits = tf.matmul(logits, w_1) + b_1

        self.prob = tf.nn.softmax(logits)
        
#         cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tf.cast(self.truth, tf.int64), name='cross_entropy_per_example')
#         self.loss = tf.reduce_mean(cross_entropy, name='cross_entropy')

        gold_matrix = tf.one_hot(self.truth, num_classes, dtype=tf.float32)
        #gold_matrix = tf.one_hot(self.truth, num_classes)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=gold_matrix))

        correct = tf.nn.in_top_k(logits, self.truth, 1)
        self.eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
        self.predictions = tf.arg_max(self.prob, 1)

        if optimize_type == 'adadelta':
            clipper = 50 
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
            tvars = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
            self.loss = self.loss + lambda_l2 * l2_loss
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clipper)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars)) 
        elif optimize_type == 'sgd':
            self.global_step = tf.Variable(0, name='global_step', trainable=False) # Create a variable to track the global step.
            min_lr = 0.000001
            self._lr_rate = tf.maximum(min_lr, tf.train.exponential_decay(learning_rate, self.global_step, 30000, 0.98))
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self._lr_rate).minimize(self.loss)
        elif optimize_type == 'ema':
            tvars = tf.trainable_variables()
            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
            # Create an ExponentialMovingAverage object
            ema = tf.train.ExponentialMovingAverage(decay=0.9999)
            # Create the shadow variables, and add ops to maintain moving averages # of var0 and var1.
            maintain_averages_op = ema.apply(tvars)
            # Create an op that will update the moving averages after each training
            # step.  This is what we will use in place of the usual training op.
            with tf.control_dependencies([train_op]):
                self.train_op = tf.group(maintain_averages_op)
        elif optimize_type == 'adam':
            clipper = 50 
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            tvars = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
            self.loss = self.loss + lambda_l2 * l2_loss
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clipper)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars)) 

        extra_train_ops = []
        train_ops = [self.train_op] + extra_train_ops
        self.train_op = tf.group(*train_ops)

    def get_predictions(self):
        return self.__predictions


    def set_predictions(self, value):
        self.__predictions = value


    def del_predictions(self):
        del self.__predictions



    def get_eval_correct(self):
        return self.__eval_correct


    def set_eval_correct(self, value):
        self.__eval_correct = value


    def del_eval_correct(self):
        del self.__eval_correct


    def get_question_lengths(self):
        return self.__question_lengths


    def get_passage_lengths(self):
        return self.__passage_lengths


    def get_truth(self):
        return self.__truth


    def get_in_question_words(self):
        return self.__in_question_words

    def get_in_question_dep_con(self):
        return self.__in_question_dep_con

    def set_in_question_dep_con(self, value):
        self.__in_question_dep_con = value
    
    def del_in_question_dep_con(self):
        del self.__in_question_dep_con
    
    def get_in_passage_dep_con(self):
        return self.__in_passage_dep_con
    
    def set_in_passage_dep_con(self, value):
        self.__in_passage_dep_con = value
    
    def del_in_passage_dep_con(self):
        del self.__in_passage_dep_con

    def get_in_passage_words(self):
        return self.__in_passage_words
    
    def get_in_question_dependency(self):
        return self.__in_question_dependency
    
    def get_in_passage_dependency(self):
        return self.__in_passage_dependency

    def get_word_embedding(self):
        return self.__word_embedding


    def get_in_question_poss(self):
        return self.__in_question_POSs


    def get_in_passage_poss(self):
        return self.__in_passage_POSs


    def get_pos_embedding(self):
        return self.__POS_embedding


    def get_in_question_ners(self):
        return self.__in_question_NERs


    def get_in_passage_ners(self):
        return self.__in_passage_NERs


    def get_ner_embedding(self):
        return self.__NER_embedding


    def get_question_char_lengths(self):
        return self.__question_char_lengths


    def get_passage_char_lengths(self):
        return self.__passage_char_lengths


    def get_in_question_chars(self):
        return self.__in_question_chars


    def get_in_passage_chars(self):
        return self.__in_passage_chars


    def get_char_embedding(self):
        return self.__char_embedding


    def get_prob(self):
        return self.__prob


    def get_prediction(self):
        return self.__prediction


    def get_loss(self):
        return self.__loss


    def get_train_op(self):
        return self.__train_op


    def get_global_step(self):
        return self.__global_step


    def get_lr_rate(self):
        return self.__lr_rate


    def set_question_lengths(self, value):
        self.__question_lengths = value


    def set_passage_lengths(self, value):
        self.__passage_lengths = value


    def set_truth(self, value):
        self.__truth = value


    def set_in_question_words(self, value):
        self.__in_question_words = value


    def set_in_passage_words(self, value):
        self.__in_passage_words = value
    
    def set_in_question_dependency(self, value):
        self.__in_question_dependency = value

    def set_in_passage_dependency(self, value):
        self.__in_passage_dependency = value

    def set_word_embedding(self, value):
        self.__word_embedding = value


    def set_in_question_poss(self, value):
        self.__in_question_POSs = value


    def set_in_passage_poss(self, value):
        self.__in_passage_POSs = value


    def set_pos_embedding(self, value):
        self.__POS_embedding = value


    def set_in_question_ners(self, value):
        self.__in_question_NERs = value


    def set_in_passage_ners(self, value):
        self.__in_passage_NERs = value


    def set_ner_embedding(self, value):
        self.__NER_embedding = value


    def set_question_char_lengths(self, value):
        self.__question_char_lengths = value


    def set_passage_char_lengths(self, value):
        self.__passage_char_lengths = value


    def set_in_question_chars(self, value):
        self.__in_question_chars = value


    def set_in_passage_chars(self, value):
        self.__in_passage_chars = value


    def set_char_embedding(self, value):
        self.__char_embedding = value


    def set_prob(self, value):
        self.__prob = value


    def set_prediction(self, value):
        self.__prediction = value


    def set_loss(self, value):
        self.__loss = value


    def set_train_op(self, value):
        self.__train_op = value


    def set_global_step(self, value):
        self.__global_step = value


    def set_lr_rate(self, value):
        self.__lr_rate = value


    def del_question_lengths(self):
        del self.__question_lengths


    def del_passage_lengths(self):
        del self.__passage_lengths


    def del_truth(self):
        del self.__truth


    def del_in_question_words(self):
        del self.__in_question_words


    def del_in_passage_words(self):
        del self.__in_passage_words

    def del_in_question_dependency(self):
        del self.__in_question_dependency

    def del_in_passage_dependency(self):
        del self.__in_passage_dependency

    def del_word_embedding(self):
        del self.__word_embedding


    def del_in_question_poss(self):
        del self.__in_question_POSs


    def del_in_passage_poss(self):
        del self.__in_passage_POSs


    def del_pos_embedding(self):
        del self.__POS_embedding


    def del_in_question_ners(self):
        del self.__in_question_NERs


    def del_in_passage_ners(self):
        del self.__in_passage_NERs


    def del_ner_embedding(self):
        del self.__NER_embedding


    def del_question_char_lengths(self):
        del self.__question_char_lengths


    def del_passage_char_lengths(self):
        del self.__passage_char_lengths


    def del_in_question_chars(self):
        del self.__in_question_chars


    def del_in_passage_chars(self):
        del self.__in_passage_chars


    def del_char_embedding(self):
        del self.__char_embedding


    def del_prob(self):
        del self.__prob


    def del_prediction(self):
        del self.__prediction


    def del_loss(self):
        del self.__loss


    def del_train_op(self):
        del self.__train_op


    def del_global_step(self):
        del self.__global_step


    def del_lr_rate(self):
        del self.__lr_rate

    def get_image_feats(self):
        return self.__image_feats

    def set_image_feats(self, value):
        self.__image_feats = value

    def del_image_feats(self):
        del self.__image_feats

    def get_emb_init(self):
        return self.__emb_init

    def set_emb_init(self, value):
        self.__emb_init=value

    def del_emb_init(self):
        del self.__emb_init

    image_feats = property(get_image_feats, set_image_feats, del_image_feats, "image_features's docstring")
    question_lengths = property(get_question_lengths, set_question_lengths, del_question_lengths, "question_lengths's docstring")
    passage_lengths = property(get_passage_lengths, set_passage_lengths, del_passage_lengths, "passage_lengths's docstring")
    truth = property(get_truth, set_truth, del_truth, "truth's docstring")
    emb_init = property(get_emb_init, set_emb_init, del_emb_init, "pretrained word embedding")
    in_question_words = property(get_in_question_words, set_in_question_words, del_in_question_words, "in_question_words's docstring")
    in_passage_words = property(get_in_passage_words, set_in_passage_words, del_in_passage_words, "in_passage_words's docstring")
    in_question_dependency = property(get_in_question_dependency, set_in_question_dependency, del_in_question_dependency, "in_question_dependency's docstring")
    in_passage_dependency = property(get_in_passage_dependency, set_in_passage_dependency, del_in_passage_dependency, "in_passage_dependency's docstring")
    in_question_dep_con = property(get_in_question_dep_con, set_in_question_dep_con, del_in_question_dep_con, "in_question_dependency connections's docstring")
    in_passage_dep_con = property(get_in_passage_dep_con, set_in_passage_dep_con, del_in_passage_dep_con, "in_passage_dependency connections's docstring")
    word_embedding = property(get_word_embedding, set_word_embedding, del_word_embedding, "word_embedding's docstring")
    in_question_POSs = property(get_in_question_poss, set_in_question_poss, del_in_question_poss, "in_question_POSs's docstring")
    in_passage_POSs = property(get_in_passage_poss, set_in_passage_poss, del_in_passage_poss, "in_passage_POSs's docstring")
    POS_embedding = property(get_pos_embedding, set_pos_embedding, del_pos_embedding, "POS_embedding's docstring")
    in_question_NERs = property(get_in_question_ners, set_in_question_ners, del_in_question_ners, "in_question_NERs's docstring")
    in_passage_NERs = property(get_in_passage_ners, set_in_passage_ners, del_in_passage_ners, "in_passage_NERs's docstring")
    NER_embedding = property(get_ner_embedding, set_ner_embedding, del_ner_embedding, "NER_embedding's docstring")
    question_char_lengths = property(get_question_char_lengths, set_question_char_lengths, del_question_char_lengths, "question_char_lengths's docstring")
    passage_char_lengths = property(get_passage_char_lengths, set_passage_char_lengths, del_passage_char_lengths, "passage_char_lengths's docstring")
    in_question_chars = property(get_in_question_chars, set_in_question_chars, del_in_question_chars, "in_question_chars's docstring")
    in_passage_chars = property(get_in_passage_chars, set_in_passage_chars, del_in_passage_chars, "in_passage_chars's docstring")
    char_embedding = property(get_char_embedding, set_char_embedding, del_char_embedding, "char_embedding's docstring")
    prob = property(get_prob, set_prob, del_prob, "prob's docstring")
    prediction = property(get_prediction, set_prediction, del_prediction, "prediction's docstring")
    loss = property(get_loss, set_loss, del_loss, "loss's docstring")
    train_op = property(get_train_op, set_train_op, del_train_op, "train_op's docstring")
    global_step = property(get_global_step, set_global_step, del_global_step, "global_step's docstring")
    lr_rate = property(get_lr_rate, set_lr_rate, del_lr_rate, "lr_rate's docstring")
    eval_correct = property(get_eval_correct, set_eval_correct, del_eval_correct, "eval_correct's docstring")
    predictions = property(get_predictions, set_predictions, del_predictions, "predictions's docstring")
    
    
