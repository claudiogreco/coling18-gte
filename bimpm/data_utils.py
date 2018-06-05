import numpy as np
import re

def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)] # zgwang: starting point of each batch

def pad_2d_matrix(in_val, max_length=None, dtype=np.int32):
    if max_length is None: max_length = np.max([len(cur_in_val) for cur_in_val in in_val])
    batch_size = len(in_val)
    out_val = np.zeros((batch_size, max_length), dtype=dtype)
    for i in xrange(batch_size):
        cur_in_val = in_val[i]
        kept_length = len(cur_in_val)
        if kept_length>max_length: kept_length = max_length
        out_val[i,:kept_length] = cur_in_val[:kept_length]
    return out_val

def pad_3d_tensor(in_val, max_length1=None, max_length2=None, dtype=np.int32):
    if max_length1 is None: max_length1 = np.max([len(cur_in_val) for cur_in_val in in_val])
    if max_length2 is None: max_length2 = np.max([np.max([len(val) for val in cur_in_val]) for cur_in_val in in_val])
    batch_size = len(in_val)
    out_val = np.zeros((batch_size, max_length1, max_length2), dtype=dtype)
    for i in xrange(batch_size):
        cur_length1 = max_length1
        if len(in_val[i])<max_length1: cur_length1 = len(in_val[i])
        for j in xrange(cur_length1):
            cur_in_val = in_val[i][j]
            kept_length = len(cur_in_val)
            if kept_length>max_length2: kept_length = max_length2
            out_val[i, j, :kept_length] = cur_in_val[:kept_length]
    return out_val



class DataStream(object):
    def __init__(self, inpath, word_vocab=None, char_vocab=None, POS_vocab=None, NER_vocab=None, label_vocab=None, batch_size=60, 
                 isShuffle=False, isLoop=False, isSort=True, max_char_per_word=10, with_dep = False, max_sent_length=200, with_image=False, image_feats=None, sick_data=False):
        instances = []
        count_ins = 0
        infile = open(inpath, 'rt')
        for line in infile:
            #print (line)
            count_ins +=1
            if count_ins == 1: continue
            if(count_ins > 3000): break
            line = line.decode('utf-8').strip()
            if line.startswith('-'): continue
            items = re.split("\t", line)
            label = items[0]
            sentence1 = items[1].lower()
            sentence2 = items[2].lower()


            img_feats = None
            img_id = None
            if with_image:
                img_id = items[3]
                try:
                    img_id = img_id.strip().split('#')[0]
                    img_feats=image_feats.get_feat(img_id)
                    img_feats = np.reshape(img_feats, (-1, 512))
                    if img_feats is None: raise Exception('feature not found for ' + items[3])
                except:
                    print("Cannot get feature for: ", img_id)
                    continue
            
            if label_vocab is not None:
                if sick_data: label = label + ' '
                label_id = label_vocab.getIndex(label)
                #if count_ins < 10: print ("label: ", label, "\tlabel_id: ", label_id)
                if label_id >= label_vocab.vocab_size: label_id = 0
            else: 
                label_id = int(label)
            word_idx_1 = word_vocab.to_index_sequence(sentence1)
            word_idx_2 = word_vocab.to_index_sequence(sentence2)
            #print(len(word_idx_1), len(word_idx_2))
            dependency1, dependency2 = None, None
            dep_con1, dep_con2 = None, None
            if with_dep:
                dependency1, dep_con1 = word_vocab.dep_sequence(items[1])#(sentence_length, dependency_dim)
                dependency2, dep_con2 = word_vocab.dep_sequence(items[2])

            char_matrix_idx_1 = char_vocab.to_character_matrix(sentence1)
            char_matrix_idx_2 = char_vocab.to_character_matrix(sentence2)
            #print('character_len: ', len(char_matrix_idx_1), len(char_matrix_idx_2))
            if len(word_idx_1)>max_sent_length: 
                word_idx_1 = word_idx_1[:max_sent_length]
                char_matrix_idx_1 = char_matrix_idx_1[:max_sent_length]
                if with_dep:
                    dependency1 = dependency1[:max_sent_length, :]
                    dep_con1 = dep_con1[:max_sent_length]
            if len(word_idx_2)>max_sent_length:
                word_idx_2 = word_idx_2[:max_sent_length]
                char_matrix_idx_1 = char_matrix_idx_1[:max_sent_length]
                if with_dep:
                    dependency2 = dependency2[:max_sent_length, :]
                    dep_con2 = dep_con1[:max_sent_length]

            POS_idx_1 = None
            POS_idx_2 = None
            if POS_vocab is not None:
                POS_idx_1 = POS_vocab.to_index_sequence(items[3])
                if len(POS_idx_1)>max_sent_length: POS_idx_1 = POS_idx_1[:max_sent_length]
                POS_idx_2 = POS_vocab.to_index_sequence(items[4])
                if len(POS_idx_2)>max_sent_length: POS_idx_2 = POS_idx_2[:max_sent_length]

            NER_idx_1 = None
            NER_idx_2 = None
            if NER_vocab is not None:
                NER_idx_1 = NER_vocab.to_index_sequence(items[5])
                if len(NER_idx_1)>max_sent_length: NER_idx_1 = NER_idx_1[:max_sent_length]
                NER_idx_2 = NER_vocab.to_index_sequence(items[6])
                if len(NER_idx_2)>max_sent_length: NER_idx_2 = NER_idx_2[:max_sent_length]
            

            instances.append((label, sentence1, sentence2, label_id, word_idx_1, word_idx_2, char_matrix_idx_1, char_matrix_idx_2,
                              POS_idx_1, POS_idx_2, NER_idx_1, NER_idx_2, dependency1, dependency2, dep_con1, dep_con2, img_feats, img_id))
        infile.close()

        # sort instances based on sentence length
        if isSort: instances = sorted(instances, key=lambda instance: (len(instance[4]), len(instance[5]))) # sort instances based on length
        self.num_instances = len(instances)
        
        # distribute into different buckets
        batch_spans = make_batches(self.num_instances, batch_size) 
        self.batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            label_batch = []
            sent1_batch = []
            sent2_batch = []
            label_id_batch = []
            word_idx_1_batch = []
            word_idx_2_batch = []
            dependency1_batch = []
            dependency2_batch = []
            dep_con1_batch = []
            dep_con2_batch = []
            char_matrix_idx_1_batch = []
            char_matrix_idx_2_batch = []
            sent1_length_batch = []
            sent2_length_batch = []
            sent1_char_length_batch = []
            sent2_char_length_batch = []

            POS_idx_1_batch = None
            if POS_vocab is not None: POS_idx_1_batch = []
            POS_idx_2_batch = None
            if POS_vocab is not None: POS_idx_2_batch = []

            NER_idx_1_batch = None
            if NER_vocab is not None: NER_idx_1_batch = []
            NER_idx_2_batch = None
            if NER_vocab is not None: NER_idx_2_batch = []
            
            img_feats_batch=None
            img_id_batch = []
            if with_image:
                img_feats_batch= []

            for i in xrange(batch_start, batch_end):
                (label, sentence1, sentence2, label_id, word_idx_1, word_idx_2, char_matrix_idx_1, char_matrix_idx_2,
                 POS_idx_1, POS_idx_2, NER_idx_1, NER_idx_2, dependency1, dependency2, dep_con1, dep_con2, img_feats, img_id) = instances[i]
                label_batch.append(label)
                sent1_batch.append(sentence1)
                sent2_batch.append(sentence2)
                label_id_batch.append(label_id)
                word_idx_1_batch.append(word_idx_1)
                word_idx_2_batch.append(word_idx_2)
                # add sentence dependencies to batch
                if with_dep:
                    dependency1_batch.append(dependency1)
                    dependency2_batch.append(dependency2)
                    dep_con1_batch.append(dep_con1)
                    dep_con2_batch.append(dep_con2)
                char_matrix_idx_1_batch.append(char_matrix_idx_1)
                char_matrix_idx_2_batch.append(char_matrix_idx_2)
                sent1_length_batch.append(len(word_idx_1))
                sent2_length_batch.append(len(word_idx_2))
                sent1_char_length_batch.append([len(cur_char_idx) for cur_char_idx in char_matrix_idx_1])
                sent2_char_length_batch.append([len(cur_char_idx) for cur_char_idx in char_matrix_idx_2])

                if POS_vocab is not None: 
                    POS_idx_1_batch.append(POS_idx_1)
                    POS_idx_2_batch.append(POS_idx_2)

                if NER_vocab is not None: 
                    NER_idx_1_batch.append(NER_idx_1)
                    NER_idx_2_batch.append(NER_idx_2)

                if with_image:
                    img_feats_batch.append(img_feats) #[batch_size, feat_dim(4096)]
                img_id_batch.append(img_id)
                
                
            cur_batch_size = len(label_batch)
            if cur_batch_size ==0: continue

            # padding
            max_sent1_length = np.max(sent1_length_batch)
            max_sent2_length = np.max(sent2_length_batch)
            #print('maxSentenceLength: ', max_sent1_length, max_sent2_length)
            max_char_length1 = np.max([np.max(aa) for aa in sent1_char_length_batch])
            if max_char_length1>max_char_per_word: max_char_length1=max_char_per_word

            max_char_length2 = np.max([np.max(aa) for aa in sent2_char_length_batch])
            if max_char_length2>max_char_per_word: max_char_length2=max_char_per_word
            
            label_id_batch = np.array(label_id_batch)
            word_idx_1_batch = pad_2d_matrix(word_idx_1_batch, max_length=max_sent1_length)
            word_idx_2_batch = pad_2d_matrix(word_idx_2_batch, max_length=max_sent2_length)
            #print('word_idx_shape:', word_idx_1_batch.shape(), word_idx_2_batch.shape())
            #print '\nword_idx_shape'
            #print np.shape(word_idx_1_batch)
            #print np.shape(word_idx_2_batch)
            # padding sentence dependency
            if with_dep:
                dependency1_batch = pad_3d_tensor(dependency1_batch, max_length1=max_sent1_length, max_length2=word_vocab.parser.typesize)
                dependency2_batch = pad_3d_tensor(dependency2_batch, max_length1=max_sent2_length, max_length2=word_vocab.parser.typesize)
                dep_con1_batch = pad_2d_matrix(dep_con1_batch, max_length=max_sent1_length)
                dep_con2_batch = pad_2d_matrix(dep_con2_batch, max_length=max_sent2_length)
            #print('dependency shape:', dependency1_batch.shape(), dependency1_batch.shape())
            #print '\ndependency shape'
            #print np.shape(dependency1_batch)
            #print np.shape(dependency2_batch)

            char_matrix_idx_1_batch = pad_3d_tensor(char_matrix_idx_1_batch, max_length1=max_sent1_length, max_length2=max_char_length1)
            char_matrix_idx_2_batch = pad_3d_tensor(char_matrix_idx_2_batch, max_length1=max_sent2_length, max_length2=max_char_length2)
            
            #print 'char_matrix shape'
            #print np.shape(char_matrix_idx_1_batch)
            #print np.shape(char_matrix_idx_2_batch)

            sent1_length_batch = np.array(sent1_length_batch)
            sent2_length_batch = np.array(sent2_length_batch)

            sent1_char_length_batch = pad_2d_matrix(sent1_char_length_batch, max_length=max_sent1_length)
            sent2_char_length_batch = pad_2d_matrix(sent2_char_length_batch, max_length=max_sent2_length)
            
            if POS_vocab is not None:
                POS_idx_1_batch = pad_2d_matrix(POS_idx_1_batch, max_length=max_sent1_length)
                POS_idx_2_batch = pad_2d_matrix(POS_idx_2_batch, max_length=max_sent2_length)
            if NER_vocab is not None:
                NER_idx_1_batch = pad_2d_matrix(NER_idx_1_batch, max_length=max_sent1_length)
                NER_idx_2_batch = pad_2d_matrix(NER_idx_2_batch, max_length=max_sent2_length)
            
            if with_image:
                img_feats_batch = np.array(img_feats_batch)

            self.batches.append((label_batch, sent1_batch, sent2_batch, label_id_batch, word_idx_1_batch, word_idx_2_batch, 
                                 char_matrix_idx_1_batch, char_matrix_idx_2_batch, sent1_length_batch, sent2_length_batch, 
                                 sent1_char_length_batch, sent2_char_length_batch,
                                 POS_idx_1_batch, POS_idx_2_batch, NER_idx_1_batch, NER_idx_2_batch, 
                                 dependency1_batch, dependency2_batch, dep_con1_batch, dep_con2_batch, img_feats_batch, img_id_batch))
        
        instances = None
        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        self.isShuffle = isShuffle
        if self.isShuffle: np.random.shuffle(self.index_array) 
        self.isLoop = isLoop
        self.cur_pointer = 0
    
    def nextBatch(self):
        if self.cur_pointer>=self.num_batch:
            if not self.isLoop: return None
            self.cur_pointer = 0 
            if self.isShuffle: np.random.shuffle(self.index_array) 
#         print('{} '.format(self.index_array[self.cur_pointer]))
        cur_batch = self.batches[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_batch

    def reset(self):
        self.cur_pointer = 0
    
    def get_num_batch(self):
        return self.num_batch

    def get_num_instance(self):
        return self.num_instances

    def get_batch(self, i):
        if i>= self.num_batch: return None
        return self.batches[i]
        
