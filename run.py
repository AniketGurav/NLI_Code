__author__ = 'david_torrejon'


from extract_sentences import give_vocabulary, read_json_file, create_sentence_ds, build_glove_dictionary, get_word_idx_vocab, make_unicode
from model_simple_rnn import paper_model
from shuffle_data import get_test_train_sets
from embeddings import create_embeddings, create_embedding_sentence
from tbir_project_data import load_data
import numpy as np
from sys import stdout
import re
import string

#split parameters for dev set
#percentage_split_ds = 0.80
#shitty_pc = True #to run it in shitty pcs turn it on
#it will shrink the ds by the fixed amount in the dev set consisting of 10k sent 0.4 will
#create a ds with only 4k pairs or both train and test...
# no need shitty_pc just fixate cut_ds to 1
cut_ds = 10000
LOAD_W = False

train_model = True
is_tbir_test = False

#dataset = create_embeddings(df_data, glove_dict, cut_ds)
#vocabulary, size_vocabulary, word2idx = give_vocabulary(df_data)
#dataset = create_sentence_ds(df_data, word2idx, cut_ds)
#print dataset[0][0][0].shape # ([[premise, hypothesis],numpy_label]) premise[n][0][0], hypothesis[n][0][1], label[n][1]
#print dataset[340][0][0].shape


#train_set, test_set = get_test_train_sets(dataset, percentage_split_ds)
# feed model with premise, hypothesis

#create class model

snli_model = paper_model(3, is_tbir=is_tbir_test)#stacked layers

remove_neutral = False
if train_model:
    df_data, max_pre, max_hypo = read_json_file(train=True, remove_neutral = remove_neutral)
    vocabulary = give_vocabulary(df_data)
    test_file, max_pre_t, max_hypo_t  = read_json_file(train=False, remove_neutral = remove_neutral)
    vocabulary_t = give_vocabulary(test_file)
    vocab = vocabulary.union(vocabulary_t)

    print "length train/test vocab", len(vocab)
    word2idx = get_word_idx_vocab(list(vocab))
    glove_dict = build_glove_dictionary(vocab)


    """
    check unknowns
    """
    unk_vector = .1 * np.random.random_sample((300,)) - 0.05
    print unk_vector.shape
    #print unk_vector
    n_symbols = len(word2idx) + 1 # adding 1 to account for 0th index (for masking)
    embedding_weights = np.zeros((n_symbols+1,300))

    rand_init = False
    if rand_init:
        print "initializing random word embeddings"
        for word,index in word2idx.items():
            embedding_weights[index,:] = create_embedding_sentence(word, glove_dict, unk_vector, maxlen=1, onehot=False, word2idx=word2idx, random_vector=True)
    else:
        print "generating embeddings matrix"
        for word,index in word2idx.items():
            embedding_weights[index,:] = create_embedding_sentence(word, glove_dict, unk_vector, maxlen=1, onehot=False, word2idx=word2idx)


    snli_model.build_model(LOAD_W=LOAD_W, max_pre = max_pre, max_hypo = max_hypo, emb_init = embedding_weights, n_symbols=n_symbols)
    print len(df_data)

    for epoch in range(0,100):
        print("epoch: %d" % epoch)
        print '-'*10
        for batch_range in range(0,len(df_data),cut_ds):
            print("batch range %s" % batch_range)
            print '-'*15
            data_train = create_embeddings(df_data, glove_dict, batch_range, cut_ds,  max_pre = max_pre, max_hypo = max_hypo, unk_vector=unk_vector, onehot=True, word2idx=word2idx)

            snli_model.train_model(data_train, batch_range=batch_range)
        '''
        After each training, test and check with the test set.
        '''
        #if is_tbir_test:
        #    test_file = load_data()
        #else:

        #print len(test_file)
        test_batch = 10000
            #snli_model.build_model(LOAD_W=LOAD_W, max_pre = max_pre, max_hypo = max_hypo)
        for batch_test in range(0, len(test_file), test_batch):
            print("batch range %s" % batch_test)
            data_test = create_embeddings(test_file[batch_test:batch_test+test_batch], glove_dict, batch_test, cut_ds,max_pre = max_pre, max_hypo = max_hypo, unk_vector=unk_vector, onehot=True, word2idx=word2idx)
            perc = snli_model.test_model(data_test, is_tbir=is_tbir_test, test_file=test_file)
            #print snli_model.layers[0]
        f = open('test_accuracy.txt', 'a')
        perc = str(perc) + '\n'
        f.write(str(perc))
        f.close()
    del df_data


#snli_model.test_model(data_test)
#train model

#model test
#snli_model.model_evaluate(test_set)
