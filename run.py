__author__ = 'david_torrejon'


from extract_sentences import give_vocabulary, read_json_file, create_sentence_ds, build_glove_dictionary
from model_simple_rnn import paper_model
from shuffle_data import get_test_train_sets
from embeddings import create_embeddings
from tbir_project_data import load_data
import numpy as np
from sys import stdout


#split parameters for dev set
percentage_split_ds = 0.80
shitty_pc = True #to run it in shitty pcs turn it on
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
glove_dict = build_glove_dictionary()
snli_model = paper_model(3, is_tbir=is_tbir_test)#stacked layers
#create MODEL ADD TYPE OF MODEL RNN OR LSTM build(model='type') word2idx might not be needed in the model anymore
# for X_train, Y_train in MiniBatchGenerator(): instead of building the embeddings i could build small chunks of 1k sentences!

if train_model:
    df_data, max_pre, max_hypo = read_json_file(train=True)
    if max_pre > max_hypo:
        maxlen = max_pre
    else:
        maxlen = max_hypo
    test_file, max_pre_t, max_hypo_t  = read_json_file(train=False)
    snli_model.build_model(LOAD_W=LOAD_W, max_pre = max_pre, max_hypo = max_hypo)
    print len(df_data)
    for epoch in range(0,5):
        print("epoch: %d" % epoch)
        print '-'*10
        for batch_range in range(0,len(df_data),cut_ds):
            print("batch range %s" % batch_range)
            print '-'*15
            data_train = create_embeddings(df_data, glove_dict, batch_range, cut_ds,  max_pre = max_pre, max_hypo = max_hypo)
            snli_model.train_model(data_train)
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
            data_test = create_embeddings(test_file[batch_test:batch_test+test_batch], glove_dict, batch_test, cut_ds, test_tbir=is_tbir_test, max_pre = max_pre, max_hypo = max_hypo)
            snli_model.test_model(data_test, is_tbir=is_tbir_test)
    del df_data



#snli_model.test_model(data_test)
#train model

#model test
#snli_model.model_evaluate(test_set)
