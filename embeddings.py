__author__ = 'david_torrejon'

from random import seed, uniform
from extract_sentences import make_unicode, label_output_data
import numpy as np
import re
import string
from sys import stdout

def create_embedding_sentence(sentence, glove_dict, unk_vector,  word2idx, onehot=False, maxlen=45):

    vectorized_sentence = []
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    sentence = regex.sub('', sentence).lower()
    tokenized_sentence = sentence.split(" ")
    if len(tokenized_sentence)>maxlen:
        return False
    if onehot:
        idx_sentence = []
        #for token in tokenized_sentence:
        #print tokenized_sentence

        for token in tokenized_sentence:
            try:
                if token in word2idx:
                    idx_sentence.append(word2idx[token])
                else:
                    idx_sentence.append(word2idx["unk0"])
            except:
                print ('token: ', token, 'went wrong...')

        #print idx_sentence
        while len(idx_sentence) < maxlen:
                idx_sentence.append(0)
        idx_sentence = np.array(idx_sentence)
        #print idx_sentence
        return idx_sentence
    else:
        for token in tokenized_sentence:
            try:
                if token in glove_dict:
                    vectorized_sentence.append(glove_dict[token])
                else:
                    vectorized_sentence.append(unk_vector) # dunno how to deal with mistakes, ask!
            except:
                print ('token: ', token, 'went wrong...')

    while len(vectorized_sentence) < maxlen:
        vectorized_sentence.append(np.zeros(300))

    if len(vectorized_sentence) > maxlen:
        return np.asarray(vectorized_sentence[:maxlen])
    else:
        return np.asarray(vectorized_sentence)


def create_embeddings(df_data, glove_dict, batch_i, batch_size, unk_vector, word2idx,test_tbir=False, max_pre=45, max_hypo=45, onehot=False):

    embedded_sentences = []
    print('Generating embeddings')
    if test_tbir is False:

        #print df_data
        list_premises = df_data['sentence1'].tolist()
        list_hypothesis = df_data['sentence2'].tolist()
        list_label = df_data['gold_label'].tolist()
        list_annotator = df_data['annotator_labels'].tolist()
        if batch_i+batch_size > len(df_data):
            limit = len(df_data)-1
        else:
            limit = batch_i+batch_size

        for i in range(batch_i, limit): #zip(list_premises, list_hypothesis, list_label, list_annotator):
            stdout.write("\rloading embeddings: %d" % i)
            stdout.flush()
            label_no_unicode = make_unicode(list_label[i])
            numpy_label = label_output_data(label_no_unicode)
            premise_encoded=create_embedding_sentence(list_premises[i], glove_dict, unk_vector=unk_vector,maxlen=max_pre, onehot=onehot, word2idx=word2idx)
            hypothesis_encoded=create_embedding_sentence(list_hypothesis[i], glove_dict, unk_vector=unk_vector,maxlen=max_hypo, onehot=onehot, word2idx=word2idx) # do the cut here whether it has same labels or not
            if premise_encoded is not False or hypothesis_encoded is not False:
                embedded_sentences.append([[premise_encoded, hypothesis_encoded], numpy_label, len(set(list_annotator[i]))])
    else:
        for i,data_point in enumerate(df_data):
            stdout.write("\rloading embeddings tbir: %d" % i)
            #embeddings creation here
            #print data_point
            #raise SystemExit(0)
            premise_encoded = create_embedding_sentence(data_point[0], glove_dict)
            hypothesis_encoded=create_embedding_sentence(data_point[1], glove_dict)
            #numpy_label = label_output_data(data_point[3])
            embedded_sentences.append([[premise_encoded, hypothesis_encoded], data_point[2], data_point[3], data_point[4]])

    #print embedded_sentences[0] #make sure that works! remove at the end
    print " "
    return embedded_sentences
