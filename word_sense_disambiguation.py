"""
check some glove words
"""
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sys import stdout
import numpy as np
from matplotlib import pyplot
import sys

def build_glove_dictionary():
    """
        builds a dictionary based on the glove model.
        http://nlp.stanford.edu/projects/glove/
        dictionary will have the form of key = token, value = numpy array with the pretrained values

        REALLY IMPORTANT the glove dataset. with the big one finds nearly everything....
        smallest one...quite baaaaaad...
    """
    print ('building glove dictionary...')
    glove_file = '../TBIR/glove.840B.300d.txt'
    glove_dict = {}
    with open(glove_file) as fd_glove:
        j=0
        for i, input in enumerate(fd_glove):
            input_split = input.split(" ")
            #print input_split
            key = input_split[0] #get key
            del input_split[0]  # remove key
            j+=1
            stdout.write("\rloading glove dictionary: %d" % j)
            stdout.flush()
            values = []
            for value in input_split:
                values.append(float(value))
            np_values = np.asarray(values)
            glove_dict[key] = np_values
            #else:
                #print key
    print ""
    print 'dictionary build with length', len(glove_dict)

    return glove_dict

def build_glove_matrix(glove_dictionary):
    """
        return word2idx and matrix
    """
    idx2word = {}
    glove_matrix = []
    i=0
    for key, value in glove_dictionary.iteritems():
        idx2word[i] = key
        glove_matrix.append(value)
        i+=1
    return np.asarray(glove_matrix), idx2word

def check_similarity(glove_matrix, word):
    return cosine_similarity(word.reshape(1, -1), glove_matrix)

def build_matrix_to_tsne(glove_dict, tokens):
    matrix = []
    for token in tokens:
        if token in glove_dict:
            matrix.append(glove_dict[token])
    return matrix
words = []
if len(sys.argv)<2:
    print 'Words not specified'
    words = ["plant", "factory", "machine", "houseplant", "cake"]
else:
    for i in range(1, len(sys.argv)):
        words.append(sys.argv[i])

print 'Words that will be used', words

glove_dict = build_glove_dictionary()
glove_matrix, idx2word = build_glove_matrix(glove_dict)
model = TSNE(n_components=2, random_state=0)

to_plot = []
labels = []
not_found = 0
len_words = len(words)
for word in words:
    try:
        cosine_matrix = check_similarity(glove_matrix, glove_dict[word])
        ind = cosine_matrix[0].argsort()[-100:][::-1]
        closest = ind.tolist()
        tokens = [idx2word[idx] for idx in closest]
        to_reduce = build_matrix_to_tsne(glove_dict, tokens)
        #print to_reduce.shape
        labels += [token for token in tokens]
        to_plot += [x_y for x_y in to_reduce]
    except:
        len_words-=1
        print 'Word not found', word

print len_words
#print to_plot.shape
#print to_plot
X_hdim = np.array(to_plot)
#print X_hdim
print X_hdim.shape
X = model.fit_transform(X_hdim)
num_words_print = 1
X_x = np.zeros((len_words*5, 2))
labels_x = []
print X.shape
k=0
ranges = [x*100 for x in range (0, len_words)]
print ranges
for i in ranges:
    for j in range(1, num_words_print+1):
        print i+j-1, k
        X_x[k] = X[i+j-1]
        k+=1
        labels[i+j-1]
        labels_x.append(labels[i+j-1])


print labels_x
print X_x.shape
pyplot.scatter(X_x[:,0],X_x[:,1])
for i, label in enumerate(labels_x):
    pyplot.annotate(label, (X_x[i,0],X_x[i,1]))
pyplot.show()
