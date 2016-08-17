
import sys
import numpy as np
from collections import Counter
from stop_words import get_stop_words
import nltk
import operator

def count_tags(words):
    """
    A word is [(word, tag), times]
    """
    tags = {}
    #print words
    for key, word in words.items():
        if key[1] in tags:
            tags[key[1]]+= word
        else:
            tags[key[1]]=word

    return tags


def get_counter_words(list_sentences, en_stop):
    list_words = []
    for sentence in list_sentences:
        list_words += [token for token in sentence.split(" ") if token not in en_stop]
    words_to_tag = [word for word in list_words if word is not ""]
    tagged = nltk.pos_tag(words_to_tag)
    no_nouns = [word for word in tagged if 'VBP' in word[1]]
    #print no_nouns
    return Counter(no_nouns)

en_stop = get_stop_words('en')

list_premises = []
list_hypothesis = []

study_words = True
if study_words:

    with open("wrong_pairs_15.txt") as wrong_pairs:
        for line in wrong_pairs:
            sentences = line.split("#")
            list_premises.append(sentences[0].rstrip(".").lower())
            list_hypothesis.append(sentences[1].strip("\n").rstrip(".").lower())

    print len(list_hypothesis), len(list_premises)
    tagged_words_counted = []
    for list_x in [list_premises, list_hypothesis]:
        tagged_words_counted=get_counter_words(list_x, en_stop)

        common = tagged_words_counted.most_common(50)
        tags = count_tags(tagged_words_counted)
        sorted_tags = sorted(tags.items(), key=operator.itemgetter(1))
        print sorted_tags[-7:]

        f=open("verbs_sentences.txt", "w")
        for c in common:
            print c[0][0], c[0][1], c[1]
            for p, h in zip(list_premises, list_hypothesis):

                if c[0][0] in p or c[0][0] in h:
                    f.write(p)
                    f.write(".")
                    f.write(h)
                    f.write("\n")

study_other=False
if study_other:
    num_line = []
    list_expected=[]
    with open("predictions_15.txt") as wrong_pairs:
        #[ 0.68830132  0.25855646  0.0531422 ] [0 1 0]

        for nb_line, line in enumerate(wrong_pairs):
            l = line.split("] [")
            for i, x in enumerate(l):
                if i == 0:
                    out_list = x.strip("[").split(" ")
                    a = np.asarray([token for token in out_list if token is not ""])
                    if np.argmax(a) == 2:
                        #print a
                        #print nb_line
                        num_line.append(nb_line)
                        list_expected.append(np.asarray(l[1].rstrip("]\n").split(" ")))

    f=open("cont_15.txt", 'w')
    i=0
    with open("wrong_pairs_15.txt") as wrong_pairs:
        for nb_line, line in enumerate(wrong_pairs):
            if nb_line in num_line:
                f.write(line)
                f.write(list_expected[i])
                f.write("\n")
                i+=1
    f.close()
