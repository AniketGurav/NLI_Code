
import numpy as np

def print_counters(c1, c2, c3, c4, c5, c6):
    total = c1+c2+c3+c4+c5+c6
    print "entailment neutral:", c1
    print "entailment contradiction:", c2
    print "neutral entailment :", c3
    print "neutral contradiction:", c4
    print "contradiction entailment:", c5
    print "contradiction neutral:", c6
    #print total
    print "wrong real entails:", (c1+c2)/float(total)
    print "wrong real neutrals:", (c3+c4)/float(total)
    print "wrong real contradictions:", (c5+c6)/float(total)
    print "wrong as neutrals", (c1+c6)/float(total)
    print "wrong as entailments", (c3+c5)/float(total)
    print "wrong as contradictions", (c2+c4)/float(total)

with open("w_predictions.txt") as wrong_pairs:
    #[ 0.68830132  0.25855646  0.0531422 ] [0 1 0]
    list_output = []
    list_expected = []
    for line in wrong_pairs:
        l = line.split("] [")
        for i, x in enumerate(l):
            if i == 0:
                out_list = x.strip("[").split(" ")
                list_output.append(np.asarray([token for token in out_list if token is not ""]))
            else:
                list_expected.append(np.asarray(x.rstrip("]\n").split(" ")))

# c_expected_output
c_ent_neu = 0
c_ent_cont = 0
c_neu_ent = 0
c_neu_cont = 0
c_cont_ent = 0
c_cont_neu = 0

correct = 0
for o, e in zip(list_output, list_expected):
    """
    remove neutral noise
    """
    if np.argmax(o) == 1:
        o[1]=0
        if np.argmax(o) == np.argmax(e):
            correct+=1
        else:
            if np.argmax(e) == 0:
                c_ent_cont+=1
            else:
                c_cont_ent+=1
    elif np.argmax(o) == 0:
        if np.argmax(e) ==1:
            c_neu_ent+=1
        else:
            c_cont_ent+=1
    else:#output is contradiction
        if np.argmax(e) ==0:
            c_ent_cont +=1
        else:
            c_neu_cont +=1

print correct

print_counters(c_ent_neu, c_ent_cont, c_neu_ent, c_neu_cont, c_cont_ent, c_cont_neu)
