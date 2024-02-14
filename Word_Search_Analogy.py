import numpy as np
import matplotlib.pyplot as plt


vocabulary_file='word_embeddings.txt'
# Read words
print('Read words...')
with open(vocabulary_file, 'r',encoding ="utf8") as f:
    words = [x.rstrip().split(' ')[0] for x in f.readlines()]

# Read word vectors
print('Read word vectors...')
with open(vocabulary_file, 'r', encoding = "utf8") as f:
    vectors = {}
    for line in f:
        vals = line.rstrip().split(' ')
        vectors[vals[0]] = [float(x) for x in vals[1:]]


vocab_size = len(words)
vocab = {w: idx for idx, w in enumerate(words)}
ivocab = {idx: w for idx, w in enumerate(words)}

# Vocabulary and inverse vocabulary (dict objects)
"""print('Vocabulary size')
print(len(vocab))
print(vocab['man'])
print(len(ivocab))
print(ivocab[10])"""

# W contains vectors for
#print('Vocabulary word vectors')
vector_dim = len(vectors[ivocab[0]])
W = np.zeros((vocab_size, vector_dim))
for word, v in vectors.items():
    if word == '<unk>':
        continue
    W[vocab[word], :] = v
#print(W.shape)
while True:
    give_in = input("\nEnter Input Analogy e.g. king-queen-prince (EXIT to break): ")
    if give_in == 'EXIT':
        break
    else:
        reformed_input = give_in.split("-")
        array_reformed_input = []
        for i in range(0,3):
            #print(reformed_input[i])
            #print(vectors[reformed_input[i]])
            array_reformed_input.append(np.array(vectors[reformed_input[i]])) 
            if i == 0:
                analogy_first_w = np.array(vectors[reformed_input[i]])
            elif i == 1:
                analogy_second_w = np.array(vectors[reformed_input[i]])
            elif i == 2:
                analogy_third_w = np.array(vectors[reformed_input[i]])

    #print(analogy_first_w)
    #print(analogy_second_w)
    #print(analogy_third_w)

    # we have to perform z = z + (y-x)
    analogy_final_w = analogy_third_w + (analogy_second_w - analogy_first_w)
    #print(analogy_final_w)

    original_wv = np.array(W)

    eucl_dis = np.linalg.norm((original_wv - analogy_final_w), axis = 1)
    #print(eucl_dis)

    sort_index = np.argsort(eucl_dis)

    low_val = eucl_dis[sort_index[0:]]
    low_val_index = sort_index[0:]
    #print(low_val)
    #print(low_val_index)

    #for x in low_val_index[0:5]:
        #print("%35s\t\t%f\n" % (ivocab[x], eucl_dis[x]))

    counter = 0
    for i in low_val_index:
        if counter < 2:
            if ivocab[i] != reformed_input[0] and ivocab[i] != reformed_input[1] and ivocab[i] != reformed_input[2]:
                print(ivocab[i])
                counter += 1
        else:
            break
        