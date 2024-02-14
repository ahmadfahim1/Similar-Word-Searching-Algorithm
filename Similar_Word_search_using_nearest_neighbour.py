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


#print(W)

while True:
    input_term = input("\nEnter three words (EXIT to break): ")
    if input_term == 'EXIT':
        break
    else:
        vectors[input_term]
        input_wv = np.array(vectors[input_term])
        original_wv = np.array(W)

        eucl_dis = np.linalg.norm((original_wv - input_wv), axis = 1)
        #print(eucl_dis)

        sort_index = np.argsort(eucl_dis)
        low_val = eucl_dis[sort_index[0:3]]
        low_val_index = sort_index[0:3]

        
        print("\n                               Word       Distance\n")
        print("---------------------------------------------------------\n")

        for x in low_val_index:
            print("%35s\t\t%f\n" % (ivocab[x], eucl_dis[x]))
