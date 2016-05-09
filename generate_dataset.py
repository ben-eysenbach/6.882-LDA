import nltk
from collections import Counter
from nltk.corpus import stopwords, reuters
import random
from nltk.stem.porter import PorterStemmer
import re
import numpy as np
from tqdm import tqdm
np.random.seed(0)
random.seed(0)

STOP_WORDS = stopwords.words('english')

def generate_dataset():
    # mode is 'train' or 'test'

    print 'Loading documents'
    train_documents = get_documents('train')
    # test_documents = get_documents('test')

    print 'Computing set of good words'
    counts = Counter((w for d in train_documents for w in d))
    vocab_size = 1000
    sorted_counts = sorted(counts.values(), reverse=True)
    if vocab_size < len(sorted_counts):
        cutoff = sorted_counts[vocab_size]
    else:
        cutoff = len(sorted_counts)

    good_words = set()
    for (word, freq) in counts.items():
        if freq > cutoff:
            good_words.add(word)

    with open('datasets/nyt_vocab_1000.txt', 'wb') as f:
        f.write('\n'.join(good_words))

    print 'Subsampling documents'
    document_size = 1000
    np.random.shuffle(train_documents)
    train_documents = train_documents[:document_size]

    # test_documents = np.random.choice(test_documents, document_size, replace=False)

    print 'Filtering documents'
    filtered_train_documents = []
    for d in train_documents:
        filtered_d = [w for w in d if w in good_words]
        filtered_train_documents.append(' '.join(filtered_d))

    # filtered_test_documents = []
    # for d in test_documents:
    #     filtered_d = [w for w in d if w in good_words]
    #     filtered_test_documents.append(' '.join(filtered_d))

    print 'Writing documents'
    with open('datasets/nyt_train_1000.txt', 'wb') as f:
        f.write('\n'.join(filtered_train_documents))

    # with open('datasets/nyt_test.txt', 'wb') as f:
    #     f.write('\n'.join(filtered_test_documents))


def get_documents(mode):
    with open('datasets/nyt_music.txt') as f:
        documents = f.readlines()
    documents = [d.split() for d in documents]
    # id_list = reuters.fileids()
    # random.shuffle(id_list)
    # id_list = [name for name in id_list if name.startswith(mode)]
    # documents = [reuters.words(id) for id in id_list]
    documents = map(tokenize, documents)
    return documents

def tokenize(words):
    words = [w.lower() for w in words]
    regex = re.compile('[a-z]+');
    words = [w for w in words if w.isalpha()]
    # words = [PorterStemmer().stem(w) for w in words]
    words = [w for w in words if w not in STOP_WORDS]
    words = [w for w in words if len(w) >= 3]
    return words




if __name__ == '__main__':
    generate_dataset()
