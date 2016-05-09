from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore
import logging
from generate_dataset import tokenize
import os
import json
from tqdm import tqdm
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from sklearn.metrics.pairwise import pairwise_distances
import shutil
import pickle
from collections import Counter

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
NUM_TOPICS = 50

def KL(p, q):
    return max(0, 0.5 *  (mutual_info_score(p, q) + mutual_info_score(p, q)))

def prepare_data():
    # returns the corpus object required by learn
    # skips datasets/dspace/2481.json
    base = 'datasets/dspace'
    documents = []
    for filename in tqdm(os.listdir(base)):
        path = os.path.join(base, filename)
        with open(path) as f:
            d = json.load(f)
            abstract = d['abstract']
            if abstract is not None:
                words = tokenize(abstract.split())
                documents.append(words)

    dictionary = Dictionary(documents)
    dictionary.filter_extremes(no_below=5, no_above=0.3)
    dictionary.save('lda.dict')
    corpus = map(dictionary.doc2bow, documents)
    return corpus

def learn(corpus):
    dictionary = Dictionary.load('lda.dict')
    lda = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=NUM_TOPICS, chunksize=10000, passes=5)
    for line in lda.print_topics(NUM_TOPICS):
        print line
    lda.save('lda.gensim')


def get_topics():
    '''Computes distribution over topics for each abstract'''

    dictionary = Dictionary.load('lda.dict')
    lda = LdaMulticore.load('lda.gensim')

    base = 'datasets/dspace'
    new_base = 'datasets/dspace_topics'
    for filename in tqdm(os.listdir(base)):
        path = os.path.join(base, filename)
        with open(path, 'r') as f:
            d = json.load(f)
            abstract = d['abstract']
            if abstract is not None:
                words = tokenize(abstract.split())
                bow = dictionary.doc2bow(words)
                topics = lda.get_document_topics(bow, minimum_probability=0)
                topics = to_vec(topics)
                d['topics'] = topics
                new_path = os.path.join(new_base, filename)
                with open(new_path, 'w') as new_f:
                    json.dump(d, new_f)

def to_vec(topics):
    vec = np.zeros(NUM_TOPICS)
    for (index, prob) in topics:
        vec[index] = prob
    return list(vec)

def learn_embedding(precompute_metric=False, use_saved=False):
    base = 'datasets/dspace_topics'
    new_base = 'datasets/dspace_embeddings'
    # Delete previous saved embedding
    if os.path.exists(new_base):
        shutil.rmtree(new_base)
    os.makedirs(new_base)

    print 'Embedding: Extracting topics'
    # choose a random subset of documents
    filename_vec = os.listdir(base)
    subsample = 5000
    filename_vec = np.random.choice(filename_vec, subsample)
    topic_vec = []
    for filename in tqdm(filename_vec):
        path = os.path.join(base, filename)
        with open(path) as f:
            d = json.load(f)
            topics = d['topics']
            topic_vec.append(topics)

    print 'Embedding: Computing pairwise distances'
    if precompute_metric:
        if use_saved:
            with open('metric.npy') as f:
                metric = np.load(f)
        else:
            metric = pairwise_distances(np.array(topic_vec), metric=KL, n_jobs=-1)
            with open('metric.npy', 'w') as f:
                np.save(f, metric)

        print 'Embedding: Learning embedding'
        tsne = TSNE(n_iter=1000, verbose=10, metric='precomputed')
        y = tsne.fit_transform(metric)
    else:
        print 'Embedding: Learning embedding'
        tsne = TSNE(n_iter=1000, verbose=10)
        y = tsne.fit_transform(topic_vec)

    print 'Embedding: Saving embedding'
    for (index, filename) in tqdm(enumerate(filename_vec), total=len(filename_vec)):
        path = os.path.join(base, filename)
        with open(path, 'r') as f:
            d = json.load(f)
            d['embedding'] = list(y[index])
            new_path = os.path.join(new_base, filename)
            with open(new_path, 'w') as new_f:
                json.dump(d, new_f)


def get_dept_dict():
    if os.path.exists('datasets/dept_dict.pkl'):
        with open('datasets/dept_dict.pkl') as f:
            return pickle.load(f)
    print 'Getting department dictionary'
    base = 'datasets/dspace'
    dept_dict = {}
    for filename in tqdm(os.listdir(base)):
        path = os.path.join(base, filename)
        with open(path) as f:
            d = json.load(f)
            dept = d['dept']
            if dept not in dept_dict:
                dept_dict[dept] = len(dept_dict)
    with open('datasets/dept_dict.pkl', 'w') as f:
        pickle.dump(dept_dict, f)
    return dept_dict

def plot_embedding(rainbow=False):
    if rainbow:
        dept_dict = get_dept_dict()
        palette = sns.color_palette("muted", n_colors = len(dept_dict))

    base = 'datasets/dspace_embeddings'
    for filename in tqdm(os.listdir(base)):
        path = os.path.join(base, filename)
        with open(path) as f:
            d = json.load(f)
            (x, y) = d['embedding']
            if rainbow:
                color = palette[dept_dict[d.get('dept', 0)]]
            else:
                color = sns.color_palette("Blues_d", n_colors=1)[0]
            plt.plot(x, y, marker='o', color=color, markersize=3)
    plt.axis('off')
    if rainbow:
        plt.savefig('dspace_rainbow_embedding_kl.png', dpi=500)
    else:
        plt.savefig('dspace_embedding.png', dpi=500)

    plt.show()

def nearest_neighbor(xx, yy, k=10):
    base = 'datasets/dspace_embeddings'
    min_dist = float('inf')
    d_vec = []
    dist_vec = []
    for filename in tqdm(os.listdir(base)):
        path = os.path.join(base, filename)
        with open(path) as f:
            d = json.load(f)
            (x, y) = d['embedding']
            dist = (xx - x)**2 + (yy - y)**2
            if dist < min_dist:
                d_vec.append(d)
                dist_vec.append(dist)
                if len(dist_vec) > k:
                    index = np.argmax(dist_vec)
                    dist_vec.pop(index)
                    d_vec.pop(index)
                    min_dist = np.max(dist_vec)
            assert len(dist_vec) == len(d_vec)
    for d in d_vec:
        print d['title']

def top_by_topic(k=10):
    '''For each topic, show the documents with the most of this topic'''
    base = 'datasets/dspace_topics'

    d_vec = [[] for _ in xrange(NUM_TOPICS)]
    dist_vec = [[] for _ in xrange(NUM_TOPICS)]
    max_dist = [0 for _ in xrange(NUM_TOPICS)]

    for filename in tqdm(os.listdir(base)):
        path = os.path.join(base, filename)
        with open(path) as f:
            d = json.load(f)
            for topic in range(NUM_TOPICS):
                if d['topics'][topic] > max_dist[topic]:
                    d_vec[topic].append(d)
                    dist_vec[topic].append(d['topics'][topic])
                    if len(dist_vec[topic]) > k:
                        index = np.argmin(dist_vec[topic])
                        dist_vec[topic].pop(index)
                        d_vec[topic].pop(index)
                        max_dist[topic] = np.min(dist_vec[topic])
                assert len(dist_vec[topic]) == len(d_vec[topic])

    for topic in range(NUM_TOPICS):
        print 'Topic: %d' % topic
        for d in d_vec[topic]:
            print '\t' + d['title']


def topic_by_dept():
    '''For each department, show the average distribution over topics'''
    base = 'datasets/dspace_topics'

    topic_vec = dict()
    count_vec = dict()

    for filename in tqdm(os.listdir(base)):
        path = os.path.join(base, filename)
        with open(path) as f:
            d = json.load(f)
            dept = d['dept']
            if dept is not None:
                count_vec[dept] = count_vec.get(dept, 0) + 1
                topic_vec[dept] = topic_vec.get(dept, np.zeros(NUM_TOPICS)) + d['topics']

    for dept in topic_vec:
        print dept, topic_vec[dept] / float(count_vec[dept])



if __name__ == '__main__':
    print 'Preparing data'
    corpus = prepare_data()
    print 'Learning data'
    learn(corpus)
    print 'Getting topics'
    get_topics()
    print 'Learning embedding'
    learn_embedding()
    print 'Plotting plain'
    plot_embedding(rainbow=False)
    print 'Plotting rainbow'
    plot_embedding(rainbow=True)
