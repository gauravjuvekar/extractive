#!/usr/bin/env python
import SIF
import sent2vec

import nltk
import nltk.cluster
import scipy
import scipy.spatial
import scipy.spatial.distance
import pickle
import functools
import statistics
from collections import defaultdict

import numpy
numpy.set_printoptions(threshold=10)

import os

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

ENABLE = (
    'sif',
    # 's2v',
    )

PREFETCH = True

sif_db = SIF.data_io.setup_db('./data/sif.db')
s2v_model = sent2vec.Sent2vecModel()

if 's2v' in ENABLE:
    s2v_model.load_model('./data/s2v_wiki_unigrams.bin')


def sif_embeds(sent_list):
    idx_mat, weight_mat, data = SIF.data_io.prepare_data(sent_list, sif_db)
    params = SIF.params.params()
    params.rmpc = 1
    embedding = SIF.SIF_embedding.SIF_embedding(idx_mat,
                                                weight_mat,
                                                data,
                                                params)
    return list(embedding)

@functools.lru_cache()
def detok_sent(sent):
    detokenizer = nltk.tokenize.moses.MosesDetokenizer()
    return detokenizer.detokenize(sent, return_str=True)

# @functools.lru_cache()
# def detok_sent(sent):
    # return ' '.join(sent)


@functools.lru_cache()
def s2v_embed_wrapped(sent):
    return s2v_model.embed_sentence(sent)


def s2v_embeds(sents):
    return [s2v_embed_wrapped(detok_sent(tuple(sent))) for sent in sents]


def underscore_tokenize(sentence):
    return tuple(
        [word for word in
            nltk.tokenize.word_tokenize(sentence.replace('_', ' '))
            ])


def medoid(vectors, dist_metric):
    dist_matrix = scipy.spatial.distance.pdist(vectors, metric=dist_metric)
    dist_matrix = scipy.spatial.distance.squareform(dist_matrix)
    return numpy.argmin(dist_matrix.sum(axis=0))


def cluster_kmeans(sents, k, dist_func=nltk.cluster.euclidean_distance):
    embeds = [d['embed'] for d in sents]
    clusterer = nltk.cluster.kmeans.KMeansClusterer(k, dist_func)
    clusters = clusterer.cluster(embeds, True)
    grouped = defaultdict(list)
    for c_tag, sent in zip(clusters, sents):
        grouped[c_tag].append(sent)
    return list(grouped.values())


if __name__ == '__main__':
    import pprint
    if True:
        d = './data/datasets/tipster/body'
        sentences = []
        for fname in os.listdir(d):
            log.debug(fname)
            f = os.path.join(d, fname)
            with open(f) as f:
                for line in f:
                    line = line.strip('\n')
                    line = line.strip()
                    words = nltk.word_tokenize(line)
                    sentences.append({'orig': line,
                                      'words': words})

        embeds = sif_embeds([x['words'] for x in sentences])
        for i, e in enumerate(embeds):
            sentences[i]['embed'] = e
        clusters = cluster_kmeans(sentences, 3)
        representatives = []
        for cluster in clusters:
            representative = cluster[medoid(cluster, 'euclidean')]
            representatives.append(representative)

        with open('./output.txt', 'w') as f:
            for s in representatives:
                print(s['orig'], file=f)
