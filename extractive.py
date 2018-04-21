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

import numpy as np
np.set_printoptions(threshold=10)

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
    dist_matrix = scipy.spatial.distance.pdist(np.array(vectors),
                                               metric=dist_metric)
    dist_matrix = scipy.spatial.distance.squareform(dist_matrix)
    return np.argmin(dist_matrix.sum(axis=0))


def cluster_kmeans(sents, k, dist_func=nltk.cluster.euclidean_distance):
    embeds = [d['embed'] for d in sents]
    clusterer = nltk.cluster.kmeans.KMeansClusterer(k, dist_func)
    clusters = clusterer.cluster(embeds, True)
    grouped = defaultdict(list)
    for c_tag, sent in zip(clusters, sents):
        grouped[c_tag].append(sent)
    return list(grouped.values())


def get_summary(clusters, dist_metric):
    representatives = []
    for cluster in clusters:
        s_embeds = [d['embed'] for d in cluster]
        representative = cluster[medoid(s_embeds, dist_metric)]
        representatives.append(representative)
    return representatives


def process_file(inputdir, input_filename, output_dir, line_proc,
                 embed_func, cluster_dist_func, medoid_dist_metric):
    filepath = os.path.join(inputdir, input_filename)
    sentences = []
    with open(filepath) as f:
        for line in f:
            line = line_proc(line)
            words = nltk.word_tokenize(line)
            sentences.append({'orig': line, 'words': words})

    embeds = embed_func([x['words'] for x in sentences])
    for i, e in enumerate(embeds):
        sentences[i]['embed'] = e
    for k in range(1, 10):
        clusters = cluster_kmeans(sentences, k, dist_func=cluster_dist_func)
        representatives = get_summary(clusters, dist_metric=medoid_dist_metric)
        output_fname = os.path.join(output_dir, input_filename + '_K' + str(k))
        with open(output_fname, 'w') as f:
            for s in representatives:
                print(s['orig'], file=f)

def line_proc_strip_n(line):
    return line.strip('\n').strip()

def line_proc_strip_n_lower(line):
    return line_proc_strip_n(line).lower()

if __name__ == '__main__':
    import pprint
    if True:
        d = './data/datasets/tipster/body'
        for fname in os.listdir(d):
            log.debug(fname)
            output_dir = './output_sif_euclidean_OrigCase'
            os.mkdir(output_dir)
            process_file(d, fname, output_dir, sif_embeds, line_proc_strip_n,
                         nltk.cluster.euclidean_distance, 'euclidean')

            output_dir = './output_sif_euclidean_lowercase'
            os.mkdir(output_dir)
            process_file(d, fname, output_dir, sif_embeds,
                line_proc_strip_n_lower,
                nltk.cluster.euclidean_distance, 'euclidean')

            output_dir = './output_sif_cosine_OrigCase'
            os.mkdir(output_dir)
            process_file(d, fname, output_dir, sif_embeds,
                line_proc_strip_n,
                nltk.cluster.cosine_distance, 'cosine')

            output_dir = './output_sif_cosine_lowercase'
            os.mkdir(output_dir)
            process_file(d, fname, output_dir, sif_embeds,
                line_proc_strip_n_lower,
                nltk.cluster.cosine_distance, 'cosine')

            output_dir = './output_s2v_euclidean_OrigCase'
            os.mkdir(output_dir)
            process_file(d, fname, output_dir, s2v_embeds, line_proc_strip_n,
                         nltk.cluster.euclidean_distance, 'euclidean')

            output_dir = './output_s2v_euclidean_lowercase'
            os.mkdir(output_dir)
            process_file(d, fname, output_dir, s2v_embeds,
                line_proc_strip_n_lower,
                nltk.cluster.euclidean_distance, 'euclidean')

            output_dir = './output_s2v_cosine_OrigCase'
            os.mkdir(output_dir)
            process_file(d, fname, output_dir, s2v_embeds,
                line_proc_strip_n,
                nltk.cluster.cosine_distance, 'cosine')

            output_dir = './output_s2v_cosine_lowercase'
            os.mkdir(output_dir)
            process_file(d, fname, output_dir, s2v_embeds,
                line_proc_strip_n_lower,
                nltk.cluster.cosine_distance, 'cosine')
