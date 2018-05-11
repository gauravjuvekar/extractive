#!/usr/bin/env python
import SIF
import sent2vec

import nltk
import nltk.cluster
import nltk.tokenize
import nltk.tokenize.moses
import scipy
import scipy.spatial
import scipy.spatial.distance
import pickle
import functools
import statistics
from collections import defaultdict

import numpy as np
np.set_printoptions(threshold=10)
np.seterr(all='raise')

import os

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

ENABLE = (
    'sif',
    's2v',
    )

PREFETCH = True

sif_db = SIF.data_io.setup_db('./data/sif.db')
s2v_model = sent2vec.Sent2vecModel()

if 's2v' in ENABLE:
    s2v_model.load_model('./data/sent2vec_wiki_unigrams.bin')


def sif_embeds(sent_list):
    idx_mat, weight_mat, data = SIF.data_io.prepare_data(sent_list, sif_db)
    params = SIF.params.params()
    params.rmpc = 1
    embedding = SIF.SIF_embedding.SIF_embedding(idx_mat,
                                                weight_mat,
                                                data,
                                                params)
    return list(embedding)


def sif_embeds_nopcr(sent_list):
    idx_mat, weight_mat, data = SIF.data_io.prepare_data(sent_list, sif_db)
    params = SIF.params.params()
    params.rmpc = 0
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
    clusterer = nltk.cluster.kmeans.KMeansClusterer(k, dist_func,
                                                    avoid_empty_clusters=True)
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


def process_file(inputdir, input_filename, output_dir, embed_func,
                 line_proc, cluster_dist_func, medoid_dist_metric):
    filepath = os.path.join(inputdir, input_filename)
    sentences = []
    with open(filepath, encoding='latin1') as f:
        for sent in nltk.sent_tokenize(f.read()):
            words = nltk.word_tokenize(sent)
            sentences.append({'orig': sent, 'words': words})

    embeds = embed_func([x['words'] for x in sentences])
    s2 = []
    for i, e in enumerate(embeds):
        if not np.any(e):
            # Remove all zero vectors because they occur for some reason
            continue
        else:
            sentences[i]['embed'] = e
            s2.append(sentences[i])
    sentences = s2
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
        os.makedirs('./output', exist_ok=True)
        os.makedirs('./output/opinosis', exist_ok=True)
        os.makedirs('./output/opinosis/sif', exist_ok=True)
        os.makedirs('./output/opinosis/s2v', exist_ok=True)
        os.makedirs('./output/opinosis/sifnopcr', exist_ok=True)
        d = './data/opinosis/topics'
        for fname in os.listdir(d):
            log.debug(fname)
            output_dir = './output/opinosis/sif/'
            try:
                os.mkdir(output_dir)
            except FileExistsError:
                pass
            process_file(d, fname, output_dir, sif_embeds, line_proc_strip_n,
                         nltk.cluster.euclidean_distance, 'euclidean')

            output_dir = './output/opinosis/sif/'
            try:
                os.mkdir(output_dir)
            except FileExistsError:
                pass
            process_file(d, fname, output_dir, sif_embeds,
                line_proc_strip_n,
                nltk.cluster.cosine_distance, 'cosine')

            output_dir = './output/opinosis/s2v/'
            try:
                os.mkdir(output_dir)
            except FileExistsError:
                pass
            process_file(d, fname, output_dir, s2v_embeds, line_proc_strip_n,
                         nltk.cluster.euclidean_distance, 'euclidean')

            output_dir = './output/opinosis/s2v/'
            try:
                os.mkdir(output_dir)
            except FileExistsError:
                pass
            process_file(d, fname, output_dir, s2v_embeds,
                line_proc_strip_n,
                nltk.cluster.cosine_distance, 'cosine')

            output_dir = './output/opinosis/sifnopcr/'
            try:
                os.mkdir(output_dir)
            except FileExistsError:
                pass
            process_file(d, fname, output_dir, sif_embeds_nopcr, line_proc_strip_n,
                         nltk.cluster.euclidean_distance, 'euclidean')

            output_dir = './output/opinosis/sifnopcr/'
            try:
                os.mkdir(output_dir)
            except FileExistsError:
                pass
            process_file(d, fname, output_dir, sif_embeds,
                line_proc_strip_n,
                nltk.cluster.cosine_distance, 'cosine')



    if True:
        os.makedirs('./output', exist_ok=True)
        os.makedirs('./output/cmplg', exist_ok=True)
        os.makedirs('./output/cmplg/sif', exist_ok=True)
        os.makedirs('./output/cmplg/s2v', exist_ok=True)
        os.makedirs('./output/cmplg/sifnopcr', exist_ok=True)
        d = './data/cmplg-xml/bodies/'
        for fname in os.listdir(d):
            log.debug(fname)
            output_dir = './output/cmplg/sif/'
            try:
                os.mkdir(output_dir)
            except FileExistsError:
                pass
            process_file(d, fname, output_dir, sif_embeds, line_proc_strip_n,
                         nltk.cluster.euclidean_distance, 'euclidean')

            output_dir = './output/cmplg/sif/'
            try:
                os.mkdir(output_dir)
            except FileExistsError:
                pass
            process_file(d, fname, output_dir, sif_embeds,
                line_proc_strip_n,
                nltk.cluster.cosine_distance, 'cosine')

            output_dir = './output/cmplg/s2v/'
            try:
                os.mkdir(output_dir)
            except FileExistsError:
                pass
            process_file(d, fname, output_dir, s2v_embeds, line_proc_strip_n,
                         nltk.cluster.euclidean_distance, 'euclidean')

            output_dir = './output/cmplg/s2v/'
            try:
                os.mkdir(output_dir)
            except FileExistsError:
                pass
            process_file(d, fname, output_dir, s2v_embeds,
                line_proc_strip_n,
                nltk.cluster.cosine_distance, 'cosine')

            output_dir = './output/cmplg/sifnopcr/'
            try:
                os.mkdir(output_dir)
            except FileExistsError:
                pass
            process_file(d, fname, output_dir, sif_embeds_nopcr, line_proc_strip_n,
                         nltk.cluster.euclidean_distance, 'euclidean')

            output_dir = './output/cmplg/sifnopcr/'
            try:
                os.mkdir(output_dir)
            except FileExistsError:
                pass
            process_file(d, fname, output_dir, sif_embeds,
                line_proc_strip_n,
                nltk.cluster.cosine_distance, 'cosine')
