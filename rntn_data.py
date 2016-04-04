#!/usr/bin/env python3

from phrase_tree import *
from rntn_dictionary import *
import csv
import configparser


def main(args):
    from docopt import docopt
    stree_name = 'STree.txt'
    sostr_name = 'SOStr.txt'
    sent_name = 'datasetSentences.txt'
    args = docopt(main.__doc__, argv=args)
    if args['--stree']:
        stree_name = args['--stree']
    if args['--sostr']:
        sostr_name = args['--sostr']
    if args['--sentences']:
        sent_name = args['--sentences']
    return load_trees(sostr_name, stree_name, sent_name)

def read_config(name='rntn.config'):
    config = configparser.ConfigParser()
    config.read(name)
    return config

def load_trees(sostr_name, stree_name, sent_name):
    with open(sent_name, 'r') as f_sent:
        csv.register_dialect('tabsep', delimiter='\t', quoting=csv.QUOTE_NONE)
        reader = csv.DictReader(f_sent, dialect='tabsep')
        sentences = {(int(row['sentence_index'])-1): row['sentence'] for row in reader}
    with open(sostr_name, 'r') as f_sostr:
        sostrs = [line.strip().split('|') for line in list(f_sostr)]
    with open(stree_name, 'r') as f_stree:
        strees = [[int(x) for x in cleaned] for cleaned in
                  [line.strip().split('|') for line in list(f_stree)]]
    tree_data = zip(sostrs, strees)
    return SentenceTree.load_trees(tree_data, sentences)


def load_splits(splits_name, trees):
    trains, valids, tests = [], [], []
    splits_map = {'1': trains, '2': tests, '3': valids}
    with open(splits_name, 'r') as f_splits:
        reader = csv.DictReader(f_splits)
        for row in reader:
            splits_map[row['splitset_label']].append(trees[int(row['sentence_index'])-1])
    return trains, valids, tests


def find_orphan_phrases(sentences, dsdict):
    unique_phrases = {tree.phrase for
                      tree in [item for
                               sublist in [sentence.nodes for
                                           sentence in sentences] for
                               item in sublist]}
    dict_phrases = {entry.phrase for entry in dsdict}
    orphan_phrases = {x for x in unique_phrases if x not in dict_phrases}
    unused_phrases = {x for x in dict_phrases if x not in unique_phrases}
    return orphan_phrases, unused_phrases


def load_dataset(dsdir=None, sostr_name='SOStr.txt', stree_name='STree.txt',
                 sentences_name='datasetSentences.txt', sentiment_name='sentiment_labels.txt',
                 dict_name='dictionary.txt', splits_name='datasetSplit.txt',
                 train_limit=None, test_limit=None, valid_limit=None):
    if dsdir:
        sostr_name = dsdir + '/' + sostr_name
        stree_name = dsdir + '/' + stree_name
        sentences_name = dsdir + '/' + sentences_name
        dict_name = dsdir + '/' + dict_name
        splits_name = dsdir + '/' + splits_name
        sentiment_name = dsdir + '/' + sentiment_name
    trees = load_trees(sostr_name, stree_name, sentences_name)
    sentences = SentenceSet(trees)
    dsdict = Dictionary.load(dict_name, sentiment_name)
    trains, valids, tests = load_splits(splits_name, sentences)
    return sentences, dsdict, trains[:train_limit], valids[:valid_limit], tests[:test_limit]

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
