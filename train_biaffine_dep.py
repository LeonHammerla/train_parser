# -*- coding: utf-8 -*-

import argparse
import sys
sys.path.append("/home/stud_homes/s5935481/work3/parser/supar2")
from parsers.dep import BiaffineDependencyParser
from cmds.cmd import parse


def main():
    parser = argparse.ArgumentParser(description='Create Biaffine Dependency Parser.')
    parser.add_argument('--tree', action='store_true', help='whether to ensure well-formedness')
    parser.add_argument('--proj', action='store_true', help='whether to projectivize the data')
    parser.add_argument('--partial', action='store_true', help='whether partial annotation is included')
    parser.set_defaults(Parser=BiaffineDependencyParser)
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # train
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--feat', '-f', choices=['tag', 'char', 'elmo', 'bert'], nargs='+', help='features to use')
    subparser.add_argument('--build', '-b', action='store_true', help='whether to build the model first')
    subparser.add_argument('--checkpoint', action='store_true', help='whether to load a checkpoint to restore training')
    subparser.add_argument('--encoder', choices=['lstm', 'bert'], default='lstm', help='encoder to use')
    subparser.add_argument('--punct', action='store_true', help='whether to include punctuation')
    subparser.add_argument('--max-len', type=int, help='max length of the sentences')
    subparser.add_argument('--buckets', default=32, type=int, help='max num of buckets to use')

    # =====================
    subparser.add_argument('--train', default='/home/stud_homes/s5935481/work3/data/de_hdt-ud-train-ful.conllu', help='path to train file')
    subparser.add_argument('--dev', default='/home/stud_homes/s5935481/work3/data/de_hdt-ud-dev.conllu', help='path to dev file')
    subparser.add_argument('--test', default='/home/stud_homes/s5935481/work3/data/de_hdt-ud-test.conllu', help='path to test file')
    """
    subparser.add_argument('--train',
                           default='/home/stud_homes/s5935481/work3/treebanks/UD_German-GSD/de_gsd-ud-train.conllu',
                           help='path to train file')
    subparser.add_argument('--dev',
                           default='/home/stud_homes/s5935481/work3/treebanks/UD_German-GSD/de_gsd-ud-dev.conllu',
                           help='path to dev file')
    subparser.add_argument('--test',
                           default='/home/stud_homes/s5935481/work3/treebanks/UD_German-GSD/de_gsd-ud-test.conllu',
                           help='path to test file')
    """
    # =====================

    subparser.add_argument('--embed', default='/home/stud_homes/s5935481/work3/data/embeddings/lemma_kom.txt',
                           help='path to pretrained embeddings')
    subparser.add_argument('--unk', default='unk', help='unk token in pretrained embeddings')
    subparser.add_argument('--n-embed', default=100, type=int, help='dimension of embeddings')
    subparser.add_argument('--bert', default='bert-base-german-cased', help='which BERT model to use')
    # evaluate
    subparser = subparsers.add_parser('evaluate', help='Evaluate the specified parser and dataset.')
    subparser.add_argument('--punct', action='store_true', help='whether to include punctuation')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data',
                           default='/home/stud_homes/s5935481/work3/data/de_hdt-ud-test.conllu',
                           help='path to dataset')
    # predict
    subparser = subparsers.add_parser('predict', help='Use a trained parser to make predictions.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data',
                           default='/home/stud_homes/s5935481/work3/data/de_hdt-ud-test.conllu',
                           help='path to dataset')
    subparser.add_argument('--pred', default='/home/stud_homes/s5935481/work3/data/pred_hdt.conllu', help='path to predicted result')
    subparser.add_argument('--prob', action='store_true', help='whether to output probs')
    parse(parser)


if __name__ == "__main__":
    main()
