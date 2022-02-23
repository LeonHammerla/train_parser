import sys
import os
from typing import Tuple, List, Optional
from conllu import TokenList
from conllu.serializer import serialize_field
import cassis
#from io import open
from conllu import parse_incr
from tqdm import tqdm


sys.path.append("/home/stud_homes/s5935481/uima_cassis/src")
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..')))


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))


def read_in_tiger_corpus(rm_multirootsents: bool = True) -> List[TokenList]:
    """
    Function for reading tiger-corpus-native conllu file
    and return it as correctly formatted TokenLists.
    :return:
    """
    path = os.path.join(ROOT_DIR,
                        "data/conllu_files/tiger_release_aug07.corrected.16012013.conll09")
    with open(path, "r", encoding="utf-8") as f:
        # ==== List to store all TokenLists (one for each sentence) ====
        sents_in_conllu = []
        # --> setting var for counting sents and text:
        sent_id = -1
        text = ""
        root_count = 0
        multi_root_sent_count = 0
        # --> Creating TokenList for one conllu-Sentence:
        compiled_toks = TokenList()
        for line in tqdm(f):
            if line == "\n":
                if len(compiled_toks) != 0:
                    sent_id += 1
                    # --> Filling in Metadata:
                    compiled_toks.metadata = {"sent_id": str(sent_id), "text": text.rstrip()}
                    if rm_multirootsents and root_count > 1:
                        multi_root_sent_count += 1
                    else:
                        sents_in_conllu.append(compiled_toks)
                    # --> Clearing:
                    compiled_toks = TokenList()
                    text = ""
                    root_count = 0
            else:
                # --> Filling in Token Information for each token in sent:
                line = line.split("\t")

                if line[4][0] == "$":
                    deprel_label = "PUNCT"
                else:
                    deprel_label = line[10]

                if line[8] == "0":
                    root_count += 1

                compiled_toks.append({'id': line[0].split("_")[-1],
                                      'form': line[1],
                                      'lemma': line[2],
                                      'upostag': "_",
                                      'xpostag': line[4],
                                      'feats': line[6],
                                      'head': line[8],
                                      'deprel': deprel_label,
                                      'deps': '_',
                                      'misc': '_'})
                text += f"{line[1]} "


    print(f"n-Sents with multi-root: {multi_root_sent_count}")

    return sents_in_conllu


def export_tiger_corpus(rm_multirootsents: bool = True):
    """
    Write Tiger-Corpus train/dev/test set to conllu-files.
    :return:
    """
    # ==== Getting all TokenLists (1/sent) ====
    sents_in_conllu = read_in_tiger_corpus(rm_multirootsents=rm_multirootsents)
    sent_count = len(sents_in_conllu)
    print(f"{sent_count} Sentences are in the Tiger-Corpus.")

    # ==== Saving as train, test, dev set ====
    train = sents_in_conllu[:int(sent_count*0.7)]
    dev = sents_in_conllu[int(sent_count*0.7):int(sent_count*0.85)]
    test = sents_in_conllu[int(sent_count*0.85):]

    if rm_multirootsents:
        root = "singleroot"
    else:
        root = "multiroot"

    # --> train:
    with open(os.path.join(ROOT_DIR, f"data/conllu_files/tiger_train_{root}.conllu"), "w") as f:
        for compiled_sentence in train:
            f.write(compiled_sentence.serialize())
    # --> test:
    with open(os.path.join(ROOT_DIR, f"data/conllu_files/tiger_test_{root}.conllu"), "w") as f:
        for compiled_sentence in dev:
            f.write(compiled_sentence.serialize())
    # --> dev:
    with open(os.path.join(ROOT_DIR, f"data/conllu_files/tiger_dev_{root}.conllu"), "w") as f:
        for compiled_sentence in test:
            f.write(compiled_sentence.serialize())


export_tiger_corpus()