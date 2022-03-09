import pathlib
import sys
import os
from typing import Tuple, List, Optional, Callable, Dict, Union
from conllu import TokenList
from conllu.serializer import serialize_field
import cassis
from tqdm import tqdm
sys.path.append("/home/stud_homes/s5935481/uima_cassis/src")
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..')))
from cassis_utility.loading_utility import load_cas_from_xmi_dir, \
    load_cas_from_dir, \
    find_paths_in_dir, \
    load_typesystem_from_path, \
    load_cas_from_path

from cassis_utility.selecting_utility import select_sentences_from_cas, \
    select_dependencies_from_sentence, \
    select_tokens_of_sentence_from_cas, \
    select_pos_from_tokenlist


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))


def corpus_sent_info(punct: bool = True,
                     print_short_sents: bool = False,
                     exclude_empty_sents: bool = True) -> Dict[str, Dict[str, Union[List[str], Callable[[str], str], Dict[str, List[int]]]]]:
    """
    Function for collecting sent length info for different corpora.
    :param exclude_empty_sents:
    :param print_short_sents:
    :param punct:
    :return:
    """
    # --> Typesystem:
    typesystem = load_typesystem_from_path(os.path.join(ROOT_DIR, "data/TypeSystem.xml"))
    # --> Splitter Functions (for determining year of document--is in title):
    splitter_27329: Callable[[str], str] = lambda x: x.split("/")[-1].split(".")[0]
    splitter_28451: Callable[[str], str] = lambda x: "-".join([s for s in x.split("-") if s.isdigit()])
    # --> Different Corpora:
    corpus_info = {"27329": {"paths": [], "timebuckets": dict(), "splitter": splitter_27329},
                   "28451": {"paths": [], "timebuckets": dict(), "splitter": splitter_28451}}
    for dir in corpus_info.keys():
        corpus_info[dir]["paths"] = find_paths_in_dir(os.path.join(ROOT_DIR, f"data/{dir}"))[0]

    for dir in corpus_info.keys():

        for caspath in corpus_info[dir]["paths"]:
            # --> load cas:
            cas = load_cas_from_path(filepath=caspath,
                                     typesystem=typesystem)
            # --> If None path is invalid and Exception is raised:
            if cas is None:
                raise Exception(f"Cas File is invalid: {caspath}")
            view = cas.get_view('GoldStandard')
            # --> select sentences:
            sentences = select_sentences_from_cas(view)
            sentence_lengths = []
            for sentence in sentences:
                # --> Getting length of sentence:
                token = select_tokens_of_sentence_from_cas(cas=view,
                                                           sentence=sentence)
                if punct:
                    sent_length = len(token)
                else:
                    pos = select_pos_from_tokenlist(cas=view,
                                                    tokens=token)
                    assert len(pos) == len(token), print("something went wrong with pre-annotations")
                    # --> Morphological Annotation:
                    upos = [p["coarseValue"] for p in pos]
                    upos_without_punct = [p for p in upos if p != "PUNCT"]
                    sent_length = len(upos_without_punct)

                if print_short_sents:
                    if sent_length < 2:
                        print(f"sent-(punct={punct}): {sentence.get_covered_text()}")

                if exclude_empty_sents and sent_length == 0:
                    pass
                else:
                    sentence_lengths.append(sent_length)
            # --> Getting Year of document:
            cas_year_id = corpus_info[dir]["splitter"](caspath)
            # --> Filling results in result-dict:
            if cas_year_id in corpus_info[dir]["timebuckets"]:
                corpus_info[dir]["timebuckets"][cas_year_id].extend(sentence_lengths)
            else:
                corpus_info[dir]["timebuckets"][cas_year_id] = sentence_lengths

    return corpus_info


def extracting_all_sent_info(save_file_path: str,
                             print_short_sents: bool = False,
                             exclude_empty_sents: bool = True):
    """
    Function to save stats to file.
    :param exclude_empty_sents:
    :param print_short_sents:
    :param save_file_path:
    :return:
    """
    with open(save_file_path, "w") as f:
        # --> stats with puncts:
        corpus_info_with_punct = corpus_sent_info(punct=True,
                                                  print_short_sents=print_short_sents,
                                                  exclude_empty_sents=exclude_empty_sents)
        for dir in corpus_info_with_punct:
            f.write(f"{5*'='}{dir} with punctuations{5*'='}\n")
            for timeslice in corpus_info_with_punct[dir]["timebuckets"].keys():
                tmax = max(corpus_info_with_punct[dir]["timebuckets"][timeslice])
                tmin = min(corpus_info_with_punct[dir]["timebuckets"][timeslice])
                tmean = sum(corpus_info_with_punct[dir]["timebuckets"][timeslice]) / len(
                    corpus_info_with_punct[dir]["timebuckets"][timeslice])
                f.write(f"{timeslice}:\t\tmean={round(tmean, 2)}\tmax={tmax}\tmin={tmin}\n")
            f.write("\n")

        f.write("\n\n\n")
        # --> stats without puncts:
        corpus_info_with_punct = corpus_sent_info(punct=False,
                                                  print_short_sents=print_short_sents,
                                                  exclude_empty_sents=exclude_empty_sents)
        for dir in corpus_info_with_punct:
            f.write(f"{5 * '='}{dir} without punctuations{5 * '='}\n")
            for timeslice in corpus_info_with_punct[dir]["timebuckets"].keys():
                tmax = max(corpus_info_with_punct[dir]["timebuckets"][timeslice])
                tmin = min(corpus_info_with_punct[dir]["timebuckets"][timeslice])
                tmean = sum(corpus_info_with_punct[dir]["timebuckets"][timeslice]) / len(
                    corpus_info_with_punct[dir]["timebuckets"][timeslice])
                f.write(f"{timeslice}:\t\tmean={round(tmean, 2)}\tmax={tmax}\tmin={tmin}\n")
            f.write("\n")

def main():
    save_file_path = os.path.join(ROOT_DIR, "data", "eval_results", "sent_length_stats.txt")
    extracting_all_sent_info(save_file_path, True)


if __name__ == '__main__':
    main()

