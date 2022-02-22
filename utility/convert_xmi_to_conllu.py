import sys
import os
from typing import Tuple, List, Optional
from conllu import TokenList
from conllu.serializer import serialize_field
import cassis

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


def create_conllu_files_for_data():
    """
    Main Function for creating a conllu files for every subfolder in data that is listed
    in dirs-variable.
    :return:
    """
    # --> Getting Typesystem:
    typesystem = load_typesystem_from_path(os.path.join(ROOT_DIR, "data/TypeSystem.xml"))
    # --> Different directories:
    dirs = ["27294", "27329", "27624", "28451"]
    ids = ["tiger", "ud", "ud", "tiger"]
    # --> Finding all Paths:
    paths_for_corpora = find_all_caspaths_per_corpus(dirs)
    # --> Main-Loop for all corpora extracting and saving:
    for corpus_idx in range(0, len(paths_for_corpora)):
        sent_id = 0
        tokenlists = []
        # --> Collecting all Lists of Tokenlists for each xmi-path:
        for file_path in paths_for_corpora[corpus_idx]:
            res, sent_id = extract_cas_information(ids[corpus_idx], file_path, typesystem, sent_id)
            tokenlists.extend(res)
        # --> Saving as whole conllu file:
        with open(os.path.join(ROOT_DIR, f"data/conllu_files/{dirs[corpus_idx]}.conllu"), "w") as f:
            for compiled_sentence in tokenlists:
                f.write(compiled_sentence.serialize())


def find_all_caspaths_per_corpus(dirs: List[str]) -> Tuple[List[str], ...]:
    """
    Function finds all cas paths in given data-directory.
    Each tuple entry is one corpus.
    :return:
    """

    # ==== Getting Paths ====
    # --> Different Dirs for different corpora:
    path_list = []
    # --> Going through dirs:
    for dir in dirs:
        paths = find_paths_in_dir(os.path.join(ROOT_DIR, f"data/{dir}"))[0]
        path_list.append(paths)

    return tuple(path_list)


def extract_cas_information(corpus_id: str,
                            caspath: str,
                            typesystem: cassis.TypeSystem,
                            sent_id: int = 0) -> Tuple[Optional[List[TokenList]], int]:
    """
    Function returns a list of TokenLists. Every TokenList is one sentence from the cas
    object.

    :param sent_id:
    :param caspath:
    :param typesystem:
    :return:
    """
    # ==== Extracting Info ====
    cas_tokenlist_list = []
    # --> Loading cas:
    cas = load_cas_from_path(filepath=caspath,
                             typesystem=typesystem)
    # --> If None path is invalid and Exception is raised:
    if cas is None:
        raise Exception(f"Cas File is invalid: {caspath}")
    else:
        # --> Selecting Gold standard view:
        view = cas.get_view('GoldStandard')
        # --> Selecting sentences:
        sentences = select_sentences_from_cas(view)
        # --> Collecting all Information needed for each Sentence:
        c = sent_id
        for sentence in sentences:
            try:
                # --> Tokens:
                token = select_tokens_of_sentence_from_cas(cas=view,
                                                           sentence=sentence)
                token_indices = [idx for idx in range(1, len(token) + 1)]
                token_index_identifier = [(tok["begin"], tok["end"]) for tok in token]
                # --> Dependencies:
                dependencies = []
                dependencies_unordered = select_dependencies_from_sentence(cas=view,
                                                                           sentence=sentence)
                for tok in token:
                    for dep in dependencies_unordered:
                        if tok["begin"] == dep["Dependent"]["begin"] and tok["end"] == dep["Dependent"]["end"]:
                            dependencies.append(dep)
                            break
                # --> Part of Speech:
                try:
                    pos = select_pos_from_tokenlist(cas=view,
                                                    tokens=token)
                    # --> Morphological Annotation:
                    upos = [p["coarseValue"] for p in pos]
                    xpos = [p["PosValue"] for p in pos]
                except:
                    pos = ["_" for pp in token]
                    # --> Morphological Annotation:
                    upos = ["_" for p in pos]
                    xpos = ["_" for p in pos]
                # --> Metadata:
                text = sentence.get_covered_text().strip()
                sent_id = c
                # --> Syntactic Annotation:
                heads = []
                deprels = []
                for dep in dependencies:
                    deprel = dep["DependencyType"]
                    if deprel == "--":
                        if corpus_id == "tiger":
                            deprels.append("--")
                        else:
                            deprels.append("root")
                        heads.append(0)
                    else:
                        #if corpus_id != "tiger":
                        deprels.append(deprel)
                        heads.append(token_index_identifier.index((dep["Governor"]["begin"], dep["Governor"]["end"])) + 1)
                # --> Checking if everything makes sense, else sent gets discarded:
                if len(upos) == len(xpos) == len(heads) == len(deprels) == len(token) and text != "":
                    c += 1
                    keep = True
                else:
                    keep = False

                if keep:
                    # --> Creating TokenList for conllu:
                    compiled_toks = TokenList()
                    # --> Filling in Metadata:
                    compiled_toks.metadata = {"sent_id": str(sent_id), "text": text}
                    # --> Filling in Token Information for each token in sent:
                    for i in range(0, len(token)):
                        compiled_toks.append({'id': token_indices[i],
                                              'form': token[i].get_covered_text(),
                                              'lemma': '_',
                                              'upostag': upos[i],
                                              'xpostag': xpos[i],
                                              'feats': '_',
                                              'head': heads[i],
                                              'deprel': deprels[i],
                                              'deps': '_',
                                              'misc': '_'})

                    cas_tokenlist_list.append(compiled_toks)
            except Exception as e:
                print(e, caspath)

        return cas_tokenlist_list, c



if __name__ == '__main__':
    create_conllu_files_for_data()
