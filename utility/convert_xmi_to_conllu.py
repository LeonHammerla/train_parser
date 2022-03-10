import pathlib
import sys
import os
from typing import Tuple, List, Optional
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


def create_conllu_files_for_data(save_corpus: bool = True,
                                 save_document: bool = False,
                                 save_sent: bool = False):
    """
    Main Function for creating a conllu files for every subfolder in data that is listed
    in dirs-variable.
    :return:
    """
    # --> Creating Log-file:
    logfile = os.path.join(ROOT_DIR, "data", "conllu_files", "log.txt")
    with open(logfile, "w") as f:
        f.write("This is a automatically created log-file:\n\n")
    # --> Getting Typesystem:
    typesystem = load_typesystem_from_path(os.path.join(ROOT_DIR, "data/TypeSystem.xml"))
    # --> Different directories:
    dirs = ["27294", "27329", "27624", "28451"]
    ids = ["tiger", "ud", "ud", "tiger"]
    # --> Finding all Paths:
    paths_for_corpora = find_all_caspaths_per_corpus(dirs)
    # --> pbars:
    main_bar = tqdm(total=4, desc="Converting Corpora", leave=True, position=0)
    side_bar1 = tqdm(total=0, desc="saving corpus-wise", leave=True, position=1)
    side_bar2 = tqdm(total=0, desc="saving doc-wise", leave=True, position=2)
    side_bar3 = tqdm(total=0, desc="saving sent-wise", leave=True, position=3)
    # --> Main-Loop for all corpora extracting and saving:
    for corpus_idx in range(0, len(paths_for_corpora)):
        sent_id = 0
        tokenlists = []
        # --> Collecting all Lists of Tokenlists for each xmi-path:
        for file_path in paths_for_corpora[corpus_idx]:
            res, sent_id = extract_cas_information(ids[corpus_idx], file_path, typesystem, logfile, sent_id)
            tokenlists.append(res)

        if save_corpus:
            side_bar1.total = sum([len(sub) for sub in tokenlists])
            pathlib.Path(os.path.join(ROOT_DIR, f"data/conllu_files/{dirs[corpus_idx]}/corpus")).mkdir(parents=True, exist_ok=True)
            # --> Saving as whole conllu file:
            with open(os.path.join(ROOT_DIR, f"data/conllu_files/{dirs[corpus_idx]}/corpus/{dirs[corpus_idx]}.conllu"), "w") as f:
                for compiled_document in tokenlists:
                    for compiled_sentence in compiled_document:
                        f.write(compiled_sentence.serialize())
                        side_bar1.update(1)

        if save_document:
            side_bar2.total = sum([len(sub) for sub in tokenlists])
            pathlib.Path(os.path.join(ROOT_DIR, f"data/conllu_files/{dirs[corpus_idx]}/document")).mkdir(parents=True, exist_ok=True)
            # --> Saving document-wise:
            for c, compiled_document in enumerate(tokenlists):
                with open(os.path.join(ROOT_DIR, f"data/conllu_files/{dirs[corpus_idx]}/document/{c}.conllu"), "w") as f:
                    for compiled_sentence in compiled_document:
                        f.write(compiled_sentence.serialize())
                        side_bar2.update(1)

        if save_sent:
            side_bar3.total = sum([len(sub) for sub in tokenlists])
            pathlib.Path(os.path.join(ROOT_DIR, f"data/conllu_files/{dirs[corpus_idx]}/sent")).mkdir(parents=True, exist_ok=True)
            # --> Saving sent-wise:
            c = 0
            for compiled_document in tokenlists:
                for compiled_sentence in compiled_document:
                    with open(os.path.join(ROOT_DIR, f"data/conllu_files/{dirs[corpus_idx]}/sent/{c}.conllu"), "w") as f:
                        f.write(compiled_sentence.serialize())
                        side_bar3.update(1)
                    c += 1
        # --> Handling bars:
        main_bar.update(1)
        side_bar1.n = 0
        side_bar2.n = 0
        side_bar3.n = 0
        side_bar1.total = 0
        side_bar2.total = 0
        side_bar3.total = 0


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
                            logfile: str,
                            sent_id: int = 0) -> Tuple[Optional[List[TokenList]], int]:
    """
    Function returns a list of TokenLists. Every TokenList is one sentence from the cas
    object.

    :param logfile:
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
            # --> Getting Status Annotation:
            status_annotation = view.select_covered("org.texttechnologylab.annotation.administration.AnnotationStatus", sentence)
            correct_status = True
            try:
                assert len(status_annotation) == 1
            except:
                correct_status = False
                with open(logfile, "a") as f:
                    if len(status_annotation) > 1:
                        f.write(f"{caspath}: Error--More than one status-annotations for sent\n")
                    else:
                        f.write(f"{caspath}: Error--Less than one status-annotations for sent\n")
            # --> Checking if Sentence is "Processed":
            if correct_status and status_annotation[0]["status"] == "Processed":
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
                        if len(upos) == len(xpos) == len(heads) == len(deprels) == len(token) == 0:
                            with open(logfile, "a") as f:
                                f.write(f"{caspath}: Error--Sentence-length is zero\n")
                        else:
                            print(f"============{caspath}=================")
                            print(f"Length: upos:{len(upos)}, xpos:{len(xpos)},heads:{len(heads)},deprels:{len(deprels)},token:{len(token)}")
                            print(text)
                            print("========================================================================================================")
                            with open(logfile, "a") as f:
                                f.write(f"{caspath}: Error--Sentence-has uneven annotations?\n")

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
                    with open(logfile, "a") as f:
                        f.write(f"{caspath}: Error--{e}")

        return cas_tokenlist_list, c



if __name__ == '__main__':
    create_conllu_files_for_data(save_corpus=True,
                                 save_sent=True,
                                 save_document=True)
