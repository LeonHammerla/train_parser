import sys
import os
from typing import Tuple, List, Optional
from contextlib import redirect_stdout
import supar
from tqdm import tqdm

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..')))


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))


def eval_model(conllu_file: str, model_names: [str], save_path: str):
    with open(save_path, 'w') as f:
        with redirect_stdout(f):
            for name in tqdm(model_names):
                model = supar.Parser.load(name)
                if 'crf_dep' in name or 'crf2o_dep' in name:
                    out = model.evaluate(conllu_file, verbose=True, prob=True, tree=True, proj=True, mbr=True)
                else:
                    out = model.evaluate(conllu_file, verbose=True, prob=True, tree=True, proj=True)


def main():

    directory = "/home/stud_homes/s5935481/work3/models"
    """
    models_names = ['biaffine_dep_de', 'biaffine_dep_gbert_de', 'biaffine_dep_roberta_de', 'crf_dep_de', 'crf2o_dep_de']    
    dirs = ["27294", "27329", "27624", "28451"]
    ids = ["tiger", "ud", "ud", "tiger"]
    """

    model_conllu_dict = {"27294": {"conllu": os.path.join(ROOT_DIR, "data", "conllu_files", "27294.conllu"),
                                   "models": [os.path.join(directory, model) for model in ['biaffine_dep_de_tiger', 'crf_dep_de_tiger', 'crf2o_dep_de_tiger']]},
                         "27329": {"conllu": os.path.join(ROOT_DIR, "data", "conllu_files", "27329.conllu"),
                                   "models": [os.path.join(directory, model) for model in ['biaffine_dep_de_tiger', 'crf_dep_de_tiger', 'crf2o_dep_de_tiger']]},
                         "27624": {"conllu": os.path.join(ROOT_DIR, "data", "conllu_files", "27624.conllu"),
                                   "models": [os.path.join(directory, model) for model in ['biaffine_dep_de_gsd', 'crf_dep_de_gsd', 'crf2o_dep_de_gsd']]},
                         "28451": {"conllu": os.path.join(ROOT_DIR, "data", "conllu_files", "28451.conllu"),
                                   "models": [os.path.join(directory, model) for model in ['biaffine_dep_de_tiger', 'crf_dep_de_tiger', 'crf2o_dep_de_tiger']]},
                         }

    print(model_conllu_dict["27624"]["models"])
    eval_model(conllu_file=model_conllu_dict["27624"]["conllu"],
               model_names=model_conllu_dict["27624"]["models"],
               save_path=os.path.join(ROOT_DIR, "data", "eval_results", "27624_results.txt"))


if __name__ == "__main__":
    main()