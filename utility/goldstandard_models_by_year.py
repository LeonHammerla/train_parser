import sys
import os
from typing import Tuple, List, Optional
from contextlib import redirect_stdout

import pathlib
import supar
from tqdm import tqdm

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..')))


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))


def eval_model(conllu_path: str,
               model_names: [str],
               save_path: str,
               stype: str,
               file_opening_option: str = "w"):
    """

    :param conllu_path:
    :param model_names:
    :param save_path:
    :param stype:
    :param file_opening_option:
    :return:
    """

    with open(save_path, file_opening_option) as f:
        # ==== Writing evaluation-results to file ====
        # --> Header for Measurement used:
        f.write(f"{50*'='}\n")
        f.write(f"{stype}-based\n")
        f.write(f"{50*'='}\n")
        # --> Going through all related models:
        for name in tqdm(model_names):
            f.write(name + "\n")
            model = supar.Parser.load(name)
            # --> different args depending on model-type:
            if 'crf_dep' in name or 'crf2o_dep' in name:
                args = {"verbose": False, "prob": True, "tree": True, "proj": True, "mbr": True}
            else:
                args = {"verbose": False, "prob": True, "tree": True, "proj": True}


            # --> checking whether a directory or file is given:
            if os.path.isdir(conllu_path):
                # --> if directory: colelct all files:
                paths = [os.path.join(conllu_path, file_path) for file_path in os.listdir(conllu_path)]
                paths = [pp for pp in paths if os.path.isfile(pp)]
                result_dict = {"UCM": [], "LCM": [], "UAS": [], "LAS": []}
                # --> Going through files and collecting scores:
                for path in paths:
                    loss, metric = model.evaluate(path, **args)
                    result_dict["UCM"].append(metric.ucm)
                    result_dict["LCM"].append(metric.lcm)
                    result_dict["UAS"].append(metric.uas)
                    result_dict["LAS"].append(metric.las)
                # --> building average for all those files:
                for key in result_dict.keys():
                    result_dict[key] = sum(result_dict[key]) / len(result_dict[key])
                # --> returning result string
                s = f"UCM: {result_dict['UCM']:6.2%} LCM: {result_dict['LCM']:6.2%} "
                s += f"UAS: {result_dict['UAS']:6.2%} LAS: {result_dict['LAS']:6.2%}"

            else:
                # -_> short version for only one file given:
                loss, metric = model.evaluate(conllu_path, **args)
                s = f"UCM: {metric.ucm:6.2%} LCM: {metric.lcm:6.2%} "
                s += f"UAS: {metric.uas:6.2%} LAS: {metric.las:6.2%}"
            # --> writing results to file:
            f.write(s + "\n")

        f.write(f"{50*'='}\n")
        f.write("\n")



def eval_conllu(conllu_id: str,
                models_directory: str,
                conllu_dir_name: str):
    """
    :param stypes:
    :param conllu_dir_name:
    :param models_directory:
    :param conllu_id:
    :return:
    """
    pathlib.Path(os.path.join(ROOT_DIR, "data", "eval_results_new")).mkdir(parents=True, exist_ok=True)


    model_conllu_dict = {"27294": {"conllu_path": os.path.join(ROOT_DIR, "data", conllu_dir_name, "27294"),
                                   "models": [os.path.join(models_directory, model) for model in
                                              ['biaffine_dep_de_tiger', 'crf_dep_de_tiger', 'crf2o_dep_de_tiger']]},

                         "27329": {"conllu_path": os.path.join(ROOT_DIR, "data", conllu_dir_name, "27329"),
                                   "models": [os.path.join(models_directory, model) for model in
                                              ['biaffine_dep_de_gum', 'crf_dep_de_gum', 'crf2o_dep_de_gum']]},

                         "27624": {"conllu_path": os.path.join(ROOT_DIR, "data", conllu_dir_name, "27624"),
                                   "models": [os.path.join(models_directory, model) for model in
                                              ['biaffine_dep_de_gsd', 'crf_dep_de_gsd', 'crf2o_dep_de_gsd']]},

                         "28451": {"conllu_path": os.path.join(ROOT_DIR, "data", conllu_dir_name, "28451"),
                                   "models": [os.path.join(models_directory, model) for model in
                                              ['biaffine_dep_de_tiger', 'crf_dep_de_tiger', 'crf2o_dep_de_tiger']]},
                         }

    paths = [os.path.join(model_conllu_dict[conllu_id]["conllu_path"],
                          file_path) for file_path in os.listdir(model_conllu_dict[conllu_id]["conllu_path"])]

    paths = [pp for pp in paths if os.path.isdir(pp)]
    with open(os.path.join(ROOT_DIR, "data", "eval_results_new", f"{conllu_id}_results.txt"), "w") as f:
        f.write(f"Evaluation of {conllu_id}\n\n")

    for p in paths:
        with open(os.path.join(ROOT_DIR, "data", "eval_results_new", f"{conllu_id}_results.txt"), "a") as f:
            f.write(f"{100 * '='}\n")
            f.write(f"File: {p}\n")
            f.write(f"{100 * '='}\n")


        eval_model(conllu_path=(p + ".conllu"),
                    model_names=model_conllu_dict[conllu_id]["models"],
                    save_path=os.path.join(ROOT_DIR, "data", "eval_results_new", f"{conllu_id}_results.txt"),
                    stype="Document",
                    file_opening_option="a")


        eval_model(conllu_path=p,
                    model_names=model_conllu_dict[conllu_id]["models"],
                    save_path=os.path.join(ROOT_DIR, "data", "eval_results_new", f"{conllu_id}_results.txt"),
                    stype="Sentence",
                    file_opening_option="a")

        with open(os.path.join(ROOT_DIR, "data", "eval_results_new", f"{conllu_id}_results.txt"), "a") as f:
            f.write(f"\n\n\n")


if __name__ == "__main__":
    dirs = ["27294", "27329", "27624", "28451"]
    ids = ["tiger", "ud", "ud", "tiger"]
    for dir in dirs:
        eval_conllu(conllu_id=dir,
                    models_directory="/resources/public/hammerla/parser/models",
                    conllu_dir_name="conllu_files_new")

