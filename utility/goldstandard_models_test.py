import sys
import os
from typing import Tuple, List, Optional
from contextlib import redirect_stdout
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
        f.write(f"{110*'='}\n")
        f.write(f"{stype}-based\n")
        f.write(f"{110*'='}\n")
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



        f.write(f"{110*'='}\n")
        f.write("\n")



def eval_conllu(conllu_id: str,
                score_type: str = "corpus",
                directory: str = "/home/stud_homes/s5935481/work3/models"):
    """
    Function for evaluating conllu file.
    Score_type Options: "corpus", "document", "sent", "all"
    :param directory:
    :param score_type:
    :param conllu_id:
    :return:
    """


    model_conllu_dict = {"27294": {"conllu": {"corpus": os.path.join(ROOT_DIR, "data", "conllu_files", "27294", "corpus", "27294.conllu"),
                                              "document": os.path.join(ROOT_DIR, "data", "conllu_files", "27294", "document"),
                                              "sent": os.path.join(ROOT_DIR, "data", "conllu_files", "27294", "sent")},
                                   "models": [os.path.join(directory, model) for model in
                                              ['biaffine_dep_de_tiger', 'crf_dep_de_tiger', 'crf2o_dep_de_tiger']]},
                         "27329": {"conllu": {"corpus": os.path.join(ROOT_DIR, "data", "conllu_files", "27329", "corpus", "27329.conllu"),
                                              "document": os.path.join(ROOT_DIR, "data", "conllu_files", "27329", "document"),
                                              "sent": os.path.join(ROOT_DIR, "data", "conllu_files", "27329", "sent")},
                                   "models": [os.path.join(directory, model) for model in
                                              ['biaffine_dep_de_gum', 'crf_dep_de_gum', 'crf2o_dep_de_gum']]},
                         "27624": {"conllu": {"corpus": os.path.join(ROOT_DIR, "data", "conllu_files", "27624", "corpus", "27624.conllu"),
                                              "document": os.path.join(ROOT_DIR, "data", "conllu_files", "27624", "document"),
                                              "sent": os.path.join(ROOT_DIR, "data", "conllu_files", "27624", "sent")},
                                   "models": [os.path.join(directory, model) for model in
                                              ['biaffine_dep_de_gsd', 'crf_dep_de_gsd', 'crf2o_dep_de_gsd']]},
                         "28451": {"conllu": {"corpus": os.path.join(ROOT_DIR, "data", "conllu_files", "28451", "corpus", "28451.conllu"),
                                              "document": os.path.join(ROOT_DIR, "data", "conllu_files", "28451", "document"),
                                              "sent": os.path.join(ROOT_DIR, "data", "conllu_files", "28451", "sent")},
                                   "models": [os.path.join(directory, model) for model in
                                              ['biaffine_dep_de_tiger', 'crf_dep_de_tiger', 'crf2o_dep_de_tiger']]},
                         }

    if score_type == "all":
        types = ["corpus", "document", "sent"]
        modes = ["w", "a", "a"]
        for stype_idx in range(0, 3):
            eval_model(conllu_path=model_conllu_dict[conllu_id]["conllu"][types[stype_idx]],
                       model_names=model_conllu_dict[conllu_id]["models"],
                       save_path=os.path.join(ROOT_DIR, "data", "eval_results", f"{conllu_id}_results.txt"),
                       stype=types[stype_idx],
                       file_opening_option=modes[stype_idx])
    else:
        eval_model(conllu_path=model_conllu_dict[conllu_id]["conllu"][score_type],
                   model_names=model_conllu_dict[conllu_id]["models"],
                   save_path=os.path.join(ROOT_DIR, "data", "eval_results", f"{conllu_id}_results.txt"),
                   stype=score_type)


def main():
    """
    directory = "/home/stud_homes/s5935481/work3/models"

    models_names = ['biaffine_dep_de', 'biaffine_dep_gbert_de', 'biaffine_dep_roberta_de', 'crf_dep_de', 'crf2o_dep_de']    
    dirs = ["27294", "27329", "27624", "28451"]
    ids = ["tiger", "ud", "ud", "tiger"]


    model_conllu_dict = {"27294": {"conllu": os.path.join(ROOT_DIR, "data", "conllu_files", "27294.conllu"),
                                   "models": [os.path.join(directory, model) for model in ['biaffine_dep_de_tiger', 'crf_dep_de_tiger', 'crf2o_dep_de_tiger']]},
                         "27329": {"conllu": os.path.join(ROOT_DIR, "data", "conllu_files", "27329.conllu"),
                                   "models": [os.path.join(directory, model) for model in ['biaffine_dep_de_gum', 'crf_dep_de_gum', 'crf2o_dep_de_gum']]},
                         "27624": {"conllu": os.path.join(ROOT_DIR, "data", "conllu_files", "27624.conllu"),
                                   "models": [os.path.join(directory, model) for model in ['biaffine_dep_de_gsd', 'crf_dep_de_gsd', 'crf2o_dep_de_gsd']]},
                         "28451": {"conllu": os.path.join(ROOT_DIR, "data", "conllu_files", "28451.conllu"),
                                   "models": [os.path.join(directory, model) for model in ['biaffine_dep_de_tiger', 'crf_dep_de_tiger', 'crf2o_dep_de_tiger']]},
                         }

    # print(model_conllu_dict["27624"]["models"])
    eval_model(conllu_file=model_conllu_dict["27294"]["conllu"],
               model_names=model_conllu_dict["27294"]["models"],
               save_path=os.path.join(ROOT_DIR, "data", "eval_results", "27294_results.txt"))
    """
    pass


if __name__ == "__main__":
    dirs = ["27294", "27329", "27624", "28451"]
    ids = ["tiger", "ud", "ud", "tiger"]
    for dir in dirs:
        eval_conllu(dir,
                    score_type="all")

