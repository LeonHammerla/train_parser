import copy
import os
from typing import Dict, Tuple, List

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
PARTITIONING = {"27294": {"spacy_1895_corrected.txt.xmi": "old",
                          "spacy_1918_corrected.txt.xmi": "old",
                          "spacy_1933_corrected.txt.xmi": "new",
                          "spacy_1942_corrected.txt.xmi": "new",
                          "spacy_bucket1_corrected.txt.xmi": "unknown",
                          "spacy_bucket5_corrected.txt.xmi": "unknown",
                          "spacy_bucket10_corrected.txt.xmi": "unknown",
                          "spacy_bucket15_corrected.txt.xmi": "unknown",
                          "spacy_bucket19_corrected.txt.xmi": "unknown"
                          },
                "27329": {"1803.csv.extracted_tool_StepsParser_model_en_basic_xlmr.xmi": "old",
                          "1820.csv.extracted_tool_StepsParser_model_en_basic_xlmr.xmi": "old",
                          "1840.csv.extracted_tool_StepsParser_model_en_basic_xlmr.xmi": "old",
                          "1860.csv.extracted_tool_StepsParser_model_en_basic_xlmr.xmi": "old",
                          "1880.csv.extracted_tool_StepsParser_model_en_basic_xlmr.xmi": "old",
                          "1900.csv.extracted_tool_StepsParser_model_en_basic_xlmr.xmi": "old",
                          "1920.csv.extracted_tool_StepsParser_model_en_basic_xlmr.xmi": "old",
                          "1940.csv.extracted_tool_StepsParser_model_en_basic_xlmr.xmi": "new",
                          "1960.csv.extracted_tool_StepsParser_model_en_basic_xlmr.xmi": "new",
                          "1980.csv.extracted_tool_StepsParser_model_en_basic_xlmr.xmi": "new",
                          "2000.csv.extracted_tool_StepsParser_model_en_basic_xlmr.xmi": "new"
                          },
                "27624": {"spacy_1895_corrected.txt.xmi": "old",
                          "spacy_1918_corrected.txt.xmi": "old",
                          "spacy_1933_corrected.txt.xmi": "new",
                          "spacy_1942_corrected.txt.xmi": "new",
                          "spacy_bucket1_corrected.txt.xmi": "unknown",
                          "spacy_bucket5_corrected.txt.xmi": "unknown",
                          "spacy_bucket10_corrected.txt.xmi": "unknown",
                          "spacy_bucket15_corrected.txt.xmi": "unknown",
                          "spacy_bucket19_corrected.txt.xmi": "unknown"
                          },
                "28451": {"spacy_biofid-pre-1890-sel.txt.0.txt.xmi": "old",
                          "spacy_biofid-pre-1890-sel.txt.1.txt.xmi": "old",
                          "spacy_biofid-1890-1920-sel_new.txt.1.txt.xmi": "old",
                          "spacy_biofid-1890-1920-sel_new.txt.0.txt.xmi": "old",
                          "spacy_biofid-1920-1930-sel.txt.0.txt.xmi": "old",
                          "spacy_biofid-1920-1930-sel.txt.1.txt.xmi": "old",
                          "spacy_biofid-1930-1950-sel_new.txt.0.txt.xmi": "new",
                          "spacy_biofid-1930-1950-sel_new.txt.1.txt.xmi": "new",
                          "spacy_biofid-since-1950-sel.txt.1.txt.xmi": "new",
                          "spacy_biofid-since-1950-sel.txt.0.txt.xmi": "new"
                          }
                }


def load_dict_from_string(dict_str: str) -> Dict[str, float]:
    """
    function for loading result-dict from string.
    :param dict_str:
    :return:
    """
    res_dict = dict()
    result_list = dict_str.rstrip("%").split("% ")
    for i, res in enumerate(result_list):
        measure, value = res.split(": ")
        res_dict[measure] = float(value)
    return res_dict


def combine_dict_list(dict_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Combining dicts of form {'UCM': 0, 'LCM': 0, 'UAS': 0, 'LAS': 0}.
    :param dict_list:
    :return:
    """
    final_dict = {'UCM': 0, 'LCM': 0, 'UAS': 0, 'LAS': 0}
    length = len(dict_list)
    for dic in dict_list:
        for measure in dic:
            final_dict[measure] += dic[measure]

    for measure in final_dict:
        final_dict[measure] = final_dict[measure] / length

    return final_dict


def file_wise_stats(file_res: str) -> Tuple[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Function for returning results.
    :param file_res:
    :return:
    """
    results = dict()
    file_res = file_res.split("\n")
    file_name = file_res[1].split("/")[-1]

    # doc-based results -> Dict[str, Dict[str, float]]
    results["doc-base"] = dict()
    results["doc-base"][file_res[6].split("/")[-1]] = load_dict_from_string(file_res[7])
    results["doc-base"][file_res[8].split("/")[-1]] = load_dict_from_string(file_res[9])
    results["doc-base"][file_res[10].split("/")[-1]] = load_dict_from_string(file_res[11])

    # sent-based results -> Dict[str, Dict[str, float]]
    results["sent-base"] = dict()
    results["sent-base"][file_res[17].split("/")[-1]] = load_dict_from_string(file_res[18])
    results["sent-base"][file_res[19].split("/")[-1]] = load_dict_from_string(file_res[20])
    results["sent-base"][file_res[21].split("/")[-1]] = load_dict_from_string(file_res[22])

    return file_name, results


def dir_wise_stats(dir_id: str) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Function for getting dir_wise stats from txt files.
    Result is a dict with filename as keys and value is another dict with doc/sent-base as key and another dict
    as value with parser type as key and another dict as
    value with the stats.
    BSP: {'spacy_bucket5_corrected.txt.xmi': {'doc-base': {'biaffine_dep_de_tiger': {'UCM': 26.32, 'LCM': 21.05, 'UAS': 91.54, 'LAS': 89.05}, ...
    :param dir_id:
    :return:
    """
    file_wise_results = dict()
    with open(os.path.join(ROOT_DIR, "data", "eval_results_new", f"{dir_id}_results.txt"), "r") as f:
        content = "".join(f.readlines()[2:])
        content = content.split("\n\n\n\n\n")
        content = [con for con in content if con != ""]

        for file_res in content:
            file_name, results = file_wise_stats(file_res)
            file_wise_results[file_name] = results
    return file_wise_results


def new_stats():
    new_stat = dict()
    dirs = ["27294", "27329", "27624", "28451"]

    # loading stats:
    for d in dirs:
        new_stat[d] = dir_wise_stats(d)

    # dir-wise-stats:
    dir_stats = {"27294": {"old": {"doc-base": dict(), "sent-base": dict()},
                           "new": {"doc-base": dict(), "sent-base": dict()},
                           "unknown": {"doc-base": dict(), "sent-base": dict()}},
                 "27329": {"old": {"doc-base": dict(), "sent-base": dict()},
                           "new": {"doc-base": dict(), "sent-base": dict()},
                           "unknown": {"doc-base": dict(), "sent-base": dict()}},
                 "27624": {"old": {"doc-base": dict(), "sent-base": dict()},
                           "new": {"doc-base": dict(), "sent-base": dict()},
                           "unknown": {"doc-base": dict(), "sent-base": dict()}},
                 "28451": {"old": {"doc-base": dict(), "sent-base": dict()},
                           "new": {"doc-base": dict(), "sent-base": dict()},
                           "unknown": {"doc-base": dict(), "sent-base": dict()}}
                 }
    for d in dirs:
        for filename in new_stat[d]:
            for base in new_stat[d][filename]:
                for parsername in new_stat[d][filename][base]:
                    if parsername not in dir_stats[d][PARTITIONING[d][filename]][base]:
                        dir_stats[d][PARTITIONING[d][filename]][base][parsername] = [ new_stat[d][filename][base][parsername] ]
                    else:
                        dir_stats[d][PARTITIONING[d][filename]][base][parsername].append(new_stat[d][filename][base][parsername])

    total = {"old": {"doc-base": dict(), "sent-base": dict()},
             "new": {"doc-base": dict(), "sent-base": dict()},
             "unknown": {"doc-base": dict(), "sent-base": dict()}}

    for d in dirs:
        for age in dir_stats[d]:
            for base in dir_stats[d][age]:
                for parsername in dir_stats[d][age][base]:
                    dir_stats[d][age][base][parsername] = combine_dict_list(dir_stats[d][age][base][parsername])
                    if parsername not in total[age][base]:
                        total[age][base][parsername] = [ copy.copy(dir_stats[d][age][base][parsername]) ]
                    else:
                        total[age][base][parsername].append(copy.copy(dir_stats[d][age][base][parsername]))

    for age in total:
        for base in total[age]:
            for parsername in total[age][base]:
                total[age][base][parsername] = combine_dict_list(total[age][base][parsername])

    dir_stats["total"] = total

    for di in dir_stats:
        print("-" * 300)
        print(f"Corpus: {di}")
        print("-" * 300)
        for age in dir_stats[di]:
            print(f"{age} : ")
            for base in dir_stats[di][age]:
                print(f"\t{base} : ")
                for parsername in dir_stats[di][age][base]:
                    print(f"\t\t{parsername} : ")
                    print(f"\t\t\t{dir_stats[di][age][base][parsername]}")




if __name__ == "__main__":
    new_stats()


