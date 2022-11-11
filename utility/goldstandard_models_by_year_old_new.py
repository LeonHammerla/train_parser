import os
from typing import Dict, Tuple

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


def dir_wise_stats(dir_id: str):
    file_wise_results = dict()
    with open(os.path.join(ROOT_DIR, "data", "eval_results_new", f"{dir_id}_results.txt"), "r") as f:
        content = "".join(f.readlines()[2:])
        content = content.split("\n\n\n\n\n")
        content = [con for con in content if con != ""]

        for file_res in content:
            file_name, results = file_wise_stats(file_res)
            file_wise_results[file_name] = results
    return file_wise_results

def
if __name__ == "__main__":
    dirs = ["27294", "27329", "27624", "28451"]
    x = dir_wise_stats(dirs[0])
    print(x)


