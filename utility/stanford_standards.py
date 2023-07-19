"""
import stanfordnlp
def get_dependencies(obj):
    res = []
    for dep_edge in obj.dependencies:
        res.append((dep_edge[2].text, dep_edge[0].index, dep_edge[1]))
    return res
stanfordnlp.download('en')   # This downloads the English models for the neural pipeline
nlp = stanfordnlp.Pipeline() # This sets up a default neural pipeline in English
doc = nlp("These are three distinct conditions of service, but the same man may engage to fulfil them all three.")
# stanfordnlp.Document.conll_file
# doc.sentences[0].print_dependencies()
a = get_dependencies(doc.sentences[0])
"""
import copy
from typing import List, Tuple, Union, Dict
from conllu import parse_incr
import conllu
import stanza
from abc import ABC, abstractmethod
from os import listdir
from os.path import isfile, join
from os.path import realpath
from tqdm import tqdm


PARTITIONING_IM = {"27294": {"spacy_1895_corrected.txt.xmi": "old",
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
PARTITIONING = dict()
for key in PARTITIONING_IM.keys():
    PARTITIONING[key] = dict()
    for key2 in PARTITIONING_IM[key].keys():
        PARTITIONING[key][f"{key2}.conllu"] = PARTITIONING_IM[key][key2]


class StanfordDParser:
    def __init__(self, lang: str = "en"):
        self.nlp = stanza.Pipeline(lang=lang,
                                   processors='tokenize, mwt, pos, lemma, depparse',
                                   tokenize_pretokenized=True,
                                   tokenize_no_ssplit=True)

    def tuple_format(self, sent: List[str]) -> List[Tuple[str, int, str]]:
        anno_sent = self.nlp([sent]).sentences[0]
        res = []
        for tok in anno_sent.words:
            res.append((tok.text, tok.head, tok.deprel))
        return res

    def __call__(self, sent: List[str], *args, **kwargs) -> Tuple[List[str], List[int]]:
        anno_sent = self.nlp([sent]).sentences[0]
        heads = []
        deprels = []
        for tok in anno_sent.words:
            heads.append(tok.head)
            deprels.append(tok.deprel)
        return deprels, heads


class ConlluParser(ABC):
    @staticmethod
    def load(file_path: str) -> List[conllu.TokenList]:
        with open(file_path, "r", encoding="utf-8") as cf:
            sentences = list(parse_incr(cf))
        return sentences

    @staticmethod
    def tuple_format(tokl: conllu.TokenList) -> List[Tuple[str, int, str]]:
        res = []
        for tok in tokl:
            res.append((tok["form"], tok["head"], tok["deprel"]))
        return res

    @staticmethod
    def extract(tokl: conllu.TokenList) -> Tuple[List[str], List[str], List[int]]:
        heads = []
        deprels = []
        texts = []
        for tok in tokl:
            heads.append(tok["head"])
            deprels.append(tok["deprel"])
            texts.append(tok["form"])
        return texts, deprels, heads


class Metric:
    def __init__(self):
        # UCM LCM
        self.n_sent = 0.0
        self.n_tok = 0.0

        # LAS UAS
        self.uas_c = 0.0
        self.las_c = 0.0

        # UCM LCM
        self.ucm_c = 0.0
        self.lcm_c = 0.0

    def calc(self,
             deprels: List[str],
             heads: List[int],
             pred_deprels: List[str],
             pred_heads: List[int]):
        try:
            assert len(deprels) == len(heads) == len(pred_deprels) == len(pred_heads)
            for i in range(len(deprels)):
                if heads[i] == pred_heads[i]:
                    self.uas_c += 1
                    if deprels[i] == pred_deprels[i]:
                        self.las_c += 1
                self.n_tok += 1

            self.n_sent += 1
            if heads == pred_heads:
                self.ucm_c += 1
                if deprels == pred_deprels:
                    self.lcm_c += 1
        except:
            print("Something went wrong!")

    def clr(self):
        self.n_sent = 0.0
        self.n_tok = 0.0

        self.uas_c = 0.0
        self.las_c = 0.0

        self.ucm_c = 0.0
        self.lcm_c = 0.0

    def score(self) -> Dict[str, float]:
        res = {"uas": round(self.uas_c / self.n_tok, 4),
               "las": round(self.las_c / self.n_tok, 4),
               "ucm": round(self.ucm_c / self.n_sent, 4),
               "lcm": round(self.lcm_c / self.n_sent, 4)}
        self.clr()
        return res

    @staticmethod
    def avg_dicts(dcl: List[Dict[str, float]]) -> Dict[str, float]:
        n = len(dcl)
        res = {"uas": 0.0,
               "las": 0.0,
               "ucm": 0.0,
               "lcm": 0.0}
        if n == 0:
            return res
        for dc in dcl:
            res["uas"] += dc["uas"]
            res["las"] += dc["las"]
            res["ucm"] += dc["ucm"]
            res["lcm"] += dc["lcm"]

        res["uas"] = round(res["uas"] / n, 4)
        res["las"] = round(res["las"] / n, 4)
        res["ucm"] = round(res["ucm"] / n, 4)
        res["lcm"] = round(res["lcm"] / n, 4)
        return res

    def __call__(self, inp: List[Tuple[list, list, list, list]], *args, **kwargs) -> Tuple[
        Dict[str, float], Dict[str, float]]:
        # Doc-Based
        for tup in inp:
            self.calc(*tup)
        doc_res = self.score()

        # Sent-Based
        res_dicts = []
        for tup in inp:
            self.calc(*tup)
            res_dicts.append(self.score())
        sent_res = self.avg_dicts(res_dicts)

        return doc_res, sent_res


class EvalConlluFile(ABC):
    @staticmethod
    def dict_string(dic: Dict[str, float]) -> str:
        return f"UCM: {dic['ucm']:6.2%} LCM: {dic['lcm']:6.2%} UAS: {dic['uas']:6.2%} LAS: {dic['las']:6.2%}"

    @staticmethod
    def eval(file_path: str, parser: StanfordDParser) -> Tuple[Dict[str, float], Dict[str, float]]:
        tok_ls = ConlluParser.load(file_path)
        inputs = []
        for tokl in tqdm(tok_ls, desc=f"Calculating Dependencies for: {file_path}"):
            sent, deprels, heads = ConlluParser.extract(tokl)
            pred_deprels, pred_heads = parser(sent)
            inputs.append((deprels, heads, pred_deprels, pred_heads))

        metric = Metric()
        doc_res, sent_res = metric(inputs)
        #print("D:", EvalConlluFile.dict_string(doc_res))
        #print("S:", EvalConlluFile.dict_string(sent_res))
        # return EvalConlluFile.dict_string(doc_res), EvalConlluFile.dict_string(sent_res)
        return doc_res, sent_res


def main(dirs: List[str] = ("27329", "27624")):
    base_p = realpath(join(realpath(__file__), "../../data/conllu_files_new"))
    save_p = realpath(join(realpath(__file__), "../../data/stanford_results.txt"))
    f = open(save_p, "w")
    # dirs = [join(base_p, di) for di in dirs]
    total_ages_sent = {"old": [], "new": [], "unknown": []}
    total_ages_doc = {"old": [], "new": [], "unknown": []}
    lang_dic = {"27329":"en", "27624": "de"}
    for di in dirs:
        dp = StanfordDParser(lang=lang_dic[di])
        dir_ages_sent = {"old": [], "new": [], "unknown": []}
        dir_ages_doc = {"old": [], "new": [], "unknown": []}
        dip = join(base_p, di)
        files = [join(dip, fn) for fn in PARTITIONING[di].keys()]
        ages = [PARTITIONING[di][fn] for fn in PARTITIONING[di].keys()]
        f.write(100*"=")
        f.write("\n")
        f.write(f"DIR: {di}\n")
        f.write(100 * "=")
        f.write("\n\n\n\n")
        for i in range(len(files)):
            doc_res, sent_res = EvalConlluFile.eval(files[i], dp)

            total_ages_sent[ages[i]].append(copy.deepcopy(sent_res))
            total_ages_doc[ages[i]].append(copy.deepcopy(doc_res))

            dir_ages_sent[ages[i]].append(copy.deepcopy(sent_res))
            dir_ages_doc[ages[i]].append(copy.deepcopy(doc_res))

            f.write(50 * "-")
            f.write("\n")
            f.write(f"Filename: {files[i]}\n")
            f.write("\n")
            f.write(f"Document-Based: \n")
            f.write(f"{EvalConlluFile.dict_string(doc_res)}\n\n")
            f.write(f"Sentence-Based: \n")
            f.write(f"{EvalConlluFile.dict_string(sent_res)}\n")
            f.write(50 * "-")
            f.write("\n\n\n")


        for age in ["old", "new", "unknown"]:
            f.write(50 * "-")
            f.write("\n")
            f.write(f"AGE: {age}\n")
            f.write("\n")
            f.write(f"Document-Based: \n")
            f.write(f"{EvalConlluFile.dict_string(Metric.avg_dicts(dir_ages_doc[age]))}\n\n")
            f.write(f"Sentence-Based: \n")
            f.write(f"{EvalConlluFile.dict_string(Metric.avg_dicts(dir_ages_sent[age]))}\n")
            f.write(50 * "-")
            f.write("\n\n\n")

    f.write(100 * "=")
    f.write("\n")
    f.write(f"TOTAL:\n")
    f.write(100 * "=")
    f.write("\n\n\n\n")

    for age in ["old", "new", "unknown"]:
        f.write(50 * "-")
        f.write("\n")
        f.write(f"AGE: {age}\n")
        f.write("\n")
        f.write(f"Document-Based: \n")
        f.write(f"{EvalConlluFile.dict_string(Metric.avg_dicts(total_ages_doc[age]))}\n\n")
        f.write(f"Sentence-Based: \n")
        f.write(f"{EvalConlluFile.dict_string(Metric.avg_dicts(total_ages_sent[age]))}\n")
        f.write(50 * "-")
        f.write("\n\n\n")

    f.flush()
    f.close()

    # onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]


if __name__ == "__main__":
    main()

"""fp = "/home/leon/work/train_parser/data/conllu_files_new/27329/1940.csv.extracted_tool_StepsParser_model_en_basic_xlmr.xmi/0.conllu"
a = ConlluParser.load(fp)[0]

for i in ConlluParser.extract(a):
    print(i)

print(100*"=")
p = StanfordDParser()
for i in p("I am not sure that I understand the question myself and partly because I have put it privately in a letter to the Financial Secretary to the Treasury .".split(" ")):
    print(i)
"""
"""# fp = "/home/leon/work/train_parser/data/conllu_files_new/27329/1803.csv.extracted_tool_StepsParser_model_en_basic_xlmr.xmi.conllu"
fp = "/home/leon/work/train_parser/data/conllu_files_new/27329/1900.csv.extracted_tool_StepsParser_model_en_basic_xlmr.xmi.conllu"
dp = StanfordDParser()
EvalConlluFile.eval(fp, dp)"""

"""
('These', '5', 'nsubj')
('are', '5', 'cop')
('three', '5', 'nummod')
('distinct', '5', 'amod')
('conditions', '0', 'root')
('of', '7', 'case')
('service', '5', 'nmod')
(',', '14', 'punct')
('but', '14', 'cc')
('the', '12', 'det')
('same', '12', 'amod')
('man', '14', 'nsubj')
('may', '14', 'aux')
('engage', '5', 'conj')
('to', '16', 'mark')
('fulfil', '14', 'xcomp')
('them', '16', 'obj')
('all', '19', 'det')
('three', '16', 'obj')
('.', '5', 'punct')
"""
