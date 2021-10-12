import supar
from tqdm import tqdm
from contextlib import redirect_stdout

file = "/home/stud_homes/s5935481/work3/data/de_hdt-ud-test.conllu"
directory = "/home/stud_homes/s5935481/work3/models/"
models_names = ['biaffine_dep_de', 'biaffine_dep_gbert_de', 'biaffine_dep_roberta_de', 'crf_dep_de', 'crf2o_dep_de']

with open("/home/stud_homes/s5935481/work3/data/model_eval_results.txt", 'w') as f:
    with redirect_stdout(f):
        for name in tqdm(models_names):
            model = supar.Parser.load(directory + name)
            if name in ['crf_dep_de', 'crf2o_dep_de']:
                out = model.evaluate(file, verbose=True, prob=True, tree=True, proj=True, mbr=True)
            else:
                out = model.evaluate(file, verbose=True, prob=True, tree=True, proj=True)


