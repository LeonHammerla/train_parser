import supar

directory = "/home/stud_homes/s5935481/work3/models/"
models = ['biaffine_dep_de', 'biaffine_dep_gbert_de', 'biaffine_dep_roberta_de']

for model in models:
    model = supar.Parser.load(directory + model)
    loss, metric = model.evaluate('/home/stud_homes/s5935481/work3/data/de_hdt-ud-test.conllu', verbose=False, proj=True, tree=True)
    print(model + ": Loss: {}; Scores: {}".format(loss, metric))
