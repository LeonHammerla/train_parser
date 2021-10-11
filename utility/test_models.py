import supar
from tqdm import tqdm


text = [['Zuletzt', 'war', 'er', 'Präsident', 'des', 'Reichshofrates', '.'],
        ['Der', 'Aktienkurs', 'wird', 'hierdurch', 'sicher', 'sinken', '.'],
        ['Im', 'Jahr', '1941', 'starb', 'seine', 'Frau', 'Kathleen', '.']]

directory = "/home/stud_homes/s5935481/work3/models/"
models_names = ['biaffine_dep_de', 'biaffine_dep_gbert_de', 'biaffine_dep_roberta_de', 'crf_dep_de']

results = dict()
for name in tqdm(models_names):
    model = supar.Parser.load(directory + name)
    results[name] = model.predict(text, prob=True, verbose=False)

f = open("/home/stud_homes/s5935481/work3/data/test_stats.txt", "w")

for i in range(0, len(text)):
    print("Satz: {}".format(" ".join(text[i])))
    f.write("Satz: {}".format(" ".join(text[i])))
    for key in results:
        print("Model: {}; Arcs: {}, Rels: {}".format(key, results[key][i].arcs, results[key][i].rels))
        f.write("Model: {}; Arcs: {}, Rels: {}".format(key, results[key][i].arcs, results[key][i].rels))

f.close()
