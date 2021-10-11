from io import open
from conllu import parse_incr
import codecs
from tqdm import tqdm

files = ["/home/stud_homes/s5935481/work3/treebanks/UD_German-HDT/de_hdt-ud-train-a-1.conllu",
         "/home/stud_homes/s5935481/work3/treebanks/UD_German-HDT/de_hdt-ud-train-a-2.conllu",
         "/home/stud_homes/s5935481/work3/treebanks/UD_German-HDT/de_hdt-ud-train-b-1.conllu",
         "/home/stud_homes/s5935481/work3/treebanks/UD_German-HDT/de_hdt-ud-train-b-2.conllu"]

merged_tokenlists = []

for file in files:
    data_file = open(file, "r", encoding="utf-8")
    for tokenlist in tqdm(parse_incr(data_file)):
        merged_tokenlists.append(tokenlist)

result = "".join([i.serialize() for i in merged_tokenlists])

with codecs.open("/home/stud_homes/s5935481/work3/treebanks/UD_German-HDT/de_hdt-ud-train-ful.conllu", "w", "utf-8") as f:
    f.write(result)