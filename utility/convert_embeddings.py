import codecs
from tqdm import tqdm

path = "/home/stud_homes/s5935481/work3/data/embeddings/lemma_kom.txt"

with codecs.open(path, 'r', "utf-8") as f:
    lines = [line.strip() for line in f]
lines = lines[1:]
splits = [line.split() for line in lines]
tokens, vectors = [], []
for s in splits:
    tokens.append(s[0])
    vectors.append([float(ss) for ss in s[-100:]])
tokens = [token.split("_")[-1] for token in tokens]
path2 = "/home/stud_homes/s5935481/work3/data/embeddings/lemma_kom_clean.txt"
token_dict = dict()
for i in range(0, len(tokens)):
    if tokens[i] in token_dict:
        token_dict[tokens[i]].append(i)
    else:
        token_dict[tokens[i]] = [i]

for key in tqdm(token_dict):
    indices = token_dict[key]
    final_vec = []
    for i in range(0, 100):
        final_vec_indx = []
        for j in indices:
            final_vec_indx.append(vectors[j][i])
        final_vec.append(sum(final_vec_indx) / len(indices))
    token_dict[key] = final_vec
del tokens
del vectors

tokens, vectors = [], []
for key in token_dict:
    tokens.append(key)
    vectors.append(token_dict[key])

print("Info: Number of Token: {}".format(len(tokens)))
with codecs.open(path2, "w") as f:
    for i in range(0, len(tokens)):
        f.write(tokens[i] + " " + " ".join([str(vec) for vec in vectors[i]]) + "\n")