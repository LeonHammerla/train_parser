

/path/to/python3 -m supar.cmds.biaffine_dep train --train /path/to/de_hdt-ud-train-ful.conllu --dev /path/to/de_hdt-ud-dev.conllu --test /path/to/de_hdt-ud-test.conllu --embed /path/to/embeddings/lemma_kom_clean.txt --n-embed 100 -b -d 1 -p /path/to/models/biaffine_dep_gbert_de -f bert -c /path/to/configs/ptb.biaffine.dep.gbert.ini --encoder bert --bert deepset/gbert-large

