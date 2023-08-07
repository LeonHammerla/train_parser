

/path/to/python3 -m supar.cmds.biaffine_dep train --train /path/to/de_hdt-ud-train-ful.conllu --dev /path/to/de_hdt-ud-dev.conllu --test /path/to/de_hdt-ud-test.conllu --embed /path/to/embeddings/lemma_kom_clean.txt --n-embed 100 -b -d 0 -p /path/to/models/biaffine_dep_roberta_de -f bert -c /path/to/configs/ptb.biaffine.dep.roberta.ini --encoder bert --bert xlm-roberta-large-finetuned-conll03-german


