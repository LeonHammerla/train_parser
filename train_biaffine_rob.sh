

/home/stud_homes/s5935481/work3/venv/bin/python3.8 -m supar.cmds.biaffine_dep train --train /home/stud_homes/s5935481/work3/data/de_hdt-ud-train-ful.conllu --dev /home/stud_homes/s5935481/work3/data/de_hdt-ud-dev.conllu --test /home/stud_homes/s5935481/work3/data/de_hdt-ud-test.conllu --embed /home/stud_homes/s5935481/work3/data/embeddings/lemma_kom_clean.txt --n-embed 100 -b -d 0 -p /home/stud_homes/s5935481/work3/models/biaffine_dep_roberta_de -f bert -c /home/stud_homes/s5935481/work3/data/configs/ptb.biaffine.dep.roberta.ini --encoder bert --bert xlm-roberta-large-finetuned-conll03-german


