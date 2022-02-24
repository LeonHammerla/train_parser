

# hdt
# /home/stud_homes/s5935481/work3/venv/bin/python3.8 -m supar.cmds.biaffine_dep train --train /home/stud_homes/s5935481/work3/data/de_hdt-ud-train-ful.conllu --dev /home/stud_homes/s5935481/work3/data/de_hdt-ud-dev.conllu --test /home/stud_homes/s5935481/work3/data/de_hdt-ud-test.conllu --embed /home/stud_homes/s5935481/work3/data/embeddings/lemma_kom_clean.txt --n-embed 100 -b -d 0 -p /home/stud_homes/s5935481/work3/models/biaffine_dep_de -f char -c /home/stud_homes/s5935481/work3/data/configs/ptb.biaffine.dep.lstm.char.ini

# gsd
# /home/stud_homes/s5935481/work3/venv/bin/python3.8 -m supar.cmds.biaffine_dep train --train /home/stud_homes/s5935481/work3/data/de_gsd-ud-train.conllu --dev /home/stud_homes/s5935481/work3/data/de_gsd-ud-dev.conllu --test /home/stud_homes/s5935481/work3/data/de_gsd-ud-test.conllu --embed /home/stud_homes/s5935481/work3/data/embeddings/lemma_kom_clean.txt --n-embed 100 -b -d 0 -p /home/stud_homes/s5935481/work3/models/biaffine_dep_de_gsd -f char -c /home/stud_homes/s5935481/work3/data/configs/ptb.biaffine.dep.lstm.char.ini

# tiger
# /home/stud_homes/s5935481/work3/venv/bin/python3.8 -m supar.cmds.biaffine_dep train --train /home/stud_homes/s5935481/work3/data/tiger_train.conllu --dev /home/stud_homes/s5935481/work3/data/tiger_dev.conllu --test /home/stud_homes/s5935481/work3/data/tiger_test.conllu --embed /home/stud_homes/s5935481/work3/data/embeddings/lemma_kom_clean.txt --n-embed 100 -b -d 0 -p /home/stud_homes/s5935481/work3/models/biaffine_dep_de_tiger -f char -c /home/stud_homes/s5935481/work3/data/configs/ptb.biaffine.dep.lstm.char.ini
# /home/stud_homes/s5935481/work3/venv/bin/python3.8 -m supar.cmds.biaffine_dep train --train /home/stud_homes/s5935481/work3/data/tiger_train_singleroot.conllu --dev /home/stud_homes/s5935481/work3/data/tiger_dev_singleroot.conllu --test /home/stud_homes/s5935481/work3/data/tiger_test_singleroot.conllu --embed /home/stud_homes/s5935481/work3/data/embeddings/lemma_kom_clean.txt --n-embed 100 -b -d 1 -p /home/stud_homes/s5935481/work3/models/biaffine_dep_de_tiger -f char -c /home/stud_homes/s5935481/work3/data/configs/ptb.biaffine.dep.lstm.char.ini

# gum (en)
/home/stud_homes/s5935481/work3/venv/bin/python3.8 -m supar.cmds.biaffine_dep train --train /home/stud_homes/s5935481/work3/data/en_gum-ud-train.conllu --dev /home/stud_homes/s5935481/work3/data/en_gum-ud-dev.conllu --test /home/stud_homes/s5935481/work3/data/en_gum-ud-test.conllu --embed /home/stud_homes/s5935481/work3/data/embeddings/glove.6B.100d.txt --n-embed 100 -b -d 0 -p /home/stud_homes/s5935481/work3/models/biaffine_dep_de_gum -f char -c /home/stud_homes/s5935481/work3/data/configs/ptb.biaffine.dep.lstm.char.ini
