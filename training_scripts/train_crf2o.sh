
# hdt
# /home/stud_homes/s5935481/work3/venv/bin/python3.8 -m supar.cmds.crf2o_dep train --train /home/stud_homes/s5935481/work3/data/de_hdt-ud-train-ful.conllu --dev /home/stud_homes/s5935481/work3/data/de_hdt-ud-dev.conllu --test /home/stud_homes/s5935481/work3/data/de_hdt-ud-test.conllu --embed /home/stud_homes/s5935481/work3/data/embeddings/lemma_kom_clean.txt --n-embed 100 -b -d 0 -p /home/stud_homes/s5935481/work3/models/crf2o_dep_de -f char -c /home/stud_homes/s5935481/work3/data/configs/ptb.crf2o.dep.lstm.char.ini --proj --mbr


# gsd
# /home/stud_homes/s5935481/work3/venv/bin/python3.8 -m supar.cmds.crf2o_dep train --train /home/stud_homes/s5935481/work3/data/de_gsd-ud-train.conllu --dev /home/stud_homes/s5935481/work3/data/de_gsd-ud-dev.conllu --test /home/stud_homes/s5935481/work3/data/de_gsd-ud-test.conllu --embed /home/stud_homes/s5935481/work3/data/embeddings/lemma_kom_clean.txt --n-embed 100 -b -d 0 -p /home/stud_homes/s5935481/work3/models/crf2o_dep_de_gsd -f char -c /home/stud_homes/s5935481/work3/data/configs/ptb.crf2o.dep.lstm.char.ini --proj --mbr

# tiger
/home/stud_homes/s5935481/work3/venv/bin/python3.8 -m supar.cmds.crf2o_dep train --train /home/stud_homes/s5935481/work3/data/tiger_train.conllu --dev /home/stud_homes/s5935481/work3/data/tiger_dev.conllu --test /home/stud_homes/s5935481/work3/data/tiger_test.conllu --embed /home/stud_homes/s5935481/work3/data/embeddings/lemma_kom_clean.txt --n-embed 100 -b -d 0 -p /home/stud_homes/s5935481/work3/models/crf2o_dep_de_tiger -f char -c /home/stud_homes/s5935481/work3/data/configs/ptb.crf2o.dep.lstm.char.ini --proj --mbr

