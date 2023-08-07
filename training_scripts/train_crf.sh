
# hdt
# /path/to/python3 -m supar.cmds.crf_dep train --train /path/to/de_hdt-ud-train-ful.conllu --dev /path/to/de_hdt-ud-dev.conllu --test /path/to/de_hdt-ud-test.conllu --embed /path/to/embeddings/lemma_kom_clean.txt --n-embed 100 -b -d 0 -p /path/to/models/crf_dep_de -f char -c /path/to/configs/ptb.crf2o.dep.lstm.char.ini --proj --mbr

# gsd
# /path/to/python3 -m supar.cmds.crf_dep train --train /path/to/de_gsd-ud-train.conllu --dev /path/to/de_gsd-ud-dev.conllu --test /path/to/de_gsd-ud-test.conllu --embed /path/to/embeddings/lemma_kom_clean.txt --n-embed 100 -b -d 0 -p /path/to/models/crf_dep_de_gsd -f char -c /path/to/configs/ptb.crf2o.dep.lstm.char.ini --proj --mbr

# tiger
# /path/to/python3 -m supar.cmds.crf_dep train --train /path/to/tiger_train.conllu --dev /path/to/tiger_dev.conllu --test /path/to/tiger_test.conllu --embed /path/to/embeddings/lemma_kom_clean.txt --n-embed 100 -b -d 0 -p /path/to/models/crf_dep_de_tiger -f char -c /path/to/configs/ptb.crf2o.dep.lstm.char.ini --proj --mbr
# /path/to/python3 -m supar.cmds.crf_dep train --train /path/to/tiger_train_singleroot.conllu --dev /path/to/tiger_dev_singleroot.conllu --test /path/to/tiger_test_singleroot.conllu --embed /path/to/embeddings/lemma_kom_clean.txt --n-embed 100 -b -d 1 -p /path/to/models/crf_dep_de_tiger -f char -c /path/to/configs/ptb.crf2o.dep.lstm.char.ini --proj --mbr

# gum (en)
/path/to/python3 -m supar.cmds.crf_dep train --train /path/to/en_gum-ud-train.conllu --dev /path/to/en_gum-ud-dev.conllu --test /path/to/en_gum-ud-test.conllu --embed /path/to/embeddings/glove.6B.100d.txt --n-embed 100 -b -d 0 -p /path/to/models/crf_dep_de_gum -f char -c /path/to/configs/ptb.crf2o.dep.lstm.char.ini --proj --mbr
