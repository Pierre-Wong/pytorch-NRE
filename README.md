# pytorch-NER
Neural Relation Extraction in Pytorch

A pytorch version of https://github.com/thunlp/TensorFlow-NRE
To run the code, the dataset should be put in the folder origin_data/ using the following format, containing four files

    train.txt: training file, format (fb_mid_e1, fb_mid_e2, e1_name, e2_name, relation, sentence).
    test.txt: test file, same format as train.txt.
    relation2id.txt: all relations and corresponding ids, one per line.
    vec.txt: the pre-train word embedding file

which can be gotten from repository https://github.com/thunlp/TensorFlow-NRE.

For training,you need to type python3 relation.py and torch_test.py is for testing.
A pre-train model  "model10300.pth" is given.

The code has been tested for pytorch 0.3.0
