# lstm-parser-with-beam-search
An addition to [this dependency parser](https://github.com/clab/lstm-parser) which includes implementations for beam search, selectional branching, and heuristic backtracking, as described in [this paper](http://aclweb.org/anthology/D/D16/D16-1254.pdf).

If using this code, please cite the paper:

```
@InProceedings{buckman-ballesteros-dyer:2016:EMNLP2016,
  author    = {Buckman, Jacob  and  Ballesteros, Miguel  and  Dyer, Chris},
  title     = {Transition-Based Dependency Parsing with Heuristic Backtracking},
  booktitle = {Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing},
  month     = {November},
  year      = {2016},
  address   = {Austin, Texas},
  publisher = {Association for Computational Linguistics},
  pages     = {2313--2318},
  url       = {https://aclweb.org/anthology/D16-1254}
}
```

# Using alternate decoding methods

To use each decoding method, simply add the correct flags to the model (all numbers in these examples are tunable parameters):

`-b 12`: Beam search with 12 beams

`-D -b 50`: Dynamic beam search, cutting off any beams whose score is less than 50% of the top beam

`-b 32 -M .105`: Selectional branching, maximum of 32 beams, spawn a new beam whenever the negative log of the difference in probabilities between the top two beams is less than .105

`-B 12`: Heuristic backtracking, backtrack 12 times

`-B 12 --hb_cutoff`: Heuristic backtracking, backtrack up to 12 times unless we cutoff earlier

# Training the cutoff model

The cutoff model freezes the parameters of the regular network and trains an additional Stack-LSTM to predict where to cut off parsing. To train this model, use the normal training script, make sure you load an already-trained model, and add the `--train_hb` flag.

# Checking out the project for the first time

The first time you clone the repository, you need to sync the `cnn/` submodule.

    git submodule init
    git submodule update

# Build instructions

    mkdir build
    cd build
    cmake -DEIGEN3_INCLUDE_DIR=/opt/tools/eigen-dev ..
    make -j2

# Update cnn instructions
To sync the most recent version of `cnn`, you need to issue the following command:
 
    git submodule foreach git pull origin master
    
    
# How to get the arc-std oracles (traning set and dev set of the parser) having a CoNLL 2006 file:
   
    java -jar ParserOracleArcStd.jar -t -1 -l 1 -c train10.conll -i train10.conll > oracleTrainArcStd.txt
    (oracle code is in: /usr2/home/miguel/ParserOracle)
    (the train10.conll file should be fully projective)
    


