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

# Checking out the project for the first time

The first time you clone the repository, you need to sync the `cnn/` submodule.

    git submodule init
    git submodule update

# Build instructions

    mkdir build
    cd build
    cmake -DEIGEN3_INCLUDE_DIR=/opt/tools/eigen-dev ..
    (in allegro: cmake -DEIGEN3_INCLUDE_DIR=/opt/tools/eigen-dev -G 'Unix Makefiles')
    make -j2

# Update cnn instructions
To sync the most recent version of `cnn`, you need to issue the following command:
 
    git submodule foreach git pull origin master
    
    
# Command to run the parser (in allegro): 

    parser/lstm-parse -h
    
    parser/lstm-parse -T /usr0/home/cdyer/projects/lstm-parser/train.txt -d /usr0/home/cdyer/projects/lstm-parser/dev.txt --hidden_dim 100 --lstm_input_dim 100 -w /usr3/home/lingwang/chris/sskip.100.vectors --pretrained_dim 100 --rel_dim 20 --action_dim 20 -t -P
    
# How to get the arc-std oracles (traning set and dev set of the parser) having a CoNLL 2006 file:
   
    java -jar ParserOracleArcStd.jar -t -1 -l 1 -c train10.conll -i train10.conll > oracleTrainArcStd.txt
    (oracle code is in: /usr2/home/miguel/ParserOracle)
    (the train10.conll file should be fully projective)
    


