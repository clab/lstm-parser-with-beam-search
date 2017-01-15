# nt-parser
Top-down transition-based constituency parser with LSTMs

### Train discriminative model

    ../build/nt-parser/nt-parser --cnn-mem 2048 -x -T ../oracles/tx-pos.oracle -d ../oracles/dx-pos.oracle -C ../wsj/dev.24 -P -t --pretrained_dim 64 -w /usr1/corpora/gigaword_5/vectors.64.txt.gz --lstm_input_dim 128 --hidden_dim 128 -D 0.3
    
### (ONLY) Parsing with discriminative model

    ./nt-parser --cnn-mem 2048 -x -T ../oracles/tx-pos.oracle -p ../oracles/textx-pos.oracle -C ../wsj/test.23 -P --pretrained_dim 64 -w ../vectors.64.txt.gz --lstm_input_dim 128 --hidden_dim 128 -D 0.3 -m latest_model > output.txt
    
    Note: the output will be stored in /tmp/parse/parser_test_eval.xxxx.txt and the parser will output F1 score calculated with EVALB.

### Train generative model

    ../nt-parser/nt-parser-gen -x -T ../oracles/train-gen.oracle -d ../oracles/dev-gen.oracle -t --clusters ../clusters-train-berk.txt --input_dim 256 --lstm_input_dim 256 --hidden_dim 256 -D 0.3

### Sample trees from discriminative model for test set

    ../build/nt-parser/nt-parser -x -T ../oracles/tx-pos.oracle -p ../oracles/testx-pos.oracle -P -m ntparse_pos_0_2_32_64_16_60-pid43697.params --alpha 0.85 -s 100 > test-samples.props

important parameters

 * s = # of samples
 * alpha = posterior scaling (since this is a proposal, a higher entropy distribution is probably good, so a value < 1 is sensible.

### Prepare samples for likelihood evaluation

    ~/projects/cdec/corpus/cut-corpus.pl 3 test-samples.props > test-samples.trees

### Evaluate joint likelihood under generative model

    ../build/nt-parser/nt-parser-gen -x -T ../oracles/train-gen.oracle --clusters ../clusters-train-berk.txt --input_dim 256 --lstm_input_dim 256 --hidden_dim 256 -p test-samples.trees -m /path/to/gen-params.txt > test-samples.likelihoods

### Estimate marginal likelihood

    ./is-estimate-marginal-llh.pl 2416 100 test-samples.props test-samples.likelihoods > llh.txt 2> rescored.trees

 * 100 = # of samles
 * 2416 = # of sentences in test set
 * `rescored.trees` will contain the samples reranked by p(x,y)

### Compute parser evaluation

    ../nt-parser/add-fake-preterms-for-eval.pl rescored.trees > rescored.preterm.trees
    ../nt-parser/replace-unks-in-trees.pl ../oracles/testx-pos.oracle rescored.preterm.trees > hyp.trees

### Map a dev/test parser tokens and POS-tags output that may contain 'UNK' tokens and 'XX' tags to match the gold dev/test data, for EVALB

    python remove_dev_unk.py [gold-dev-file] [output-dev-file]
