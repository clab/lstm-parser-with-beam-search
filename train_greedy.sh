screen -AmdS train-greedy bash
screen -S train-greedy -p 0 -X stuff 'build/parser/lstm-parse -T $HOME/lstm_oracles/english.train.oracle -d $HOME/lstm_oracles/english.dev.oracle --hidden_dim 100 --lstm_input_dim 100 -w $HOME/english.vectors --pretrained_dim 100 --rel_dim 20 --action_dim 20 -t -P > results/conll/train-greedy-dev.conll 2> results/log/train-greedy.log\r'
