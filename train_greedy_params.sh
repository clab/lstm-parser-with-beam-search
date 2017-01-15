if [ "$1" = "english" ]; then
    dims=100
else
    dims=80
fi

screen -AmdS train-greedy-$1-$2 bash
screen -S train-greedy-$1-$2 -p 0 -X stuff 'build/parser/lstm-parse -T $HOME/lstm_oracles/'$1'.train.oracle -d $HOME/lstm_oracles/'$1'.dev.oracle --hidden_dim 100 --lstm_input_dim 100 -w $HOME/'$1'.vectors --pretrained_dim '$dims' --rel_dim 20 --action_dim 20 -t -P > side_results/conll/train-greedy-'$1'-'$2'.conll 2> side_results/log/train-greedy-'$1'-'$2'.log\r'
