if [ "$1" = "english" ]; then
    dims=100
else
    dims=80
fi

if [ "$3" = "-G" ]; then
    mem=8096
else
    mem=1024
fi

build/parser/lstm-parse --cnn-mem $mem -T "$HOME/lstm_oracles/$1.train.oracle" -d "$HOME/lstm_oracles/$1.test.oracle" --hidden_dim 100 --lstm_input_dim 100 -w "$HOME/$1.vectors" --pretrained_dim $dims --rel_dim 20 --action_dim 20 -P $3 -m models/$1.$2.params -b 32 > results/conll/test-$1.$2.conll 2> results/log/test-$1.$2.log

perl eval.pl -g $HOME/lstm_oracles/$1.test.conll -s results/conll/test-$1.$2.conll -q > results/scores/$1.$2.score

