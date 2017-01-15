if [ "$1" = "english" ]; then
    dims=100
else
    dims=80
fi

if [ "${3:0:2}" = "-G" ]; then
    mem=8096
else
    mem=1024
fi

if [ "$3" = "" ]; then
    name="greedy"
else
    name="$4"
fi

line=`head -n 6 side_results/log/train-greedy-$1-$2.log | tail -1`
modelname=${line:28}
cp $modelname many_models/$1-$2.params

build/parser/lstm-parse --cnn-mem $mem -T "$HOME/lstm_oracles/$1.train.oracle" -d "$HOME/lstm_oracles/$1.test.oracle" --hidden_dim 100 --lstm_input_dim 100 -w "$HOME/$1.vectors" --pretrained_dim $dims --rel_dim 20 --action_dim 20 -P -m many_models/$1-$2.params $3 > results2/conll/test-$1-$2.$name.conll 2> results2/log/test-$1-$2.$name.log

perl eval.pl -g $HOME/lstm_oracles/$1.test.conll -s results2/conll/test-$1-$2.$name.conll -q > results2/scores/$1-$2.$name.score

