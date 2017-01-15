This folder contains the programs for the stack-LSTM based MT oracle.  It takes a file of alinged sentences of the form sentence1 ||| sentence2, and a file of aligments, and creates a file that contains the necessary actions to transform the input to the output.  

I should probably write a script that does everything at some point, but for now:

1. get the alignment and alinged sentences files.
2. run dualign.py like so:
	python dualign.py sentences alignments > output
3. then run that output through chudata.py:
	python chudata.py <output> formatted_data

and the data is all ready to go.  if you need to, chureal.py outputs the rearrangined sentence, and chuall smushes everything into the output.  run them the same way as chudata.py.  yay!