import sys
from itertools import izip

re_sent = open(sys.argv[1])
paired_sent = open(sys.argv[2])

for sent1, pair in izip(re_sent, paired_sent):

	two_sent = pair.split('|||')
	print sent1.strip() + ' ||| ' + two_sent[1].strip()
	