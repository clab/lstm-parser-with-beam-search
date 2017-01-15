import sys
from collections import defaultdict

for line in sys.stdin:
	new_line = ''
	words = line.strip()
	good_words = ['SHIFT', 'SWAP', 'DELETE', 'OUT', 'OUT_E', 'COPY']
	if words not in good_words:
		new_line = words.lower();
	else:
		new_line = words

	print new_line