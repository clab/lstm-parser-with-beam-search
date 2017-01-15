import sys
from collections import defaultdict

wd = defaultdict(int)
f = []
for line in sys.stdin:
	ll = line.split()
	f.append(ll)
	for word in ll:
		wd[word] += 1

'''
total_words = len(wd)
nonsingle = 0
for word in wd:
	if wd[word] > 1:
		nonsingle += 1
'''
for ll in f:
	new_line = ""
	for word in ll:
		if wd[word] == 1:
			new_line += "UNKS "
		else:
			new_line += word+ " "
	print new_line

print >> sys.stderr, len(wd)
print >> sys.stderr, sum([1 for w in wd if wd[w] != 1])

