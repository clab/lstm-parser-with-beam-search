from sys import stdin
import re
import string


for line in stdin:
	newline = re.sub('.*?\|.?\|.?\|', '', line,  1)
	n = re.sub('\|.?\|.?\|', '|||', newline, 1)
	
	#messy, but python 2 has awful unicode handling
	#and python 3 doesn't do filters
	#run using python 3
	temp =''.join(list(filter(lambda x: x != '|', string.punctuation)))
	abup = '['+temp+'“”,、》」「《？；：‘］［｝｛＋＝－——）（＊&％¥＃@！～，。'+']'
	n = re.sub(abup, '', n)
	print(n, end='')
	#if len(newline) > 1:
	#	n = re.sub(ur'\|.?\|.?\|', '|||', newline[1], 1)
	#	print n,

