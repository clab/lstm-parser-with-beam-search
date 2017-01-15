import sys

'''
This reads in two files.  one of them is the alignment (as defined by cdec),  in the form
0-0 0-1 1-2 2-1 3-3 4-4 5-5
chinese
english
another is the data before alignment, so something like:

where the lines corrospond to each other.

It outputs a in the form:
0 1 2 3 4 5 / 0 2 1 3 4 5
chinese
english

I would give examples but python 2.~ all hate chinese characters

this should be run as:

python dualign.py (aligned text) (alignments) > output_file.txt
'''

#opens the files
#file1 is the alignments
file1 = sys.argv[2]
file2 = sys.argv[1]

f1 = open(file1)
f2 = open(file2)

line1 = f1.readline()
line2 = f2.readline()

parl = []

#this adds a pair to a dual (a, b)
def appair(st, pair):
	st[0].append(pair[0])
	st[1].append(pair[1])

count = 0

#for each line
while(line1):
	#split the alignment data a-b
	#and stores it [a, b]
	split_s = line1.split(' ')
	split_d = map(lambda x: x.split('-'), split_s)

	try:
		split_i = map(lambda x: [int(x[0]), int(x[1])], split_d)			
	except:
		print count
	count += 1
	
	#gets the two language pairs
	zh_en = line2.split('|||')
	zh = zh_en[0]
	en = zh_en[1]

	#this sets the source and target arrays
	#I replace everything not in the source array as just a number not in the source array
	# so go through the alignment vector and update the target
	source = range(len(zh.split()))

	target = range(len(zh.split()), len(en.split())+len(zh.split()))
	#this just makes sure that the outputs are well formed
	#they aren't always, unfortunately
	#sometimes you might get 0-6, and have len(target) = 2 or something
	#so sanity checking
	ying_print = True
	for i in split_i:
		if i[1] < len(target):
			target[i[1]] = i[0]
		else:
			ying_print = False

	#this just prints everything
	if ying_print:
		for s in source:
			print str(s) + ' ',
		print '/',

		for t in target:
			print str(t) + ' ',
		print ''	
		print zh
		print en,
	

	line1 = f1.readline()
	line2 = f2.readline()
