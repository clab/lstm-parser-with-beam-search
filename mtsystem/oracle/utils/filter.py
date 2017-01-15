from itertools import izip

data = open('trainul.txt')
align = open('train/align')

out_data = open('trainul_filtered.txt', 'w+')
out_align = open('align_filtered.txt', 'w+')

for dline, aline in izip(data, align):
	sentence_pair = dline.split('|||')
	if len(sentence_pair[1].split()) < 80:
		out_data.write(dline)
		out_align.write(aline)
