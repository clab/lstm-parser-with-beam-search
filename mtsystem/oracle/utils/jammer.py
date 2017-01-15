en_file = open('test/en')
ja_file = open('test/ja')

en = []
ja = []

for e in en_file:
	en.append(e.strip())
for j in ja_file:
	ja.append(j.strip())

assert(len(en) == len(ja))

for i in range(len(en)):
	e_s = en[i]
	j_s = ja[i]

	print  e_s.lower() + ' ||| ' + j_s

