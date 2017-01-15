input_file = open('dev.24', 'r')
output_file = open('dev.24.mapped', 'w+')
dev_unk = open('dev_unk_file.txt', 'r')

dev_unk_vocab = []
for line in dev_unk:
    dev_unk_vocab.append(line.rstrip())

unk_replacement = ' UNK)'
freq_replacement = 0
for line in input_file:
    line_temp = line
    for vocab in dev_unk_vocab:
        to_find = ' ' + vocab.rstrip() + ')'
        if line_temp.find(to_find) < 0:
            continue
        else:
            freq_replacement += 1
            line_temp = line_temp.replace(to_find, unk_replacement)
    output_file.write(line_temp)

print "frequency of replacement = " + str(freq_replacement)

input_file.close()
output_file.close()
dev_unk.close()
