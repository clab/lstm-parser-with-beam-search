# map all singleton words in train file to UNK
# additionally, map all unknown words in the dev set to UNK
# works directly with the oracles, instead of the original training set

train_file = open("train.oracle", "r")
dev_file = open("dev.oracle", "r")
train_file_mapped = open("train.oracle.mapped", "w+")
dev_file_mapped = open("dev.oracle.mapped", "w+")

vocab = {}
# first, get all the vocabulary to determine the singletons during training
start_sentence = True # make sure we don't process the sentence at the start (see the oracle format for details)
for line in train_file:
    if line[0] == '#':
        continue
    if len(line) > 2 and not(start_sentence):
        if line[0:5] == "SHIFT": # if SHIFT, process the vocabulary
            curr_word = line[6:-2].rstrip()         
            if not(curr_word in vocab):
                vocab[curr_word] = 1
            else:
                vocab[curr_word] = vocab[curr_word] + 1
    elif start_sentence:
        start_sentence = False # the sentence is only printed once
    elif len(line) <= 2:
        start_sentence = True # empty line signifies the start of a new sentence                        
        

ctr = 0
for key in vocab:
    if vocab[key] == 1:
        ctr += 1
        if ctr < 50:
            print key

print "total number of singletons = " + str(ctr)
print "train vocabulary size"
print len(vocab)

# step 2: map all training singletons to UNK
train_file.close()
train_file = open("train.oracle", "r")
freq_singleton = 0
full_sentence = False
for line in train_file:
    if line[0] == '#':
        full_sentence = True
        train_file_mapped.write(line)
        continue
    if full_sentence:
        full_sentence = False
        line_split = line.rstrip().split()    
        to_print = []
        for elem in line_split:
            assert elem in vocab
            if vocab[elem] > 1:
                to_print.append(elem.rstrip())
            else:
                to_print.append('UNK')                
        train_file_mapped.write(" ".join(to_print) + "\n")
    else:
        if not(line[0:5] == 'SHIFT'):
            train_file_mapped.write(line)
        else:
            curr_word = line[6:-2].rstrip()
            #print curr_word
            if vocab[curr_word] > 1:
                train_file_mapped.write(line)
            else:
                train_file_mapped.write("SHIFT(UNK)\n")    
                freq_singleton += 1

print "Sanity check: frequency of singletons = " + str(freq_singleton)
assert freq_singleton == ctr

freq_oov = 0
total_dev_words = 0
dev_unk_file = open('dev_unk_file.txt', 'w+')
# step 3: map all dev unknowns to UNK
for line in dev_file:
    if line[0] == '#':
        full_sentence = True 
        dev_file_mapped.write(line)
        continue
    if full_sentence: # handle the case of the full sentence on top
        full_sentence = False
        line_split = line.rstrip().split()    
        to_print = []
        for elem in line_split:
            #assert elem in vocab
            if elem in vocab and vocab[elem] > 1:
                to_print.append(elem.rstrip())
            else:
                to_print.append('UNK')
        dev_file_mapped.write(" ".join(to_print) + "\n")
    else:
        if not(line[0:5] == 'SHIFT'):
            dev_file_mapped.write(line)
        else:
            total_dev_words += 1
            curr_word = line[6:-2].rstrip()
            if curr_word in vocab and vocab[curr_word] > 1:
                dev_file_mapped.write(line)
            else:
                dev_unk_file.write(curr_word + '\n')
                dev_file_mapped.write("SHIFT(UNK)\n")
                freq_oov += 1
dev_unk_file.close()

#for line in dev_file:
#    if not(line[0:5] == 'SHIFT'):
#        dev_file_mapped.write(line)
#    else:
#        total_dev_words += 1
#        curr_word = line[6:-2].rstrip() 
#        if curr_word in vocab and vocab[curr_word] > 1:
#            dev_file_mapped.write(line)
#        else:
#            dev_file_mapped.write("SHIFT(UNK)\n")    
#            freq_oov += 1

print "freq oov = " + str(freq_oov)
print "Dev coverage = " + str(1.0 - float(freq_oov) / float(total_dev_words))

train_file.close()
dev_file.close()
train_file_mapped.close()
dev_file_mapped.close()
