import sys
import perm_re as perm


'''
This outputs the data in proper form for learning
something like 
[][][input data]
action1
aciton2...
it should be called like:
python chudata.py <input file> output file
where input file has the form
seq1 / seq2
input data
(target)
the target is ignored here
it may be used elsewhere
where seq1, 2 define the permutation
so something like:
0 1 2 3 4 5 / 5 2 1 3 4 0

The output of dualign should correspond with this
'''

count = 0
seq = []
print ''

for line in sys.stdin:
    #for the first line, read in the line and set everything up
    if count%3  == 2:
        en = line.split()
    elif count%3 == 0:
        st = line.split('/')
        sstr = st[0].split()
        tstr = st[1].split()
        source = map(lambda x: int(x), sstr)
        target = map(lambda x: int(x), tstr)
        #get the list of actions
        seq = perm.rearrange(source, target)
    else:
        #then print everything
        sys.stdout.write('[][')
        linec = 1
        for tok in line.split():
            if linec == len(line.split()):
                sys.stdout.write(tok + '-' + 'a')
            else:
                sys.stdout.write(tok + '-a' ', ')
            linec += 1
        print ']'
        for action in seq:
            print action
            print '[][]'
        print ''
    count += 1
