'''
This can both find the list of actions (rearrange(source, target)) and apply a list of actions to an inpuit array
(reorder(source, actions))

Run in python 2.7
'''


def peek(listt):
    temp = listt.pop()
    listt.append(temp)
    return temp

#swap is of the form swap(stack, buffer), while the others are all in order buffer, stack, output
def swap(a, b):
    temp1 = a.pop()
    temp2 = a.pop()
    a.append(temp1)
    b.append(temp2)


def delete(a):
    a.pop()


#the n~ are all the same operation, but instead of directly changing the array, they copy it and then return the copy
def ndelete(a):
    temp = a[:]
    delete(a)
    return temp


def nswap(a, b):
    c = a[:]
    d = b[:]
    swap(c, d)
    return c, d


def copy(buf):
    temp = buf.pop()
    buf.append(temp)
    buf.append(temp)


def ncopy(buf):
    temp = buf[:]
    copy(temp)
    return temp


def oute(output, element):
    output.append(element)


def noute(output, element):
    temp = output[:]
    oute(temp, element)
    return temp


def shift(buf, stack):
    stack.append(buf.pop())


def nshift(buf, stack):
    a = buf[:]
    b = stack[:]
    shift(a,b)
    return a, b


def output(stack, out):
    out.append(stack.pop())


def noutput(stack, out):
    a = stack[:]
    b = out[:]
    output(a, b)
    return a, b


def nappend(seq, s):
    if len(seq) > 0:
        a = seq[:]
    else:
        a = []
    a.append(s)
    return a


#this returns a list of actions to convert the source into the target
#it uses a buffer, stack, and output
#the input is reversed, so call it on [0 1 2 3 4 5] to read the input from left to right
def rearrange(source, tar):
    inp = source[:]
    target = tar[:]

    inp.reverse()
    stack = []
    buf = inp[:]
    out = []

    seq = []
    count = 0

    #limit the number of iterations tried
    #not sure if this is a good idea, as it it isn't too nonlinear
    while(target != out):
        #print buf, stack, out
        #print seq

        #the current top of the stack
        k = len(stack) - 1

        #one after the top of the output
        #I don't think I'm using this anymore
        i = len(out)

        #this is pretty much equivalent to nivre's algorithm when swap is an action
        if target[i] not in inp:
            oute(out, target[i])
            seq.append('OUT_E')

        elif len(buf) > 0 and target.count(peek(buf)) < buf.count(peek(buf)):
            delete(buf)
            seq.append('DELETE')
        elif len(buf) > 0 and target.count(peek(buf)) > buf.count(peek(buf)) + stack.count(peek(buf)) + out.count(peek(buf)):
            copy(buf)
            seq.append('COPY')
    
        elif len(stack) == 0 and len(buf) > 0:
            shift(buf, stack)
            seq.append('SHIFT')

        elif len(stack) == 1:
            if stack[0] == target[i]:
                output(stack, out)
                seq.append('OUT')
            elif len(buf) > 0 and target.count(peek(buf)) > buf.count(peek(buf)) + stack.count(peek(buf)) + out.count(peek(buf)):
                copy(buf)
                seq.append('COPY')
            elif len(buf) > 0:
                shift(buf, stack)
                seq.append('SHIFT')

        else:
            if stack[k] == target[i]:
                output(stack, out)
                seq.append('OUT')
            else:
                if target[i] not in buf:
                    order = [1,0]
                else:
                    order = [0,1]
                if order[0]:
                    swap(stack, buf)
                    seq.append('SWAP')
                if order[1] and len(buf) > 0:
                    seq.append('SHIFT')
                    shift(buf, stack)
        count += 1
    return seq

#this takes a list of actions and applies them to an input
#then returns the output
#should be fairly self-explanitory
def reorder(sequence, actions):
    stack = []
    out = []
    buf = sequence[:]
    buf.reverse()
    for action in actions:
        #print action
        if action == 'SWAP':
            swap(stack, buf)
        if action == 'SHIFT':
            shift(buf, stack)
        if action == 'OUT':
            output(stack, out)
        if action == 'COPY':
            copy(buf)
        if action == 'OUT_E':
            oute(out, 'epsilon')
        if action == 'DELETE':
            delete(buf)

        #print buf
        #print stack
        #print out

    return out

#inpu =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]




#target = [3, 6, 6, 10, 29, 7, 8, 4, 0, 34, 2, 36, 37, 38, 39, 40, 41, 42, 9, 44, 45, 46, 14, 14, 49, 13, 11, 11, 15, 19, 55, 56, 57, 17, 59, 18, 61, 62, 23, 23, 21, 24, 67, 68, 22, 22]

#inpu = [0,1,2,3,4,5]
#inpu = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
#target = [1, 3, 2, 0, 45, 46, 17, 16, 16, 50, 9, 52, 10, 11, 12, 56, 13, 14, 15, 18, 19, 23, 22, 64, 65, 21, 21, 68, 24, 25, 26, 72, 27, 74, 30, 76, 28, 29, 31, 80, 37, 82, 83, 33, 34, 35, 34, 36, 89, 32, 38, 39, 40]
#target  = [0, 2, 1, 3, 4, 5]
#print target
#inpu.reverse()

#x = rearrange(inpu, target)

#print x
#inpu.reverse()
#y = reorder(inpu, x)

#print inpu
#print target
#print y


'''
allTrue = True
steparray = []
for j in range(start, end):
    initial = range(1, j + 1)
    perm = itertools.permutations(initial)
    for pg in perm:
        initial = range(1, j + 1)
        temppg = list(pg)
        x, y = rearrange(initial, temppg)
        steparray.append(x)
        allTrue = allTrue and y
        if not y:
            print pg, x

#rearrange(inpu, target)
print allTrue
'''
