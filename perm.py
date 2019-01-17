from itertools import permutations 
from itertools import product

def setting(goal):

    lst = goal.split("-")
    n = len(lst)
    top = []
    for i in range(0,len(goal)):
        if goal[i]!='0' and goal[i]!='-' and goal[i+1]=='0':
            top.append(goal[i]) 
    #print(top)
    t = tuple(range(1,n+1))
    h = tuple(range(1,len(lst[0])+1))
    tp = tuple(range(0,1))
    test = product(t, h, tp)

    def seq2conf(grid,seq):
        seq = seq.replace("-","")
        seqlst = list(seq)
        d = dict(zip(seqlst,grid))
        d.pop('0', None)
        for k,v in d.items(): 
            print(k,v)
            if k in top: 
                d[k] = v[:-1] + (1,)
        print(d)
    
    validgoalstates = permutations(lst)
    for i in list(validgoalstates):
        l = list(i)
        print(l) 
        l = "-".join(l)
        print(l)
        #seq2conf(l)
        seq2conf(test,l)
        print(list(test))

goal = 'BG000-P0000-R0000-Y0000-00000'
setting(goal)


