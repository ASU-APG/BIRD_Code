import re
import numpy as np

def plantopolicy(mov):
	x = ['r','g','b','y','p','o']
	y = ['r','g','b','y','p','o','table','outoftable']
	xycomb = []
	for i in x:
		for j in y:
			if i!=j:
				xycomb.append("move("+i+","+j+")")

	movelist = re.findall(r'move\([^()]*\)', mov)
	plist = []

	for i in movelist :
		ind = 0
		p = list(np.zeros(42, dtype=int))
		for j in range(0,len(xycomb)):
			if xycomb[j]==i:
				ind = j
		p[ind]=1
		plist.append(p)

	return plist

pl= []
pl = plantopolicy("move(r,table)")
print(pl)

'''
move(x,y)
x - r/g/b/y/p/o 
y - r/g/b/y/p/o/table/outoftable 
if x!=y
6*8-6 
'''