from collections import OrderedDict

x = 'BGT00-PYO00-Z0000-00000-00000'

def filecorrection(x):
	x = x.split("-")
	x1 = [i for i in x if i!='00000']
	x2 = [j for j in x if j=='00000']

	d = {}
	for word in x1:
	  word1 = word.replace("0","")	
	  key = len(word1)
	  d[key] = d.get(key, []) + [word]
	d = dict(sorted(d.items()))

	x1new = []
	for v in d.values():
		for i in v:
			x1new.append(i)

	return x1new+x2

print(filecorrection(x))
