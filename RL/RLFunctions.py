from itertools import permutations                                              
from itertools import product  
import random
              
start = 'GB000-R0000-P0000-YO000-00000'                                                                                                                          
goal = 'BG000-P0000-R0000-Y0000-00000'   

#start = 'RGBPY-00000-00000-00000-00000'                                                                                                                          
#goal = 'BG000-P0000-R0000-Y0000-00000'                                          

#start = 'R0000-G0000-B0000-P0000-Y0000'                                                                                                                          
#goal = 'RGBPY-00000-00000-00000-00000'                                          

#invalid
bl = ['r','g','b','p','y','o']
loc = ['r','g','b','p','y','o','t','n']
allmoves = [(i,j) for (i,j) in product(bl, loc) if i!=j]
#print(allmoves)

n1 = 5
n2 = 5
tablelocs = [] 

def getvalidmoves(allmoves, state, goal):
	top = []
	state = state.lower()
	for i in range(0,len(state)):
		if state[i]!='0' and state[i]!='-' and (state[i+1]=='0' or state[i+1]=='-'):
			top.append(state[i])                                                 
	print(top) 

	t = tuple(range(1,n1+1))                                                     
	h = tuple(range(1,n2+1))                                           
	tp = tuple(range(0,1))                                                      
	grid = [list(tup) for tup in product(t, h, tp)]   
	global tablelocs 
	tablelocs = [[i,j,k] for [i,j,k] in grid if j==1]
	add = []
	for i in tablelocs:
		add.append([i[0],i[1]])
	tablelocs = add	
	#print(tablelocs)   
	#print(grid)

	#goal = goal.replace("-","").replace("0","").lower()    
	#goallst = [i for i in goal]
	#print(goallst)
	state = state.replace("-","")                                               
	statelst = list(state) 

	d = dict(zip(statelst,grid))                                              
	d.pop('0', None)                                                        
	for k,v in d.items():                                                   
		#print(k,v)                                                          
		if k in top:                                                        
			d[k] = v[:-1] + [1,]                                            
	print(d)    

	# a block can be moved only if its on top, only be placed on block which is on top or table or can be moved out of table
	movessubset = [(i,j) for (i,j) in allmoves if i in top and (j in top or j=='t' or j=='n')]
	# if block is not needed in goal, then only outoftable is a valid move
	# movessubset = [(i,j) for (i,j) in movessubset if  not (i in goallst and j=='n')]
	print(movessubset)
	return d, movessubset
	
def printgrid(grid):
	#print(grid)
	g = [[0] * n1 for x in range(n2)]
	#print(g)
	for i,j in grid.items():
		#print(j[0])
		#print(j[0],j[1],i)
		g[j[0]-1][j[1]-1] = i
	#print(g)
	print("---------------")
	g = [list(i) for i in zip(*g)]
	for i in g[::-1]:
		#print(i)
		print(i)
	print("---------------")

def moveeffects(gridarr,move):
	print("===========================")
	printgrid(gridarr)
	#move = movessubset[1]
	print(move)
	global tablelocs
	if move[1]!='t' and move[1]!='n':
		gridarr[move[0]][0] = gridarr[move[1]][0] 
		gridarr[move[0]][1] = gridarr[move[1]][1]+1 
		gridarr[move[1]][2] = 0  # if some block is moved on this block, this block is no longer on top
		print(gridarr[move[0]])   
		print(gridarr[move[1]])	
	elif move[1]=='t':
		for i in tablelocs:
			if i == gridarr[move[0]][0:2]:
				break
			else:
				if i not in [i[0:2] for i in gridarr.values()]:
					gridarr[move[0]] = i+[1]
		print(gridarr[move[0]]) 
	elif move[1]=='n':
		gridarr = { k:v for k,v in gridarr.items() if k!=move[0] }	
	printgrid(gridarr)
	print("===========================")
	return gridarr
'''	
def Diff(li1, li2): 
	#print(set(li1)-set(li2))
	return len(set(li1)-set(li2))
'''
def RL(start, goal):

	glst = goal.lower().split("-")  
	slst = start.lower().split("-")
	#print(Diff(slst,glst))
	validgoalstates = permutations(glst) 

	global allmoves
	movetried = []
	moveuseful = []

	gridarr, movessubset = getvalidmoves(allmoves,start,goal)
	print(movessubset)
	move = random.choice(movessubset)
	print(move)
	movetried.append(move)
	print(movetried)
	currentgrid = moveeffects(gridarr, move)

RL(start, goal)


