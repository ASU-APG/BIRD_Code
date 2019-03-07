import numpy as np 
from planner import extractfacts, extract1movepolicy 

def bin2int(x):
	'''
	if x is a n-bit binary array, convert it to its equivalent integer value
	'''
	x_str = [str(xx) for xx in x]
	x_str = ''.join(x_str)

	return int(x_str, 2)

def fromACtoB(A, C):
	'''
	if A is the arrangement vector and C is the color vector,
	B is the 5x5 block representation with
	{0, 1, 2, 3, 4, 5, 6}  ==> {empty, R, G, B, Y, P, O}
	'''
	B = np.zeros((5,5))
	count = 0

	for y in range(0,5):
		for x in range(4, -1, -1):
			if A[x,y] == 1:
				B[x,y] = bin2int(C[:, count])
				count = count +1

	return B 


def fromBtoAC(B):
	'''
	given B, get A, C
	'''
	A = np.zeros((5, 5))
	C = []
	for y in range(0,5):
		for x in range(4, -1, -1):
			if B[x,y] != 0:
				A[x,y] = 1

				color_bin_str = '{0:03b}'.format(int(B[x,y]))
				color_bin_list = list(map(int, list(color_bin_str)))

				C = C + color_bin_list
	return A, C

def getTop(B):
	'''
	get top-most blocks in each column
	'''
	top_B = []
	loc_top_B = []
	for y in range(0, 5):
		next_top = (next((bb for i, bb in enumerate(B[:, y]) if bb), None))
		next_loc = (next((i for i, bb in enumerate(B[:, y]) if bb), None))
		if next_top is not None:
			top_B.append(int(next_top))
			loc_top_B.append([next_loc,y])
	return top_B, loc_top_B

def fromBtoFilename(B):
	# B to filename
	int_col_map = ['0', 'R', 'G', 'B', 'Y', 'P', 'O']
	B_rs = np.reshape(np.flip(B, axis=0), (25, ), 'F')
	# B_rs = B.tolist()
	#print(B_rs)
	fname = ''
	for i in range(len(B_rs)):
		if i % 5 == 0:
			fname = fname + '0-'
		fname = fname + int_col_map[int(B_rs[i])]
		
	fname = fname[2:] + '0-000000'
	#print("\nfilename:", fname)
	return fname

def logic_engine(A, C, mov, shout=0):
	'''
	main
	'''
	A = np.reshape(A, (5, 5), 'F')
	C = np.reshape(C, (3, 5), 'F')

	A_new = A 
	C_new = C

	# print("---- arrangement :")
	# print(A)

	# print("---- colors :")
	# print(C)

	B = fromACtoB(A, C)

	# filename = fromBtoFilename(B)


	first_empty = np.where(B[-1, :] == 0)[0][0]
	top_B, loc_top_B = getTop(B)
	# print(top_B, loc_top_B)
	legit_1 = top_B
	legit_2 = top_B + [7,0] # 7: ground, 0: out

	mov = [bin2int(mov[:3]), bin2int(mov[3:])]

	if shout:
		print("START STATE")
		print("--------------------")
		print(B)
		print("\nACTION:")
		print("--------------------")
		print("move " + str(mov[0]) + " to " + str(mov[1]))

	B_new = B

	if (mov[0] in legit_1) and (mov[1] in legit_2):
		if mov[1] in legit_1:
			B_new[np.where(B == mov[0])] = 0
			new_loc = np.where(B == mov[1])
			B_new[new_loc[0][0] - 1, new_loc[1][0]] = mov[0]

			A_new, C_new = fromBtoAC(B_new)
		else:
			if mov[1] == 0:
				B_new[np.where(B == mov[0])] = 0
			if mov[1] == 7:
				B_new[np.where(B == mov[0])] = 0
				B_new[4, first_empty] = mov[0]

		# print(C_new)
		# print(A_new)
		if shout:
			print("\nNEW CONFIGURATION")
			print("--------------------")
			print(B_new)
	else:
		print("\nIllegal MOV")
		return None, None

	return A_new, C_new 

#colmap = { "r": [0, 0, 1], "g": [0, 1, 0], "b": [0, 1, 1], "y": [1, 0, 0], "p": [1, 0, 1], "o": [1, 1, 0]}

def ACtoFile(A,C):
    A_rs = np.reshape(A, (5, 5), 'F')
    C_rs = np.reshape(C, (3, 5), 'F')
    #print(A_rs)
    #print(C_rs)
    B = fromACtoB(A_rs, C_rs)
    file = fromBtoFilename(B)
    return file

def arrangement2policy(A1,C1,A2,C2):
	filename1 = ACtoFile(A1,C1)
	print(filename1)
	filename2 = ACtoFile(A2,C2)
	print(filename2)
	x=extractfacts(filename1,filename2)
	policy = extract1movepolicy(x[1:])
	return policy

def gt2AC(AC1,AC2):
	C1 = np.array(AC1[:15],dtype=np.int)
	A1 = np.array(AC1[15:40],dtype=np.int)
	A1tmp = np.reshape(A1, (5, 5), 'F')
	A1tmp = np.flipud(A1tmp)
	A1tmp = np.reshape(A1tmp, 25, order='F')
	#print(A1tmp)
	C2 = np.array(AC2[:15],dtype=np.int)
	A2 = np.array(AC2[15:40],dtype=np.int)
	A2tmp = np.reshape(A2, (5, 5), 'F')
	A2tmp = np.flipud(A2tmp)
	A2tmp = np.reshape(A2tmp, 25, order='F')
	#print(A2tmp)
	policy = arrangement2policy(A1tmp,C1,A2tmp,C2)
	print(policy)
	
#AC1 = [1,0,0,0,0,1,0,1,0,0,1,1,0,0,0,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#AC2 = [0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#gt2AC(AC1,AC2)

#A1 = np.array([0,0,0,0,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
#print("==")
#print(type(A1))
#C1 = np.array([1,0,0,0,0,1,0,1,0,0,1,1,0,0,0])
#A2 = np.array([0,0,0,0,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
#C2 = np.array([1,0,0,0,0,1,0,1,0,0,1,1,0,0,0])

# init
# Y00000-RGB000-000000-000000-000000
# = 1,0,0,0,0,1,0,1,0,0,1,1,0,0,0,,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

# goal
# R00000-YG0000-000000-000000-000000 
# = 0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

#"Y00000-RGB000-000000-000000-000000-000000-000000"
#A = np.array([0,0,0,0,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
#C = np.array([1,0,0,0,0,1,0,1,0,0,1,1,0,0,0])

#C = np.array([1,0,0,0,0,1,0,1,0,0,1,1,0,0,0])
#A = np.array([1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

#A_rs = np.reshape(A, (5, 5), 'F')
#C_rs = np.reshape(C, (3, 5), 'F')


#A,C to Filename
#B = fromACtoB(A_rs, C_rs)
#filename = fromBtoFilename(B)
#print(filename)

#A1 = np.array([0,0,0,0,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
#C1 = np.array([1,0,0,0,0,1,0,1,0,0,1,1,0,0,0])

#A1 = np.array([0,0,0,0,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
#C1 = np.array([1,0,0,0,0,1,0,1,0,0,1,1,0,0,0])
#print(ACtoFile(A1,C1))

#filename1 = "GYR000-O00000-000000-000000-000000"
#filename2 = "GR0000-000000-000000-000000-000000"
#p = list(np.zeros(42, dtype=int))

# LOGIC ENGINE
# mov = [0,1,1,1,1,1]
# A_new, C_new = logic_engine(A, C, mov, shout=1)