import subprocess
import sys
import re
import numpy as np
import csv

def extractfacts(filename1, filename2):
	#print(filename1)
	#print(filename2)
	empty = set()
	if set(filename1) == set(['0', '-']) and set(filename2) == set(['0', '-']):
		#print("both files have no blocks")
		returnvalue = [0,'']
	elif set(filename1) == set(['0', '-']) and set(filename2) != set(['0', '-']):
		#print("file1 has no blocks, No plan possible")
		returnvalue = ["No plan possible","No plan possible"]
	else:
		temp1 = set(filename1) - set(filename2)
		temp2 = set(filename2) - set(filename1)
		
		answersets = []

		if (temp1 != empty and temp2 != empty) or (temp1 == empty and temp2 != empty):
			#print("subset not found, No plan possible")
			returnvalue = ["No plan possible","No plan possible"]
		elif temp2 == empty and temp1 != empty:
			#print("subset found, plan possible")
			returnvalue = genfacts(filename1,filename2,list(temp1))
		elif temp1 == empty and temp2 == empty:
			#print("same blocks, plan possible")	
			returnvalue = genfacts(filename1,filename2,[])	
	#print(returnvalue)
	#print("------------------------------")
	return returnvalue

def genfacts(filename1,filename2,difflist):
	newlist = list(set(filename1)) 
	newlist.remove('0')
	newlist.remove('-')

	blockfact = "block("
	initstfact = "on("
	goalstfact = "on("

	for i in newlist:
		blockfact += i.lower()+";"

	for i in range(0,len(filename1)):
		if filename1[i]!="0" and filename1[i]!="-":
			if i==0 or filename1[i-1]=="-":
				initstfact += filename1[i].lower()+",t,0;"
			else:
				initstfact += filename1[i].lower()+","+filename1[i-1].lower()+",0;"

	for i in range(0,len(filename2)):
		if filename2[i]!="0" and filename2[i]!="-":
			if i==0 or filename2[i-1]=="-":
				goalstfact += filename2[i].lower()+",t,m;"
			else:
				goalstfact += filename2[i].lower()+","+filename2[i-1].lower()+",m;"

	if difflist:
		blockoutfact = "blockout("
		for i in difflist:
			blockoutfact += i.lower()+";"
			goalstfact += i.lower()+",n,m;"
		blockoutfact = blockoutfact[:-1]+")."
		#print(blockoutfact)

	blockfact = blockfact[:-1]+")."
	initstfact = initstfact[:-1]+")."
	goalstfact = goalstfact[:-1]+")."

	with open('plangen.lp','r') as f:
		filedata = f.read() 

		if difflist:
			filedata = filedata.replace('blockfact', blockfact).replace('blockoutfact', blockoutfact).replace('initstfact', initstfact).replace('goalstfact', goalstfact)
		else:
			filedata = filedata.replace('blockfact', blockfact).replace('blockoutfact', '').replace('initstfact', initstfact).replace('goalstfact', goalstfact)
			filedata = filedata.replace(':- not blockout(B), move(B,n,T).', '% :- not blockout(B), move(B,n,T).')
			filedata = filedata.replace(':- blockout(B), block(BB), move(BB,B,T).', '% :- blockout(B), block(BB), move(BB,B,T).')
			filedata = filedata.replace('location(n).', '%  location(n).')
				
		with open('plangentemp.lp', 'w') as f:
			f.write(filedata)

		for i in range(0,10):
			output = subprocess.getoutput("clingo -c m="+str(i)+" plangentemp.lp 0")
			label = output.splitlines()
		
			if label[-6]=="SATISFIABLE":
				#print("-----------------------------------------")
				#print("minimum length plan: "+str(i)+" steps")
				#print("-----------------------------------------")
				returnvalue = [str(i)]
				#print(output)
				label = output.splitlines()
				for i in range(4,len(label)-5,2):
					returnvalue.append(label[i])
				#print("ret"+str(returnvalue))
				break
	return returnvalue

filename1 = "GYR000-O00000-000000-000000-000000-000000"
filename2 = "GR0000-000000-000000-000000-000000-000000"
x=extractfacts(filename1, filename2)
print(x)

'''
with open('nat_img_data_30samples_blockcount_nxn.csv','r') as f:
	read = csv.reader(f,delimiter=',')
	writer = csv.writer(open('nnat_img_data_30samples_blockcount_nxn_plan.csv','w'),delimiter=',')
	for r in read:
                plan = extractfacts(r[0], r[5])
                writer.writerow(r + plan[0:2])
'''
with open('nnat_img_data_30samples_blockcount_nxn_plan.csv','r') as f:
	read = csv.reader(f,delimiter=',')
	writer = csv.writer(open('nnat_img_data_30samples_blockcount_nxn_plan_predplan.csv','w'),delimiter=',')
	for r in read:
                plan = extractfacts(r[1], r[6])
                writer.writerow(r + plan[0:2])
                

