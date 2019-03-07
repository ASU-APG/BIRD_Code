import csv

lookup = {'00000001': 'n', '00000010': 'r', '00000100': 'g', '00001000' : 'b', '00010000' : 'y', '00100000' : 'p', '01000000' : 'o', '10000000' : 't', '00000000':''}

with open('./final_train_1hot.csv','r') as f:
	reader = csv.reader(f, delimiter=',')
	writer = csv.writer(open('./final_train_1hot_new.csv','w'), delimiter=',')
	for line in reader:
		#print(line[5:])
		arrlnlstnew = []
		l = []
		flag = False
		strmlst = []
		
		for ln in line[5:]:
			if line[4]=="0":
				row = line[0:5]+[""]
			elif line[4]=="No plan possible":
				row = line[0:5]+["No plan possible"]
			else:
				lnlist = [ln[i:i+8] for i in range(0, 128, 8)]
				lnlstnew = []
				for i in lnlist:
					lnlstnew.append(lookup.get(i))
				l.append(lnlstnew)
				flag = True
		
		if flag:
			for colrep in l:
				strm = ''
				for i,k in zip(colrep[0::2],colrep[1::2]):
					if not (i=='' and k==''):
						strm = strm + " " + "move("+i+","+k+")"
				strm = strm[1:]
				strmlst.append(strm)
			row = line[0:5]+strmlst
		#print(row)
		writer.writerow(row)