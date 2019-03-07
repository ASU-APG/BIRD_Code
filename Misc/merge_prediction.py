import csv
import commands
import os

files = os.listdir("./")


def combnxn(in_file):
	out_file = in_file[:-4]+"_nxn.csv"
	with open(in_file,'r') as f:
		read = csv.reader(f,delimiter=',')
		writer = csv.writer(open(out_file,'w'),delimiter=',')
		alllines = []
		for line in read:
			alllines.append(line)
		for i in range(0,len(alllines)):
			for j in range(0,len(alllines)):
					newline = alllines[i]+alllines[j]
					writer.writerow(newline)

#combnxn('nat_img_data1.csv')
combnxn('nat_img_data2.csv')

def merge_predictions_natural(in_file):
	out_file = in_file[:-4]+"_pred.csv"
	with open(in_file,'r') as f:
		read = csv.reader(f,delimiter=',')
		writer = csv.writer(open(out_file,'w'),delimiter=',')f
		for line in read:
			s1,o1 = commands.getstatusoutput("awk -F, '($1==\""+line[0]+"\") && ($3==\""+line[5]+"\") final_1M.csv")
			#s2,o2 = commands.getstatusoutput("awk -F, '$1==\""+line[5]+"\"' final_1M.csv.csv")
			pred_data1 = o1.rstrip().split(',')
			#pred_data2 = o2.rstrip().split(',')[1:]
			print(pred_data1)
			#newline = line[0:2]+pred_data1+line[2:4]+pred_data2+line[4:]
			#print(newline)
			#writer.writerow(newline)

merge_predictions_natural('nat_img_data2_nxn.csv')

'''
def merge_predictions(in_file):
	out_file = in_file[:-4]+"_pred.csv"
	with open(in_file,'r') as f:
		read = csv.reader(f,delimiter=',')
		writer = csv.writer(open(out_file,'w'),delimiter=',')
		for line in read:
			s1,o1 = commands.getstatusoutput("awk -F, '$1==\""+line[0]+"\"' gt_pred_fnames.csv")
			s2,o2 = commands.getstatusoutput("awk -F, '$1==\""+line[2]+"\"' gt_pred_fnames.csv")
			pred_data1 = o1.rstrip().split(',')[1:]
			pred_data2 = o2.rstrip().split(',')[1:]
			newline = line[0:2]+pred_data1+line[2:4]+pred_data2+line[4:]
			writer.writerow(newline)

for fil in files:
	if fil.endswith("final_move.csv"):
		merge_predictions(fil)
		print("done "+fil)
'''


