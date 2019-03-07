import csv
import itertools 
from itertools import zip_longest
import numpy as np
from logic_engine import * 


def get_blocks_in_tower(tower):
	tower_list = list(tower)
	blocks_in_tower = set([x for x in tower_list if x in ['R', 'G', 'B', 'Y', 'P', 'O']])

	return blocks_in_tower, len(blocks_in_tower)

# print(get_blocks_in_tower('RB0000'))

# def find_tower_matches(tower1, tower2):


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]



with open('./predicted_arrangement_all.csv', 'r') as a, open('./predicted_colors.csv', 'r') as c, open('./gt_pred_fnames.csv', 'w', newline='') as f:
	reader_a = csv.reader(a)
	reader_c = csv.reader(c)
	writer = csv.writer(f)

	count = 0

	for row_a, row_c in zip(reader_a, reader_c):

		count = count + 1
		print("count", count)

		# if count > 50:
		# 	break

		if row_a[0] != row_c[0]:
			raise Exception('not same file for A, C')
		else:
			# print(type(row_a[1:]))
			a = [int(x) for x in row_a[1:]]
			# print(type(row_c[1:16]))
			c = [int(x) for x in row_c[1:16]]


			# print(a, type(a), len(a))
			# print(c, type(c), len(c))

			A = np.flip(np.reshape(a, (5, 5)).T, 0)
			# print(A)
			C = np.reshape(c, (3, 5), 'F')

			B = fromACtoB(A,C)

			# print(B)

			fname_orig = row_a[0]
			fname_pred = fromBtoFilename(B)
			# print(fname_pred)

			
			# fname_pred_as_list = list(fname_pred)
			# blocks_in_fname_pred = [x for x in fname_pred_as_list if x in ['R', 'G', 'B', 'Y', 'P', 'O']]
			# set_pred = set(blocks_in_fname_pred)

			# fname_orig_as_list = list(row_a[0])
			# blocks_in_fname_orig = [x for x in fname_orig_as_list if x in ['R', 'G', 'B', 'Y', 'P', 'O']]
			# set_orig = set(blocks_in_fname_orig)
			# num_in_orig = len(set_orig)
			
			set_pred, num_in_pred = get_blocks_in_tower(fname_pred)
			set_orig, num_in_orig = get_blocks_in_tower(fname_orig)
			
			
			fname_pred_split = fname_pred[:-4].split('-')
			fname_orig_split = fname_orig[:-4].split('-')

			num_correct = 0
			num_wrong = 0
			for tower in fname_pred_split:
				if tower in fname_orig_split:
					_, add_correct = get_blocks_in_tower(tower)
					fname_orig_split.remove(tower)
					num_correct = num_correct + add_correct
				else:
					is_worst = 1000
					for tower_orig in fname_orig_split:
						add_wrong = levenshtein(tower, tower_orig)
						if add_wrong < is_worst:
							tower_worst = tower_orig 
							is_worst = add_wrong

							print(is_worst, tower, tower_orig)
					fname_orig_split.remove(tower_worst)
					num_wrong = num_wrong + is_worst
					print(num_wrong)



			# print(fname_pred_split)
			
			# num_true = len(set_orig.intersection(set_pred))
			# num_wrong = num_in_orig - num_true
			# gt_fname, pred_fname
			row_new = [row_a[0], fname_pred, num_in_orig, num_correct, num_wrong]
			writer.writerow(row_new)


