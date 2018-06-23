from time import sleep
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																									

label_file = open('label.txt', 'r')

ground_data = []
data_name = []
for line in label_file:
	name = line.strip().split(' ')
	ground_data.append(line)
	data_name.append(name[0])
label_file.close()

label_file = open('generated.txt', 'a')

#print(data_name)
#print('2flowers/jpg/0/image_0561.jpg' in data_name)



file = open('match_file_0.txt', 'r')

for line in file:
	data = line.strip().split(' ')
	#print(data[0], data[1])
	img_id = data[0].strip().split('2flowers')
	full_name = '2flowers'+img_id[1]
	if (full_name, full_name in data_name):
		ind = data_name.index(full_name)
		annot = ground_data[ind].strip().split(' ')[2]
		unlabel_name = data[1].strip().split('2flowers')
		un_full_name = '2flowers' + unlabel_name[1] 
		label = un_full_name + ' ' + '3' + ' '+annot+'\n'
		label_file.write(label)
		#sleep(2)

file.close()

file = open('match_file_1.txt', 'r')

for line in file:
	data = line.strip().split(' ')
	#print(data[0], data[1])
	img_id = data[0].strip().split('2flowers')
	full_name = '2flowers'+img_id[1]
	if (full_name in data_name):
		ind = data_name.index(full_name)
		annot = ground_data[ind].strip().split(' ')[2]
		unlabel_name = data[1].strip().split('2flowers')
		un_full_name = '2flowers' + unlabel_name[1] 
		label = un_full_name + ' ' + '4' + ' '+annot+'\n'
		label_file.write(label)

print('done')
file.close()

label_file.close()


with open('label.txt') as f:
	with open('fine_tune_list.txt', 'w') as f1:
		for line in f:
			f1.write(line)


with open('generated.txt') as f:
	with open('fine_tune_list.txt', 'a') as f1:
		for line in f:
			f1.write(line)
