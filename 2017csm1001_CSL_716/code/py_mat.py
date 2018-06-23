import scipy.io
import matlab.engine 
import os.path
from time import sleep


print('connecting to matlab engine......')
print(' (it may takes few minutes)')
'''
eng = matlab.engine.start_matlab()
print('connection done.....')
path1 =  '/home/kartik/Desktop/final/2flowers/jpg/5/'#+ str(cls1)+'/'
path2 = '/home/kartik/Desktop/final/2flowers/jpg/15/' #+str(cls2)+'/'

eng.test('/home/kartik/Desktop/final/2flowers/jpg/0/', path1, 0, nargout=0)

eng.test('/home/kartik/Desktop/final/2flowers/jpg/1/', path2 ,1, nargout=0)
'''

file1 = scipy.io.loadmat('label_0.mat')
file2 = scipy.io.loadmat('label_1.mat')

#print(file1)

#sleep(100)
file = open('match_file_0.txt', 'a')

for i in file1['list']:
	#print( i[2][0][0], i[2][0][0] > 0.23)
	if  i[2][0][0] > 0.23:
		#print(i[0][0], i[1][0])
		line = i[0][0] + " " + i[1][0] + "\n"
		file.write(line)

file.close()


file = open('match_file_1.txt', 'a')

for i in file2['list']:
	#print( i[2][0][0], i[2][0][0] > 0.23)
	if  i[2][0][0] > 0.13:
		#print(i[0][0], i[1][0])
		line = i[0][0] + " " + i[1][0] + "\n"
		file.write(line)

file.close()

