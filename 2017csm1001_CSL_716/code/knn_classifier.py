# USAGE
# python knn_classifier.py --dataset kaggle_dogs_vs_cats

# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

import numpy as np
import argparse
import imutils
import cv2
import os
import pickle
import codecs
from time import sleep
import matplotlib.pyplot as plt
import os.path

num_class = 17
class1, class2 = 0,0

def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])

	# handle normalizing the histogram if we are using OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)

	# otherwise, perform "in place" normalization in OpenCV 3 (I
	# personally hate the way this is done
	else:
		cv2.normalize(hist, hist)

	# return the flattened histogram as the feature vector
	return hist.flatten()


def load_data(file, save_path, Type):
	rawImages = []
	features = []
	labels = []

	with codecs.open(file, 'r') as f:
		
		lines = f.readlines()
		print('loading %s dataset......' %(Type) )
		for num, line in enumerate(lines):
			context = line.strip().split(' ')
			#train_path = os.path.join(path, context[0])
			train_path = context[0]
			train_label = context[1].strip()

			image = cv2.imread(train_path)
			#plt.imshow(image)
			#plt.show()
			#print(train_path, train_label)
			pixels = image_to_feature_vector(image)
			hist = extract_color_histogram(image)

			rawImages.append(pixels)
			features.append(hist)
			labels.append(train_label)
			if num > 0 and num % 10 == 0:
				print("[INFO] processed {}/{}".format(num, len(lines)))
		
		rawImages = np.array(rawImages)
		features = np.array(features)
		labels = np.array(labels)
		

		print("[INFO] pixels matrix: {:.2f}MB".format(
			rawImages.nbytes / (1024 * 1000.0)))
		print("[INFO] features matrix: {:.2f}MB".format(
			features.nbytes / (1024 * 1000.0)))

	return rawImages, features, labels 

def show(pred, actual):
	count = {}
	for i in range(num_class):
		count.update({str(i):{'1':0, '2':0}})

	'''	
	for i in range(len(pred)):
		print('predicted: %s    actual: %s  ' %(pred[i], actual[i]))
		#count[pred[i]] += 1


		count[actual[i]][pred[i]] += 1

	for key in count.keys():
		print(key, count[key])

	print('\n\n')
	'''

	Max, Min = 0,0
	Max_class, Min_class = 0,0
	for key in count.keys():
		#print(key, key != '1')
		#sleep(1)
		if key == '7':
			continue
		if key == '16':
			continue
		else:
			diff = count[key]['1'] - count[key]['2']
			#print('inside difference..............')
			if (diff > Max):
				Max = diff
				Max_class = key
			if (diff < Min):
				Min = diff
				Min_class = key

	print(Max_class, Min_class)

	return Max_class, Min_class

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
	help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("[INFO] describing images...")
#imagePaths = list(paths.list_images(args["dataset"]))
train_list = args["dataset"]
print(train_list)


# initialize the raw pixel intensities matrix, the features matrix,
# and labels list

save_path = './FlowerData/Finetune' #'fine_tune_list.txt'  'train_inter.txt' 'train_list.txt'

train_raw_image, train_feature, train_label = load_data('label.txt', save_path, 'train')
test_raw_image, test_feature, test_label = load_data('train_list.txt', save_path, 'test')

print('Data loaded....')


# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
#(trainRI, testRI, trainRL, testRL) = train_test_split(
#	rawImages, labels, test_size=0.25, random_state=42)
#(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
#	features, labels, test_size=0.25, random_state=42)

# train and evaluate a k-NN classifer on the raw pixel intensities
print("[INFO] evaluating raw pixel accuracy...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
	n_jobs=args["jobs"])

#model.fit(trainRI, trainRL)
model.fit(train_raw_image, train_label)
pred_i = model.predict(train_raw_image)
#print(pred_i, test_label)
class1, class2 = show(pred_i, test_label)
#acc = model.score(testRI, testRL)
#print(class1, class2)
acc = model.score(test_raw_image, test_label)

#print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

# train and evaluate a k-NN classifer on the histogram
# representations
print("[INFO] evaluating histogram accuracy...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
	n_jobs=args["jobs"])
#model.fit(trainFeat, trainLabels)
model.fit(train_feature, train_label)
pred_f = model.predict(test_feature)
class1, class2 =  show(pred_f, test_label)
#acc = model.score(testFeat, testLabels)
acc = model.score(test_feature, test_label)
#print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))
f = open('model.cpickle','wb')
f.write(pickle.dumps(model))
f.close()

#print(class1, class2)




