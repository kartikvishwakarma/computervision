# -*- coding: utf-8 -*-

import os
import cv2
import sys
import math
import codecs
import pickle
import skimage
import numpy as np
import config as cfg
import selectivesearch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from time import sleep

def resize_image(in_image, new_width, new_height, out_image=None, resize_mode=cv2.INTER_CUBIC):
    '''
    
    :param in_image: Input picture
    :param new_width:  The width of the new picture after resize
    :param new_height: The length of the new image after resize
    :param out_image: The address of the new picture saved after resize
    :param resize_mode: Cv2 mode for resize
    :return: New pictures after resize
    '''
    img = cv2.resize(in_image, (new_width, new_height), resize_mode)
    if out_image:
        cv2.imwrite(out_image, img)
    return img

def clip_pic(img, rect):
    '''
    
    :param img: Input picture
    :param rect: 4 parameters of rect rectangle
    :return: A pair of diagonal points and length-width information of the portion of the input image corresponding to the rect position and the rectangle
    '''
    x, y, w, h = rect[0], rect[1], rect[2], rect[3]
    x_1 = x + w
    y_1 = y + h
    return img[y:y_1, x:x_1, :], [x, y, x_1, y_1, w, h]

def IOU(ver1, vertice2):
    '''
    IOU for calculating two rectangular boxes
    :param ver1: The first rectangle
    :param vertice2: The second rectangle
    :return: The IOU of two rectangular boxes
    '''
    vertice1 = [ver1[0], ver1[1], ver1[0]+ver1[2], ver1[1]+ver1[3]]
    area_inter = if_intersection(vertice1[0], vertice1[2], vertice1[1], vertice1[3], vertice2[0], vertice2[2], vertice2[1], vertice2[3])
    if area_inter:
        area_1 = ver1[2] * ver1[3]
        area_2 = vertice2[4] * vertice2[5]
        iou = float(area_inter) / (area_1 + area_2 - area_inter)
        return iou
    return False

def view_bar(message, num, total):
    '''
    Progress bar tool
    :param message: Information to display before the progress bar
    :param num: The number of objects currently processed
    :param total: The total number of objects to be processed
    :return: 
    '''
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()

def show_rect(img_path, regions, message):
    '''
    :param img_path: The original picture to be displayed
    :param regions: The parameters of the rectangle to be marked on the original picture
    :param message: Information added around the rectangle
    :return: 
    '''
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for x, y, w, h in regions:
        x, y, w, h =int(x),int(y),int(w),int(h)
        rect = cv2.rectangle(
            img,(x, y), (x+w, y+h), (0,255,0),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, message, (x+20, y+40),font, 1,(255,0,0),2)
    plt.imshow(img)
    plt.show()

def image_proposal(img_path):
    '''
    Enter the picture to be candidate box extraction
    The candidate box is extracted by using the characteristics of each pixel of the picture, 
    because the number of candidate boxes is too many and the size of candidate boxes required for different problem backgrounds is not the same.
    Therefore, it is necessary to go through a series of rules to further reduce the number of feature frames.
    '''
    img = cv2.imread(img_path)
    img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
    candidates = set()
    images = []
    vertices = []
    for r in regions:
        if r['rect'] in candidates:
            continue
        if r['size'] < 220:
            continue
        if (r['rect'][2] * r['rect'][3]) < 500:
            continue
        proposal_img, proposal_vertice = clip_pic(img, r['rect'])
        if len(proposal_img) == 0:
            continue
        x, y, w, h = r['rect']
        if w == 0 or h == 0:
            continue
        [a, b, c] = np.shape(proposal_img)
        if a == 0 or b == 0 or c == 0:
            continue
        resized_proposal_img = resize_image(proposal_img, cfg.Image_size, cfg.Image_size)
        candidates.add(r['rect'])
        img_float = np.asarray(resized_proposal_img, dtype="float32")
        images.append(img_float)
        vertices.append(r['rect'])
    return images, vertices

def if_intersection(xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b, ymax_b):
    if_intersect = False
    if xmin_a < xmax_b <= xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_a <= xmin_b < xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_b < xmax_a <= xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    elif xmin_b <= xmin_a < xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    else:
        return if_intersect
    if if_intersect:
        x_sorted_list = sorted([xmin_a, xmax_a, xmin_b, xmax_b])
        y_sorted_list = sorted([ymin_a, ymax_a, ymin_b, ymax_b])
        x_intersect_w = x_sorted_list[2] - x_sorted_list[1]
        y_intersect_h = y_sorted_list[2] - y_sorted_list[1]
        area_inter = x_intersect_w * y_intersect_h
        return area_inter

class Train_Alexnet_Data(object):
    '''
    This class is used to process flower17 data and save it as a file
    '''
    def __init__(self):
        self.train_batch_size = cfg.T_batch_size
        self.image_size = cfg.Image_size

        self.train_list = cfg.Train_list
        self.train_class_num = cfg.T_class_num
        self.flower17_data = []
        self.data = cfg.DATA
        if not os.path.isdir(self.data):
            os.makedirs(self.data)
        self.epoch = 0
        self.cursor = 0
        self.load_17flowers()


    def load_17flowers(self,save_name='17flowers.pkl'):
        '''
        In the train_txt file, sequentially obtain the image address, picture category, etc. by column.
        Save image matrix data (img) and picture category data (lable) as a whole
        '''
        save_path = os.path.join(self.data, save_name)
        if os.path.isfile(save_path):
            self.flower17_data = pickle.load(open(save_path, 'rb'))
        else:
            with codecs.open(self.train_list, 'r', 'utf-8') as f:
                lines = f.readlines()
                print('load 17flowers data...........')
                for num, line in enumerate(lines):
                    context = line.strip().split(' ')
                    image_path = context[0]
                    index = int(context[1])
                    #print('load_17flowers: ',image_path)
                    img = cv2.imread(image_path)
                    img = resize_image(img, self.image_size, self.image_size)
                    img = np.asarray(img, dtype='float32')

                    label = np.zeros(self.train_class_num)
                    label[index] = 1
                    self.flower17_data.append([img, label])
                    view_bar("Process train_image of %s" % image_path, num + 1, len(lines))
            pickle.dump(self.flower17_data,open(save_path,'wb'))
            

    def get_batch(self):
        '''
        Call get_batch to get data for each round of training during network training
        '''
        images = np.zeros((self.train_batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros((self.train_batch_size, self.train_class_num))
        count = 0
        while( count < self.train_batch_size):
            images[count] = self.flower17_data[self.cursor][0]
            labels[count] = self.flower17_data[self.cursor][1]
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.flower17_data) :
                self.cursor = 0
                self.epoch += 1
                np.random.shuffle(self.flower17_data)
                print(self.epoch)
        return images, labels

class FineTun_And_Predict_Data(object):
    '''
    The functions and functions of this class are similar to the previous class
    '''
    def __init__(self, solver=None, is_svm=False, is_save=True):
        self.solver = solver
        self.is_svm = is_svm
        self.is_save = is_save

        self.fineturn_list = cfg.Finetune_list
        self.image_size = cfg.Image_size
        self.F_class_num = cfg.F_class_num
        self.R_class_num = cfg.R_class_num

        self.fineturn_batch_size = cfg.F_batch_size
        self.Reg_batch_size = cfg.R_batch_size

        self.fineturn_save_path = cfg.Fineturn_save
        if not os.path.isdir(self.fineturn_save_path):
            os.makedirs(self.fineturn_save_path)

        self.SVM_and_Reg_save_path = cfg.SVM_and_Reg_save
        if not os.path.isdir(self.SVM_and_Reg_save_path):
            os.makedirs(self.SVM_and_Reg_save_path)

        self.fineturn_threshold = cfg.F_fineturn_threshold
        self.svm_threshold = cfg.F_svm_threshold
        self.reg_threshold = cfg.F_regression_threshold

        self.SVM_data_dic = {}
        self.Reg_data = []
        self.fineturn_data = []

        self.cursor = 0
        self.epoch = 0
        if self.is_svm:
            if len(os.listdir(self.SVM_and_Reg_save_path)) == 0:
                self.load_2flowers()
        else:
            if len(os.listdir(self.fineturn_save_path)) == 0:
                self.load_2flowers()
        self.load_from_npy()

    def load_2flowers(self):
        with codecs.open(self.fineturn_list, 'r', 'utf-8') as f:
            lines = f.readlines()
            print('load 2flowers data...........')
            for num, line in enumerate(lines):
                labels = []
                labels_bbox = []
                images = []
                context = line.strip().split(' ')
                #print(context)
                image_path = context[0]
                
                if len(context) == 3 :
                    ref_rect = context[2].split(',')
                    ground_truth = [int(i) for i in ref_rect]
                else:
                    ref_rect = ['10','10','100','100']
                    ground_truth = [0,0,0,0]
                #print(image_path)
                #sleep(1000)
                img = cv2.imread(image_path)
                #print(img)
                #sleep(1000)
                img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
                candidate = set()
                for r in regions:
                    if r['rect'] in candidate:
                        continue
                    if r['size'] < 200 :
                        continue
                    if (r['rect'][2] * r['rect'][3]) <500:
                        continue
                    proposal_img, proposal_vertice = clip_pic(img, r['rect'])
                    if len(proposal_img) == 0:
                        continue
                    x, y, w, h = r['rect']
                    if w==0 or h==0 :
                        continue
                    [a, b, c] =np.shape(proposal_img)
                    if a==0 or b==0 or c==0 :
                        continue
                    resized_proposal_img = resize_image(proposal_img, self.image_size,self.image_size)
                    candidate.add(r['rect'])
                    img_float = np.asarray(resized_proposal_img, dtype="float32")
                    if self.is_svm:
                        feature = self.solver.predict([img_float])
                        images.append(feature[0])
                    else :
                        images.append(img_float)

                    iou_val = IOU(ground_truth, proposal_vertice)
                    px = float(proposal_vertice[0]) + float(proposal_vertice[4] / 2.0)
                    py = float(proposal_vertice[1]) + float(proposal_vertice[5] / 2.0)
                    ph = float(proposal_vertice[5])
                    pw = float(proposal_vertice[4])

                    gx = float(ref_rect[0])
                    gy = float(ref_rect[1])
                    gw = float(ref_rect[2])
                    gh = float(ref_rect[3])

                    index = int(context[1])
                    if self.is_svm:
                        if iou_val < self.svm_threshold:
                            labels.append(0)
                        else:
                            labels.append(index)
                        label = np.zeros(5)
                        label[1:5] = [(gx - px) / pw, (gy - py) / ph, np.log(gw / pw), np.log(gh / ph)]
                        if iou_val < self.reg_threshold:
                            label[0] = 0
                        else:
                            label[0] = 1
                        labels_bbox.append(label)

                    else:
                        label = np.zeros(self.F_class_num )
                        if iou_val < self.fineturn_threshold :
                            label[0] = 1
                        else:
                            label[index] = 1
                        labels.append(label)
                view_bar("Process SVM_and_Reg_image of %s" % image_path, num + 1, len(lines))
                if self.is_save:
                    if self.is_svm:
                        if not os.path.exists(os.path.join(self.SVM_and_Reg_save_path, str(context[1]))):
                            os.makedirs(os.path.join(self.SVM_and_Reg_save_path, str(context[1])))
                        np.save((os.path.join(self.SVM_and_Reg_save_path, str(context[1]), context[0].split('/')[-1].split('.')[0].strip())
                                                    + '_data.npy'),[images, labels, labels_bbox])
                    else:
                        np.save((os.path.join(self.fineturn_save_path, context[0].split('/')[-1].split('.')[0].strip()) +
                                                     '_data.npy'),[images, labels])

            

    def load_from_npy(self):
        if  self.is_svm:
            data_set = self.SVM_and_Reg_save_path
            data_dirs = os.listdir(data_set)
            for data_dir in data_dirs:
                SVM_data = []
                data_list = os.listdir(os.path.join(data_set, data_dir))
                for ind, d in enumerate(data_list):
                    i, l, k = np.load(os.path.join(data_set, data_dir,d),encoding='latin1')
                    for index in range(len(i)):
                        SVM_data.append([i[index], l[index]])
                        self.Reg_data.append([i[index], k[index]])
                    view_bar("Load SVM and Reg data of %s" % (data_dir+d), ind + 1, len(data_list))
                self.SVM_data_dic[data_dir] = SVM_data


        else:
            data_set = self.fineturn_save_path
            data_list = os.listdir(data_set)
            for ind, d in enumerate(data_list):
                i, l = np.load(os.path.join(data_set, d))
                for index in range(len(i)):
                    self.fineturn_data.append([i[index], l[index]])
                view_bar("Load fineturn data of %s" % d, ind + 1, len(data_list))

    def get_fineturn_batch(self):
        images = np.zeros((self.fineturn_batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros((self.fineturn_batch_size, self.F_class_num))
        count = 0
        while (count < self.fineturn_batch_size):
            images[count] = self.fineturn_data[self.cursor][0]
            labels[count] = self.fineturn_data[self.cursor][1]
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.fineturn_data):
                self.cursor = 0
                self.epoch += 1
                np.random.shuffle(self.fineturn_data)
                print(self.epoch)
        return images, labels


    def get_SVM_data(self, data_dir):
        images = []
        labels = []
        for index in range(len(self.SVM_data_dic[data_dir])):
            images.append(self.SVM_data_dic[data_dir][index][0])
            labels.append(self.SVM_data_dic[data_dir][index][1])
        return images, labels

    def get_Reg_batch(self):
        images = np.zeros((self.Reg_batch_size, 4096))
        labels = np.zeros((self.Reg_batch_size, self.R_class_num))
        count = 0
        while (count < self.Reg_batch_size):
            images[count] = self.Reg_data[self.cursor][0]
            labels[count] = self.Reg_data[self.cursor][1]
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.Reg_data):
                self.cursor = 0
                self.epoch += 1
                np.random.shuffle(self.Reg_data)
        return images,labels









