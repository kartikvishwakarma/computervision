# -*- coding: utf-8 -*-

import os
import codecs
import Networks
import numpy as np
import process_data
import config as cfg
import tensorflow as tf
from sklearn.externals import joblib
slim = tf.contrib.slim
flower = {1:'pancy', 2:'Tulip', 3:'Windflower', 4:'Iris'}

class Solver :
    '''
    This class is used to train or test a custom network structure. The custom network and data are raw materials,
	 Solver class belongs to the pot, based on network and data to achieve various functions

    parameter：net         --The network structure (custom) to be used for training or testing belongs to the class attribute
         data        --The data used to train the network belongs to the class attribute
         is_training --When this class is used to train the network is True, use the network to predict when false
         is_fineturn --True when this class is used for fineturn steps and feature extraction steps, and False for the rest
         is_Reg      --True when this class is used for bounding_box regression, false otherwise

    function：save_cfg()  ：Save the parameters in the network and the parameters in the training process as txt files
         train()     ：Used to train the network
         predict(input_data)  ：Input_data as the input of the network, get the results after the network is running
    '''
    def __init__(self, net, data, is_training=False, is_fineturn=False, is_Reg=False):
        self.net = net
        self.data = data
        self.is_Reg = is_Reg
        self.is_fineturn = is_fineturn
        self.summary_step = cfg.Summary_iter
        self.save_step = cfg.Save_iter
        self.max_iter = cfg.Max_iter
        self.staircase = cfg.Staircase

        if is_fineturn:
            self.initial_learning_rate = cfg.F_learning_rate
            self.decay_step = cfg.F_decay_iter
            self.decay_rate = cfg.F_decay_rate
            self.weights_file = cfg.T_weights_file
            self.output_dir = r'./output/fineturn'
        elif is_Reg:
            self.initial_learning_rate = cfg.R_learning_rate
            self.decay_step = cfg.R_decay_iter
            self.decay_rate = cfg.R_decay_rate
            if is_training == True:
                self.weights_file = None
            else:
                self.weights_file = cfg.R_weights_file
            self.output_dir = r'./output/Reg_box'
        else:
            self.initial_learning_rate = cfg.T_learning_rate
            self.decay_step = cfg.T_decay_iter
            self.decay_rate = cfg.T_decay_rate
            if is_training == True:
                self.weights_file = None
            else:
                self.weights_file = cfg.F_weights_file
            self.output_dir = r'./output/train_alexnet'
        self.save_cfg()

        #When the model and its parameters are restored, the parameters of the R-CNN/fc_11 network layer of the name are not loaded

        exclude = ['R-CNN/fc_11']
        self.variable_to_restore = slim.get_variables_to_restore(exclude=exclude)
        self.variable_to_save = slim.get_variables_to_restore(exclude=[])
        self.restorer = tf.train.Saver(self.variable_to_restore, max_to_keep=1)
        self.saver = tf.train.Saver(self.variable_to_save, max_to_keep=1)
        self.ckpt_file = os.path.join(self.output_dir, 'save.ckpt')

        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.output_dir)

        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable = False)
        self.learning_rate = tf.train.exponential_decay(
                                                         self.initial_learning_rate,
                                                         self.global_step,
                                                         self.decay_step,
                                                         self.decay_rate,
                                                         self.staircase,
                                                         name='learning_rate'
                                                        )
        if is_training :
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
                                                            self.net.total_loss ,global_step=self.global_step
                                                    )
            self.ema = tf.train.ExponentialMovingAverage(0.99)
            self.average_op = self.ema.apply(tf.trainable_variables())
            with tf.control_dependencies([self.optimizer]):
                self.train_op = tf.group(self.average_op)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if self.weights_file is not None:
            self.restorer.restore(self.sess, self.weights_file)
        self.writer.add_graph(self.sess.graph)

    def save_cfg(self):
        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)

    def train(self):
        for step in range(1, self.max_iter+1):
            if self.is_Reg:
                input, labels = self.data.get_Reg_batch()
            elif self.is_fineturn:
                input, labels = self.data.get_fineturn_batch()
            else:
                input, labels = self.data.get_batch()

            feed_dict = {self.net.input_data:input, self.net.label:labels}
            if step % self.summary_step == 0 :
                summary, loss, _=self.sess.run([self.summary_op,self.net.total_loss,self.train_op], feed_dict=feed_dict)
                self.writer.add_summary(summary, step)
                print("Data_epoch:"+str(self.data.epoch)+" "*5+"training_step:"+str(step)+" "*5+ "batch_loss:"+str(loss))
            else:
                self.sess.run([self.train_op], feed_dict=feed_dict)
            if step % self.save_step == 0 :
                print("saving the model into " + self.ckpt_file)
                self.saver.save(self.sess, self.ckpt_file, global_step=self.global_step)

    def predict(self, input_data):
        feed_dict = {self.net.input_data :input_data}
        predict_result = self.sess.run(self.net.logits, feed_dict = feed_dict)
        return predict_result

def get_Solvers():
    '''
    This function is used to get three Solvers: Solver for Feature Extraction,
	 Solver for SVM Predictive Classification, Solver for Reg_Box Prediction Box Regression

    :return:
    '''
    weight_outputs = ['train_alexnet', 'fineturn', 'SVM_model', 'Reg_box']
    for weight_output in weight_outputs:
        output_path = os.path.join(cfg.Out_put, weight_output)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    if len(os.listdir(r'./output/train_alexnet')) == 0:
        Train_alexnet = tf.Graph()
        with Train_alexnet.as_default():
            Train_alexnet_data = process_data.Train_Alexnet_Data()
            Train_alexnet_net = Networks.Alexnet_Net(is_training=True, is_fineturn=False, is_SVM=False)
            Train_alexnet_solver = Solver(Train_alexnet_net, Train_alexnet_data, is_training=True, is_fineturn=False, is_Reg=False)
            Train_alexnet_solver.train()

    if len(os.listdir(r'./output/fineturn')) == 0:
        print('Fineturn data....')
        Fineturn = tf.Graph()
        with Fineturn.as_default():
            print('loading data...')
            Fineturn_data = process_data.FineTun_And_Predict_Data()
            print('loading model...')
            Fineturn_net = Networks.Alexnet_Net(is_training=True, is_fineturn=True, is_SVM=False)
            print('calling Solver...')
            Fineturn_solver = Solver(Fineturn_net, Fineturn_data, is_training=True, is_fineturn=True, is_Reg=False)
            print('training started...')
            Fineturn_solver.train()
            print('training Done....')

    Features = tf.Graph()
    with Features.as_default():
        Features_net = Networks.Alexnet_Net(is_training=False, is_fineturn=True, is_SVM=True)
        Features_solver = Solver(Features_net, None, is_training=False, is_fineturn=True, is_Reg=False)
        Features_data = process_data.FineTun_And_Predict_Data(Features_solver, is_svm=True, is_save=True)

    svms = []
    if len(os.listdir(r'./output/SVM_model')) == 0:
        SVM_net = Networks.SVM(Features_data)
        SVM_net.train()
    for file in os.listdir(r'./output/SVM_model'):
        svms.append(joblib.load(os.path.join('./output/SVM_model', file)))

    Reg_box = tf.Graph()
    with Reg_box.as_default():
        Reg_box_data = Features_data
        Reg_box_net = Networks.Reg_Net(is_training=True)
        if len(os.listdir(r'./output/Reg_box')) == 0:
            Reg_box_solver = Solver(Reg_box_net, Reg_box_data, is_training=True, is_fineturn=False, is_Reg=True)
            Reg_box_solver.train()
        else:
            Reg_box_solver = Solver(Reg_box_net, Reg_box_data, is_training=False, is_fineturn=False, is_Reg=True)

    return Features_solver, svms, Reg_box_solver

if __name__ =='__main__':

    Features_solver, svms, Reg_box_solver =get_Solvers()


    with codecs.open(cfg.Finetune_list, 'r', 'utf-8') as f:
        lines = f.readlines()
        print('load 2flowers data...........')
        for num, line in enumerate(lines):
            labels = []
            labels_bbox = []
            images = []
            context = line.strip().split(' ')
            #print(context)
            image_path = context[0]

            #img_path = './2flowers/jpg/0/image_0566.jpg'  # or './17flowers/jpg/16/****.jpg'
            imgs, verts = process_data.image_proposal(image_path)
            #process_data.show_rect(img_path, verts, ' ')
            features = Features_solver.predict(imgs)
            print(np.shape(features))

            results = []
            results_old = []
            results_label = []
            count = 0
            for f in features:
                for svm in svms:
                    pred = svm.predict([f.tolist()])
                    # not background
                    if pred[0] != 0:
                        results_old.append(verts[count])
                        #print(Reg_box_solver.predict([f.tolist()]))
                        if Reg_box_solver.predict([f.tolist()])[0][0] > 0.5:
                            px, py, pw, ph = verts[count][0], verts[count][1], verts[count][2], verts[count][3]
                            old_center_x, old_center_y = px + pw / 2.0, py + ph / 2.0
                            x_ping, y_ping, w_suo, h_suo = Reg_box_solver.predict([f.tolist()])[0][1], \
                                                           Reg_box_solver.predict([f.tolist()])[0][2], \
                                                           Reg_box_solver.predict([f.tolist()])[0][3], \
                                                           Reg_box_solver.predict([f.tolist()])[0][4]
                            new__center_x = x_ping * pw + old_center_x
                            new__center_y = y_ping * ph + old_center_y
                            new_w = pw * np.exp(w_suo)
                            new_h = ph * np.exp(h_suo)
                            new_verts = [new__center_x, new__center_y, new_w, new_h]
                            results.append(new_verts)
                            results_label.append(pred[0])
                count += 1

            average_center_x, average_center_y, average_w,average_h = 0, 0, 0, 0

            #Gives an average of all forecasted forecasted boxes, representing their predicted final position

            if len(results):
                for vert in results:
                    average_center_x += vert[0]
                    average_center_y += vert[1]
                    average_w += vert[2]
                    average_h += vert[3]
                average_center_x = average_center_x / len(results)
                average_center_y = average_center_y / len(results)
                average_w = average_w / len(results)
                average_h = average_h / len(results)
                average_result = [[average_center_x, average_center_y, average_w, average_h]]
                result_label = max(results_label, key=results_label.count)
                #process_data.show_rect(img_path, results_old,' ')
                process_data.show_rect(image_path, average_result,flower[result_label])
            
            else:
                print('No features found....')








