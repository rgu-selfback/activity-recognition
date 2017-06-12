
# coding: utf-8

# In[3]:
from __future__ import division
import os
import numpy as np
import pandas as pd
import sys
#import matplotlib
#import matplotlib.pyplot as plt
import scipy
from scipy import stats
from scipy import fftpack
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
from sklearn import preprocessing as prep
import pickle
from sklearn.externals import joblib
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import time

#import lasagne
#import theano
#import theano.tensor as T

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

import selfback_utils as su
import selfback_stepcounter as sc

import atexit


###################################################
########### ACTVITY RECOGNITION ##################
##################################################

class ActivityRecogniser:
    
    amb_activities = ['walking', 'stairs', 'running']
    classifier = None
    scaler = None
    val_fn = None
    le = None
    feat_ext_methods = {'dct':su.extract_dct_features, 'raw':su.extract_raw_features, '3d_raw':su.extract_3d_raw_features, '3d_dct':su.extract_3d_dct_features, '3d_fft':su.extract_3d_fft_features, '3d_raw_dct':su.extract_3d_raw_dct_features, '3d_raw_fft':su.extract_3d_raw_fft_features}

    def __init__(self, modelpath='model'):
	self.model = Sequential()
        if modelpath is not None:
            self.load_models(modelpath)
    
    def load_models(self, path):        
        self.classifier = joblib.load(path+'/classifier.pkl')
        self.scaler = joblib.load(path+'/scaler.pkl')

    def iterate_batches(self, X, y, size=500):
	indices = range(len(y))
	np.random.shuffle(indices)
	#print type(indices)
	for i in range(0, len(indices)-size+1, size):
		excerpt = indices[i:i+size]
		#print excerpt
		yield X[excerpt, :], y[excerpt]

    def relabel(self, y):
	  y = su.relabel(y) 
    	  #labels = pd.Series(y)
    	  #print labels.unique()


    def extract_features(self, train_data, method='dct', samp_rate=100, window_length=10, overlap=0.5, n_comps=None):
	#########  Partition Data into Time Windows  #############################################
	time_windows = []
        for data in train_data:  
            #print len(data)
            for activity in data:
                df = data[activity]                    
                windows = su.split_windows(df, samp_rate, window_length, overlap_ratio=overlap)                    
                time_windows.extend(windows)    
        
        #########  Extract Features  #############################################            
        X, y = self.feat_ext_methods[method](time_windows, feat_len = (samp_rate * window_length), class_attr='class', n_comps=n_comps)  
	#print X.shape
	return X, y  
		
   
 
    def train_cnn_model(self, X_train, y_train):
	print 'Training CNN model....'        
	print X_train.shape
	print y_train[0]
	num_classes = len(set(y_train))
	X_train = X_train.reshape([-1, 1, 500, 3])
	#y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)

	#########  Build CNN  #########################################

	#Build Convolution layers

	self.model.add(Conv2D(self.params['conv_filter_num'][0], (1, self.params['conv_filter_dims']), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))		
	
	self.model.add(MaxPooling2D(pool_size=(1,self.params['pool_size'])))	
	
	for l in range(1,self.params['conv_layers']):
		self.model.add(Conv2D(self.params['conv_filter_num'][l], (1, self.params['conv_filter_dims']), activation='relu'))		
	
		self.model.add(MaxPooling2D(pool_size=(1,self.params['pool_size'])))		

	self.model.add(Flatten())	
	
	#Build fully connected hidden layers
	for i in range(self.params['hid_layers']):
		self.model.add(Dense(self.params['hid_units'][i], activation='tanh'))
		self.model.add(Dropout(0.5))

	#Build output layer
	self.model.add(Dense(num_classes, activation='softmax'))
	
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

	self.model.compile(loss='categorical_crossentropy', optimizer=sgd)
	
	y_train = keras.utils.to_categorical(y_train, num_classes)

	self.model.fit(X_train, y_train, epochs=self.params['epochs'], batch_size=500)
			
    
    def predict_activities(self, X_test, format_result=True):
	print 'Making Predicitons...'
	print X_test.shape
	X_test = X_test.reshape(-1, 1, 500, 3)
	pred_activities = []
        if self.model is not None:
                      
            pred_activities = np.argmax(self.model.predict(X_test, batch_size=500), axis=1) 
	    #print pred_activities

	    #cnn_X_test = self.cnn_weights(X_test)
	    #cnn_X_test = cnn_X_test.reshape([cnn_X_test.shape[0], -1])
	    #svm_preds = self.svm.predict(cnn_X_test)
	    #knn_preds = self.knn.predict(cnn_X_test)

            #print self.le.inverse_transform(pred_activities)
        else:
            raise InitError()

	if not format_result:          
	    return pred_activities
	else:
	    return self.format_output(windows, pred_activities)
            
    
   
#############################################################################
####################  Main  ##########################################
#############################################################################     
    
if __name__ == '__main__':

    dataset = 'wrist34'
    results_path = 'results/'
    results_filename = 'cnn_knn_results_'+dataset+'_tw5.txt'
    if not os.path.exists(results_path+results_filename):
	results_file = open(results_path+results_filename, 'w+')
    else:
	results_file = open(results_path+results_filename, 'a+')
    
    data_path = 'Datasets/raw/'+dataset+'/'
    lh_path = 'activity_data_wrist34/' 
    print 'Reading data: '+data_path[:-1]
    person_data = su.read_train_data(data_path)
    #lh_data = su.read_train_data(lh_path) 
    print len(person_data)
    instance_ids = person_data.keys()   
    
    for var0 in range(5,6):
    	for var in range(len(instance_ids)):
    	    index = []
	    index.append(var)
    	    all_results = {}
    	    nn_results = []
    	    svm_results = []
	    knn_results = []

	    nn_precisions = []
	    svm_precisions = []
	    knn_precisions = []

	    nn_recalls = []
  	    svm_recalls = []
	    knn_recalls = []

	    nn_accuracies = []
	    svm_accuracies = []
	    knn_accuracies = []
	
	    all_results['f1_NNet'] = nn_results
   	    all_results['f1_SVM'] = svm_results
	    all_results['f1_kNN'] = knn_results
	    all_results['prec_NNet'] = nn_precisions
	    all_results['prec_SVM'] = svm_precisions
	    all_results['prec_kNN'] = knn_precisions
	    all_results['recall_NNet'] = nn_recalls
	    all_results['recall_SVM'] = svm_recalls
 	    all_results['recall_kNN'] = knn_recalls
	    all_results['acc_NNet'] = nn_accuracies
	    all_results['acc_SVM'] = svm_accuracies
  	    all_results['acc_kNN'] = knn_accuracies
	
	    print 'Run: '+str(var)
	    #ar = ActivityRecogniser('models')
	    ar = ActivityRecogniser(modelpath=None)  
	    np.random.seed(123)
 	    test_inds = [var]
	    #test_inds = np.random.choice(range(len(person_data)),8,replace=False) 
	    test_cases = [instance_ids[ind] for ind in test_inds]
	    train_data = [value for key, value in person_data.items() if key not in test_cases]
	    #train_data = [value for key, value in lh_data.items() if key not in test_cases]
	    test_data = [value for key, value in person_data.items() if key in test_cases]
	    #test_data = [value for key,value in lh_data.items() if key in test_cases]
	    #test_cases.extend(test_cases)
 	    print 'Test cases: '+str(test_cases)
	    

	    #############################################################################
	    ####################  Train Model  ##########################################
	    #############################################################################
	    
	    params = { 
	    'win_len' : 5, 
	    'epochs': 200,
	    'conv_layers':var0,
	    'conv_filter_dims' : 10,
	    'conv_filter_num': [150,100,80,60,40,20],
	    #'conv_filter_num': [40,30,20,10,10],  
	    'pool_size': 2,  
	    'samp_rate' : 100, 
	    'hid_layers' : 2,
	    'hid_units' : [900,300],
	    'feat_type' : '3d_raw',
	    'scale_feats': False,
	    'n_comps': None,
	    'train_data': dataset,
	    'test_data': dataset
	    }
	    print params
	    ar.params = params
	    results_file.write(str(params)+'\n')

	    ##### Extract Features #####
	    X_train, y_train = ar.extract_features(train_data, window_length=params['win_len'], method=params['feat_type'], n_comps=params['n_comps']) 
	    num_classes = len(set(y_train))
	    params['num_classes'] = num_classes
	    params['data_depth'] = X_train.shape[1]

	    print 'train data dims: '+str(X_train.shape)

	    ##### Change Data Label Granularity #####  
	    #y_train = su.relabel(y_train)

	    ##### Encode Class Labels #####	
	    le = LabelEncoder()
	    y_train = le.fit_transform(y_train)	

	    ##### Scale Features #####
	    scaler = prep.StandardScaler()
	    if params['scale_feats']:
	    	X_train = scaler.fit_transform(X_train.reshape([-1,params['win_len'] * params['samp_rate']]))
	    	X_train = X_train.reshape([-1,params['data_depth'],params['win_len'] * params['samp_rate']])

	    ##### Train Model ##### 
	    params['data_width'] = X_train.shape[2]  
	    ar.train_cnn_model(X_train, y_train)        

	    
	    #####################################################################
	    ############  Classify Test Set  ####################################
	    #####################################################################

	    ##### Extact Features #####
	    X_test, y_test = ar.extract_features(test_data, window_length=params['win_len'], method=params['feat_type'], n_comps=params['n_comps'])
	    #X_test, y_test = su.extract_raw_features(test_data)
	    print 'test data dims: '+str(X_test.shape)

	    ##### Change Data Labels Granularity #####
	    #y_test = su.relabel(y_test)

	    ##### Encode Class Labels #####	
	    y_test = le.transform(y_test)

	    ##### Scale Features #####
	    if params['scale_feats']:
	    	X_test = scaler.transform(X_test.reshape([-1, params['win_len'] * params['samp_rate']]))
	    	X_test = X_test.reshape([-1, params['data_depth'], params['win_len'] * params['samp_rate']])

	    ##### Classify Samples #####
	    #cnn_preds, svm_preds, knn_preds = ar.predict_activities(X_test, format_result=False)
	    cnn_preds = ar.predict_activities(X_test, format_result=False)

	    y_true = le.inverse_transform(y_test)
	    y_pred_cnn = le.inverse_transform(cnn_preds)
	    #y_pred_svm = le.inverse_transform(svm_preds)
	    #y_pred_knn = le.inverse_transform(knn_preds)

	    cnn_class_report = classification_report(y_true, y_pred_cnn)
	    #svm_class_report = classification_report(y_true, y_pred_svm)
	    #knn_class_report = classification_report(y_true, y_pred_knn)

	    print cnn_class_report
	    #print svm_class_report
	

	    f1_cnn = f1_score(y_true, y_pred_cnn, average='macro')
	    print 'f1 Score CNN: %f'%f1_cnn
	    nn_results.append(f1_cnn)

	    #f1_svm = f1_score(y_true, y_pred_svm, average='macro')
	    #print 'f1 Score SVM: %f'%f1_svm
            #svm_results.append(f1_svm)

	    #f1_knn = f1_score(y_true, y_pred_knn, average='macro')
	    #print 'f1 Score kNN: %f'%f1_knn
	    #knn_results.append(f1_knn)

	    prec_nn = precision_score(y_true, y_pred_cnn, average='macro')
	    #prec_svm = precision_score(y_true, y_pred_svm, average='macro')
	    #prec_knn = precision_score(y_true, y_pred_knn, average='macro')
	    nn_precisions.append(prec_nn)
	    #svm_precisions.append(prec_svm)
	    #knn_precisions.append(prec_knn)

	    recall_nn = recall_score(y_true, y_pred_cnn, average='macro')
	    #recall_svm = recall_score(y_true, y_pred_svm, average='macro')
	    #recall_knn = recall_score(y_true, y_pred_knn, average='macro')
	    nn_recalls.append(recall_nn)
	    #svm_recalls.append(recall_svm)
	    #knn_recalls.append(recall_knn)

	    acc_nn = accuracy_score(y_true, y_pred_cnn)
   	    #acc_svm = accuracy_score(y_true, y_pred_svm)
	    #acc_knn = accuracy_score(y_true, y_pred_knn)
	    nn_accuracies.append(acc_nn)
	    #svm_accuracies.append(acc_svm)
	    #knn_accuracies.append(acc_knn)

	    ############  Write Results  #########################################
	    results_file.write(str(params)+'\n\n')
	    results_file.write('CNN:\n'+str(cnn_class_report)+'\n')
	    results_file.write('f1 Score: '+str(f1_cnn)+'\n')
	    #results_file.write('SVM:\n'+str(svm_class_report)+'\n')
	    #results_file.write('f1 Score: '+str(f1_svm)+'\n')
	    #results_file.write('kNN:\n'+str(knn_class_report)+'\n')
	    #results_file.write('f1 Score: '+str(f1_knn)+'\n\n\n')
    	    df = pd.DataFrame(all_results, index=index)
            csv_data_path = results_path+'_'+str(params['conv_layers'])+'cnn_svm_knn_'+dataset+'_tw5_'+str(params['epochs'])+'_large1.csv'
    	    if os.path.exists(csv_data_path):
		with open(csv_data_path, 'a+') as f:
	     	    df.to_csv(f, header=False)
    	    else:
		df.to_csv(csv_data_path)
 
    results_file.write("==============================================================\n\n")
    results_file.close()
    #print '#################    '+activity+'    ###################'
	#print predictions
