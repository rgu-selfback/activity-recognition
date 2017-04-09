
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
from sklearn.metrics import *
from sklearn import preprocessing as prep
import pickle
from sklearn.externals import joblib
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import time

import lasagne
import theano
import theano.tensor as T

import selfback_utils as su
import selfback_stepcounter as sc


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
	num_classes = len(set(y_train))

	#########  Build CNN  #########################################	
	#self.data_width = int(X_train.shape[2]/10) 
        #X_train = X_train.reshape(X_train.shape[0], 3, -1)
	print X_train.shape

	l_in = lasagne.layers.InputLayer(shape=(None, X_train.shape[1], X_train.shape[2]))

	conv_network = l_in

	#Build Convolution layers
	for l in range(self.params['conv_layers']):
		conv_network = lasagne.layers.Conv1DLayer(conv_network, num_filters=self.params['conv_filter_num'][l], filter_size=self.params['conv_filter_dims'], nonlinearity=lasagne.nonlinearities.rectify)

		print 'l_conv%d output: '%l+str(lasagne.layers.get_output_shape(conv_network))
	
		conv_network = lasagne.layers.MaxPool1DLayer(conv_network, pool_size=self.params['pool_size'])

		print 'l_pool%d output: '%l+str(lasagne.layers.get_output_shape(conv_network))
	
	conv_output = lasagne.layers.get_output(conv_network)
	
	network = conv_network
	#Build fully connected hidden layers
	for i in range(self.params['hid_layers']):
     		units = self.params['hid_units'][i]	
		network = lasagne.layers.DenseLayer(network, num_units=units, nonlinearity=lasagne.nonlinearities.tanh)	
		network = lasagne.layers.DropoutLayer(network, p=0.5)

	#Build output layer
	network = lasagne.layers.DenseLayer(network, num_units=num_classes, nonlinearity=lasagne.nonlinearities.softmax)

	
	input_var = T.tensor4('inputs')
	target_var = T.ivector('targets')
	
	predictions = lasagne.layers.get_output(network)
	conv_weights = lasagne.layers.get_output(conv_network)

	self.classifier = theano.function([l_in.input_var],predictions) 
	
	self.cnn_weights = theano.function([l_in.input_var], conv_output)	
	
	loss = lasagne.objectives.categorical_crossentropy(predictions, target_var)
	loss = loss.mean()
	
	params = lasagne.layers.get_all_params(network, trainable=True)
	updates = lasagne.updates.nesterov_momentum(loss,params, learning_rate=0.01, momentum=0.9)

	test_prediction = lasagne.layers.get_output(network, deterministic=True)
    	test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    	test_loss = test_loss.mean()
    	# As a bonus, also create an expression for the classification accuracy:
    	test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

    	# Compile a function performing a training step on a mini-batch (by giving
    	# the updates dictionary) and returning the corresponding training loss:
    	train_fn = theano.function([l_in.input_var, target_var], loss, updates=updates)

    	# Compile a second function computing the validation loss and accuracy:
    	self.val_fn = theano.function([l_in.input_var, target_var], [test_loss, test_acc])
	
	num_epochs = self.params['epochs']
	for epoch in range(num_epochs):
		start_time = time.time()
		train_err = 0
		train_batches = 0
		for batch in self.iterate_batches(X_train, y_train):
			inputs, targets = batch
			train_err += train_fn(inputs, targets.astype(np.int32))
			train_batches += 1

		#val_err = 0
		#val_acc = 0
		#val_batches = 0
		#for batch in self.iterate_batches(X_val, y_val):
			#inputs, targets = batch
		#err, acc = val_fn(X_val, y_val)
		#val_err += err
		#val_acc += acc
		#val_batches += 1

		# Then we print the results for this epoch:
		print("Epoch {} of {} took {:.3f}s".format(
		    epoch + 1, num_epochs, time.time() - start_time))
		print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
		#print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
		#print("  validation accuracy:\t\t{:.2f} %".format(
		    #val_acc / val_batches * 100))
	
	cnn_X_train = self.cnn_weights(X_train)
	cnn_X_train = cnn_X_train.reshape([cnn_X_train.shape[0], -1])
	self.svm = SVC()
	self.svm = self.svm.fit(cnn_X_train, y_train)
			
    
    def predict_activities(self, X_test, format_result=True):
	print 'Making Predicitons...'
	print X_test.shape
	#X_test = X_test.reshape(X_test.shape[0], 3, -1)
	pred_activities = []
        if self.val_fn is not None:
            # After training, we compute and print the test error:
	    test_err = 0
	    test_acc = 0
	    test_batches = 0
	    #for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
		#inputs, targets = batch
	    err, acc = self.val_fn(X_test, y_test.astype(np.int32))
	    test_err += err
	    test_acc += acc
	    test_batches += 1
	    print("Final results:")
	    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
	    print("  test accuracy:\t\t{:.2f} %".format(
		test_acc / test_batches * 100))
             
                       
            pred_activities = np.argmax(self.classifier(X_test), axis=1) 
	    print pred_activities

	    cnn_X_test = self.clf_weights(X_test)
	    cnn_X_test = cnn_X_test.reshape([cnn_X_test.shape[0], -1])
	    svm_preds = self.svm.predict(cnn_X_test)

            #print self.le.inverse_transform(pred_activities)
	elif self.classifier is not None:
	    X_test_norm = self.scaler.transform(X_test) 
	    pred_activities = self.classifier.predict(X_test_norm)
            
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

    results_path = 'results/'
    results_filename = '1D_Results_raw.txt'
    if not os.path.exists(results_path+results_filename):
	results_file = open(results_path+results_filename, 'w+')
    else:
	results_file = open(results_path+results_filename, 'a+')
    
    data_path = 'activity_data1/'
    person_data = su.read_train_data(data_path) 
    print len(person_data)
    instance_ids = person_data.keys()   
    
    for f_dim in [10]:
	    #ar = ActivityRecogniser('models')
	    ar = ActivityRecogniser(modelpath=None)  
	    np.random.seed(123)
	    test_inds = np.random.randint(0,len(person_data),10) 
	    test_cases = [instance_ids[ind] for ind in test_inds]
	    print 'Test cases: '+str(test_cases)
	    train_data = [value for key, value in person_data.items() if key not in test_cases]
	    test_data = [value for key, value in person_data.items() if key in test_cases]


	    #############################################################################
	    ####################  Train Model  ##########################################
	    #############################################################################
	    
	    params = { 
	    'win_len' : 3, 
	    'epochs': 5,
	    'conv_layers':1,
	    'conv_filter_dims' : f_dim,
	    'conv_filter_num': [40,130],  
	    'pool_size': 2,  
	    'samp_rate' : 100, 
	    'hid_layers' : 1,
	    'hid_units' : [200],
	    'feat_type' : '3d_raw',
	    'scale_feats': False,
	    'n_comps': None
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
	    predictions = ar.predict_activities(X_test, format_result=False)

	    y_true = le.inverse_transform(y_test)
	    y_pred = le.inverse_transform(predictions)
	    class_report = classification_report(y_true, y_pred)
	    print class_report
	    f1 = f1_score(y_true, y_pred, average='micro')
	    print 'f1 Score: %f'%f1

	    ############  Write Results  #########################################
	    results_file.write(str(params)+'\n\n')
	    results_file.write(str(class_report)+'\n')
	    results_file.write('f1 Score: '+str(f1)+'\n\n\n')
    results_file.write("==============================================================\n\n")
    results_file.close()
    #print '#################    '+activity+'    ###################'
	#print predictions
