
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

import selfback_utils as su
import selfback_stepcounter as sc


###################################################
########### ACTVITY RECOGNITION ##################
##################################################

class ActivityRecogniser:
    
    amb_activities = ['walking', 'stairs', 'running']
    classifier = None
    scaler = None
    #label_map = {'standing': 'inactive', 'sitting':'inactive', 'lying':'inactive', 'jogging':'active', 'walk_slow':'active', 'walk_mod':'active', 'walk_fast':'active', 'upstairs':'active', 'downstairs': 'active'}
    label_map = {'sedentary':'inactive', 'walking':'active', 'stairs':'active', 'running':'active'}

    feat_ext_method = {
	'dct': su.extract_dct_features, 
	'fft': su.extract_fft_features	
    }

    def __init__(self, modelpath='model'):
        if modelpath is not None:
            self.load_models(modelpath)
    
    def load_models(self, path):        
        self.classifier = joblib.load(path+'/classifier.pkl')
        self.scaler = joblib.load(path+'/scaler.pkl')

    def extract_windows(self, dataset, samp_rate, window_length, overlap_ratio):
	time_windows = []
	for data in dataset:  
            #print len(data)
            for activity in data:
                df = data[activity]                    
                windows = su.split_windows(df, samp_rate, window_length, overlap_ratio=overlap_ratio)                    
                time_windows.extend(windows) 
	return time_windows   
    
    
    def train_model(self, train_data, feat_len=48, samp_rate=100, window_length=10):
        ########  Split train data into time windows  ############################################    
        time_windows_train = self.extract_windows(train_data, samp_rate, window_length, 0.5)
        
        
        #########  Extract Features  #############################################            
        X_train, y_train = self.feat_ext_method[self.params['feat_type']](time_windows_train, class_attr='class', n_comps=feat_len)                 
        #X_train, y_train = su.extract_features(time_windows_train)
	print X_train.shape
        
        #############  Scale Features  ######################################
        self.scaler = prep.StandardScaler()
        #scaler = prep.MinMaxScaler()
        X_train_norm = self.scaler.fit_transform(X_train)
        
        #############  Apply PCA  ######################################
        #pca = PCA(n_components = int(n_features*n_comps))
        #X_train_norm = pca.fit_transform(X_train_norm)         
        
        ##########  Change Data Labels Granularity ##########################  
        #y_train = su.relabel(y_train) 
        self.train_labels = y_train
          
        #########  Train Classifier  #########################################
        clf = SVC()            
        clf = clf.fit(X_train_norm, y_train) 
        self.classifier = clf  
        
        ########  Persist Models  ########################################## 
        path = 'models'
        if not os.path.isdir(path):
            os.makedirs(path)
        joblib.dump(clf, path+'/classifier.pkl')
        joblib.dump(self.scaler, path+'/scaler.pkl')      
    
    
    def predict_activities(self, test_data, feat_len=48,  samp_rate=100, window_length=10, format_result=True):
        if self.classifier is not None:
            windows = self.extract_windows(test_data, samp_rate, window_length, 0.0)                            
            X_test, y_test = self.feat_ext_method[self.params['feat_type']](windows, class_attr='class', n_comps=feat_len)
	    #X_test, y_test = su.extract_features(windows)
	    print X_test.shape

	    #y_test = su.relabel(y_test)  
	    self.test_labels = y_test        
            
            ##########  Scale Features #####################################
            X_test_norm = self.scaler.transform(X_test) 
            
            ##########  Apply PCA  ######################################### 
            #X_test_norm = pca.transform(X_test_norm)    
                                        
            pred_activities = self.classifier.predict(X_test_norm) 
            
            if not format_result:          
                return pred_activities
            else:
                return self.format_output(windows, pred_activities)
        else:
            raise InitError()
            
    
    def get_stepcounts(self, df, samp_rate=100):
        x = df['x']
        y = df['y']
        z = df['z']
        mag = su.mag(x,y,z)
        steps = sc.count_steps2(mag, samp_rate)
        return steps
            
    
    def format_output(self, windows, predictions):
        output = []
        df = windows[0]
        start_time = df['time'].iloc[0]
        end_time = df['time'].iloc[len(df)-1]
        activity = activity = predictions[0]
        temp = df
        for i in range(1, len(windows)): 
            df = windows[i] 
            if predictions[i] != activity:                
                if activity in self.amb_activities:
                    step_count = self.get_stepcounts(temp)              
                    output.append({'timestamp_start':start_time, 'timestamp_end':end_time, 'activity':activity, 'steps':step_count})
                else:
                    output.append({'timestamp_start':start_time, 'timestamp_end':end_time, 'activity':activity})
                temp = df
                activity = predictions[i]
                start_time = df['time'].iloc[0]
                end_time = df['time'].iloc[len(df)-1] 
            else:
                temp = pd.concat([temp, df])
                end_time = df['time'].iloc[len(df)-1]                         
        if activity in self.amb_activities:
            step_count = self.get_stepcounts(temp)              
            output.append({'timestamp_start':start_time, 'timestamp_end':end_time, 'activity':activity, 'steps':step_count})
        else:
            output.append({'timestamp_start':start_time, 'timestamp_end':end_time, 'activity':activity})
        return output
        
class InitError(Exception):    
    def __str__(self):
        return "Model not initialised. Please provide path to model in constructor or train new model using 'train_model' method."
    
    
if __name__ == '__main__':
    
    exp_name = '4class'
    data_path = 'activity_data_wrist34/'
    lh_path = 'activity_data_wrist34/'
    person_data = su.read_train_data(data_path) 
    lh_data = su.read_train_data(lh_path)
    feat_types = ['dct']

    results_path = 'results/'
    results_filename = 'SVM_wrist34_results1.txt'
    if os.path.exists(results_path+results_filename):
    	results_file = open(results_path+results_filename, 'a+')
    else:
	results_file = open(results_path+results_filename, 'w+')    
    instance_ids = person_data.keys() 
    all_results = {}  
    for var1 in feat_types:
	    feat_results = []
	    for var2 in range(len(instance_ids)):
		    params = {
		    'feat_type':var1,
		    'feat_len': 80,
		    'win_len': 3,
		    'train_data': 'wrist',
		    'test_data': 'wrist'
		    }
		    print params
		    #ar = ActivityRecogniser('models')
		    ar = ActivityRecogniser(None)
		    ar.params = params
		    y_true = []
		    y_pred = []
		    f1_scores = []
		    
		    np.random.seed(var2)
		    #test_inds = np.random.choice(range(len(person_data)),8,replace=False)
		    test_inds = [var2] 	   
		    test_cases = [instance_ids[ind] for ind in test_inds]
		    results_file.write(str(test_cases)+'\n')
		    train_data = [value for key, value in person_data.items() if key not in test_cases]
		    #train_data = [value for key, value in lh_data.items() if key not in test_cases]
		    test_data = [value for key, value in person_data.items() if key in test_cases]
		    #test_data = [value for key, value in lh_data.items() if key in test_cases]
		    #test_cases.extend(test_cases)
	    	    print 'Test cases: '+str(test_cases)


		    '''for ind in range(1):
			print 'Test instance: '+str(ind)
			test_case = instance_ids[ind]        
			#print 'Test case: '+test_case
			train_data = [value for key, value in person_data.items() if key not in [test_case]]
			test_data = person_data[test_case]'''        
	
		    #############  Train Model  ##########################################        
		    ar.train_model(train_data, window_length=3, feat_len=params['feat_len'])        

		    ############  Classify Test Set  ####################################
		    user_y_true = []
		    user_y_pred = []
		    #for activity in test_data:  
		    	#print '#################    '+activity+'    ###################'
		    	#df = test_data[activity]      
		    labels_pred = ar.predict_activities(test_data, window_length=3, feat_len=params['feat_len'], format_result=False) 
		    labels_true = ar.test_labels
		    
		    #labels_pred = su.relabel(labels_pred, ar.label_map)
		    #labels_true = su.relabel(ar.test_labels, ar.label_map)

		    user_y_true.extend(labels_true)
		    user_y_pred.extend(labels_pred)
		    user_f1_score = su.f1_score(user_y_true, user_y_pred)    
		    f1_scores.append(user_f1_score)
		    y_true.extend(user_y_true)
		    y_pred.extend(user_y_pred)
			    #print y_pred
		    results_file.write(str(params)+'\n')
		    class_report = classification_report(y_true, y_pred)
		    print class_report
		    results_file.write(str(class_report)+'\n')
		    f1 = f1_score(y_true, y_pred, average='micro')
 		    feat_results.append(f1)
		    print 'f1 Score: %f\n\n'%f1
		    results_file.write(str(f1)+'\n')

		    out_file =  open('./results/'+exp_name+'.csv', 'w+')
		    for val in f1_scores:
			out_file.write(str(val)+'\n\n\n')
		    out_file.close()
	    all_results[var1] = feat_results
    df = pd.DataFrame(all_results)
    df.to_csv('./results/'+data_path+'.csv', index=False)
    results_file.close()







