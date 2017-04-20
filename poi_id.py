#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.grid_search import GridSearchCV

"""
financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 
'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 
'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 
'from_this_person_to_poi', 'poi', 'shared_receipt_with_poi'] 

POI label: ['poi']
"""

# create my evaluation metrics precision and recall
def compute_precision_recall(predictions,true_labels):
    t_p = 0
    t_n = 0
    f_p = 0
    f_n = 0
    for i in range(len(predictions)):
        if (predictions[i] == 1 and true_labels[i] == 1):
            t_p += 1
        if (predictions[i] == 1 and true_labels[i] == 0):
            f_p += 1
        if (predictions[i] == 0 and true_labels[i] == 0):
            t_n += 1
        if (predictions[i] == 0 and true_labels[i] == 1):
            f_n += 1
    
    try:    
        recal = (t_p + 0.0)/(t_p + f_n)
    except:
        recal = 0.0
    try:
        precision = (t_p + 0.0)/(t_p + f_p)
    except:
        precision = 0.0
    
    return precision,recal

####################################################
### Task 1: Select what features you'll use.
####################################################

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list=["poi","exercised_stock_options","restricted_stock","other","expenses"] 
#features_list = ['poi','salary','deferral_payments','total_payments','loan_advances','bonus','restricted_stock_deferred','deferred_income','total_stock_value','expenses','exercised_stock_options','other','long_term_incentive','restricted_stock','director_fees','from_poi_to_this_person','from_this_person_to_poi','shared_receipt_with_poi']

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

####################################################
### Task 2: Remove outliers
####################################################
data_dict.pop("TOTAL",0)

####################################################
### Task 3: Create new feature(s)
####################################################
def computeFraction(poi_messages,all_messages):
    fraction = 0.
    if (poi_messages == 'NaN' or all_messages == 'NaN' or all_messages == 0):
        return 0.0
    else:
        fraction = (poi_messages + 0.0)/all_messages
        return fraction

for name in data_dict:
    from_poi_to_this_person = data_dict[name]["from_poi_to_this_person"]
    to_messages = data_dict[name]["to_messages"]
    fraction_from_poi = computeFraction(from_poi_to_this_person,to_messages)
    data_dict[name]["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_dict[name]["from_this_person_to_poi"]
    from_messages = data_dict[name]["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    data_dict[name]["fraction_to_poi"] = fraction_to_poi

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

# do the feature scaling
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features,labels, test_size=0.4,random_state = 0)


clf = tree.DecisionTreeClassifier()
clf.fit(features_train,labels_train)

pred = clf.predict(features_test)
precision,recall = compute_precision_recall(pred,labels_test)
print "precision " + str(precision)
print "recall " + str(recall)

#############################################################
### Task 4: Try a varity of classifiers
#############################################################

### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html    
    




#parameters = {'C':[0.2,0.4,0.6,0.8,1.0],'kernel':('rbf','linear','poly'),'degree':[2,3,4,5]}
#temp_clf = SVC()
#parameters = {'n_neighbors':[2,3,4,5,6,7]}
#temp_clf = KNeighborsClassifier()
#parameters = {'n_estimators':[20,30,40,50,60,70],'learning_rate':[0.3,0.8,1.0,1.3]}
#temp_clf = AdaBoostClassifier()
#parameters = {'n_estimators':[5,6,7,8,9,10,11,12,13],'criterion':('gini','entropy'),'min_samples_split':[2,3,4,5,6,7,8]}

#temp_clf = RandomForestClassifier()
#parameters = {'criterion':('gini','entropy'),'splitter':('best', 'random'), 'min_samples_split':[2,4,6,8,10,12,14,16,18,20,24,26,30]}
#temp_clf = tree.DecisionTreeClassifier()
#clf = GridSearchCV(temp_clf, parameters)
#clf.fit(features_train,labels_train)
#clf = clf.best_estimator_
#print clf

#clf = GaussianNB()
#clf.fit(features_train,labels_train)

#pred = clf.predict(features_test)
#precision,recall = compute_precision_recall(pred,labels_test)


###############################################################################
### Task 5: Tune your classifier to achieve better than .3 precision and recall
###############################################################################

### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.
dump_classifier_and_data(clf, my_dataset, features_list)