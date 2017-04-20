#!/usr/bin/python

""" 
    starter code for exploring the Enron dataset (emails + finances) 
    loads up the dataset (pickled dict of dicts)

    the dataset has the form
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person
    you should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
count1 = 0
count2 = 0
for key in enron_data:
    if enron_data[key]['poi']:
        count2 += 1
        if enron_data[key]['total_payments'] == 'NaN':
            count1 += 1
    
print count1 
print count2
print (count1 + 0.0)/count2