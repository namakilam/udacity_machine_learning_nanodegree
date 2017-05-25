#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

enron_data
poi = 0
salary = 0
total_payments = 0
for key in enron_data:
    if enron_data.get(key).get('poi') == True:
        poi += 1
    if enron_data[key]['salary'] != 'NaN':
        salary += 1
    if enron_data[key]['email_address'] != 'NaN':
        email += 1
    if enron_data[key]['total_payments'] == 'NaN' and enron_data[key]['poi'] == True:
        total_payments += 1
print poi
print salary
print email
print total_payments
print 21.0/len(enron_data)
len(enron_data) + 10
enron_names = open('../final_project/poi_names.txt', 'r')
lines = enron_names.readlines()
len(lines[2:])

enron_data
enron_data['PRENTICE JAMES']['total_stock_value
enron_data['COLWELL WESLEY']['from_this_person_to_poi']
enron_data['SKILLING JEFFREY K']['exercised_stock_options']


enron_data['SKILLING JEFFREY K']['total_payments']
enron_data['LAY KENNETH L']['total_payments']
enron_data['FASTOW ANDREW S']['total_payments']
