'''

 ARTIFICIAL INTELLIGENCE II - ASSIGNMENT 2
 ------------------------------------------

 Author: Tanveer Singh Virdi
 Date: 11 November, 2019

 dlist2.py

 Decision List

'''

import pandas as pd

filename = "dataset.csv"
data = pd.read_csv(filename, header = None)
attributes = ["Alt", "Bar", "Fri", "Hun", "Pat", "Price", "Rain", "Res", "Type", "Est", "WillWait"]
data.columns = attributes
data_frame_trimmed = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
data = data_frame_trimmed

def attribute_subtree(input_data, attribute):
    attribute_subtree = {}
    unique_attribute_values = input_data[attribute].unique()
    for value in unique_attribute_values:
        attribute_subtree[value] = input_data[input_data[attribute] == value]
    return attribute_subtree


def pure_test(input_data):
    unique_target_values = input_data.iloc[:,-1].unique()
    if(len(unique_target_values) == 1) :
        return True
    return False

def testing_attribute_and_examples_t(input_data) :
    for attribute in attributes[:-1] :
        attribute_sub_tree = attribute_subtree(input_data, attribute)
        attribute_unique_values = list(attribute_sub_tree.keys())
        for value in attribute_unique_values :
            if(pure_test(attribute_sub_tree[value])) :
                examples_t = attribute_sub_tree[value]
                return attribute , value , examples_t
    raise Exception

def decision_list_learner(input_data) :
    if len(input_data) == 0:
        outcome_dict = {
            'outcome' : 'F'
        }
        return [(outcome_dict)]
    else :
        try :
            attribute,value,examples_t = testing_attribute_and_examples_t(input_data)
        except Exception :
            print("Failure. No set of examples found that have the same output for some test")
            return
        if(examples_t["WillWait"].iloc[0] == 'T') :
            outcome = 'T'
        else :
            outcome = 'F'

        input_data = input_data.drop(examples_t.index)
        attribute_dict = {'attribute' : attribute}
        value_dict = {'attribute_value' : value}
        outcome_dict = {'outcome' : outcome}
        return [(attribute_dict, value_dict, outcome_dict)] + decision_list_learner(input_data)


def classify_example(example, decision_list):
    for test in decision_list:
        if example[test[0]['attribute']] == test[1]['attribute_value']:
            return test[2]['outcome']

def error_rate(input_data, decision_list) :
    training_error = 0
    for i in range(len(input_data)):
        desired_output = input_data['WillWait'].iloc[i]
        predicted_output = classify_example(input_data.iloc[i,:-1], decision_list)
        if predicted_output != desired_output:
            training_error += 1
    return training_error / len(input_data)

def loocv_error(input_data) :
    total_rows = len(input_data)
    loocv_error = 0
    for i in range(total_rows):
        leftout_row = input_data.iloc[i,:-1]
        desired_output_leftout_row = input_data.iloc[i,-1]
        training_set = input_data.drop([i])
        learned_decision_list = decision_list_learner(training_set)
        predicted_output_leftout_row = classify_example(leftout_row, learned_decision_list)
        if predicted_output_leftout_row != desired_output_leftout_row:
            loocv_error += 1
    loocv_error_rate = loocv_error / total_rows
    return loocv_error_rate

dlist = decision_list_learner(data)
print("Decision list learned")
print("======================")
print("Training error rate for the learned decision list : ", error_rate(data, dlist))
print("LOOCV error rate for the given dataset : ", loocv_error(data))
