'''

 ARTIFICIAL INTELLIGENCE II - ASSIGNMENT 2
 ------------------------------------------

 Author: Tanveer Singh Virdi
 Date: 09 November, 2019

 dtree4.py

 Decision Tree

'''

import pandas as pd
import math

filename = "dataset.csv"
data = pd.read_csv(filename, header = None)
attributes = ["Alt", "Bar", "Fri", "Hun", "Pat", "Price", "Rain", "Res", "Type", "Est", "WillWait"]
data.columns = attributes
data_frame_trimmed = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
data = data_frame_trimmed

def total_entropy(input_data) :
    total_target_values = len(input_data)
    unique_target_values = input_data.iloc[:,-1].unique()
    entropy = 0
    for value in unique_target_values :
        num_value = len(input_data[input_data.iloc[:,-1] == value])
        prob_value = num_value / total_target_values
        entropy += math.log2(prob_value) * prob_value
    entropy = -1 * entropy
    return entropy

def attribute_entropy(input_data, attribute) :
    total_values = len(input_data)
    unique_attribute_values = input_data[attribute].unique()
    attribute_entropy_value = 0
    for attribute_value in unique_attribute_values :
        num_attribute_value = len(input_data[input_data[attribute] == attribute_value])
        attribute_value_sub_input = input_data[input_data[attribute] == attribute_value]
        prob_attribute_value = num_attribute_value / total_values
        attribute_entropy_value += prob_attribute_value * total_entropy(attribute_value_sub_input)
    return attribute_entropy_value

def splitting_attribute(input_data) :
    attribute_list = attributes[:-1]
    input_data_total_entropy = total_entropy(input_data)
    entropy_dict = {}
    for attribute in attribute_list :
        attribute_entropy_value = attribute_entropy(input_data, attribute)
        attribute_gain = input_data_total_entropy - attribute_entropy_value
        entropy_dict[attribute] = attribute_gain
    array = sorted(entropy_dict.items(), key=lambda x: x[1])
    return array[-1][0]

def attribute_subtree(input_data, attribute) :
    attribute_subtree_dict = {}
    unique_attribute_values = input_data[attribute].unique()
    for attribute_value in unique_attribute_values :
        attribute_subtree_dict[attribute_value] = input_data[input_data[attribute] == attribute_value]
    return attribute_subtree_dict

def decision_tree_learning(input_data, attributes) :
    split_attribute = splitting_attribute(input_data)
    subtree_attr = attribute_subtree(input_data, split_attribute)
    decision_tree = {}
    decision_tree[split_attribute] = {}
    for attribute_value in input_data[split_attribute].unique() :
        if(attribute_entropy(subtree_attr[attribute_value],split_attribute) == 0) :
            decision_tree[split_attribute][attribute_value] = subtree_attr[attribute_value]['WillWait'].iloc[0]
        else :
            decision_tree[split_attribute][attribute_value] = decision_tree_learning(subtree_attr[attribute_value], attributes)
    return decision_tree

def classify_example(example, decision_sub_tree) :
    while True :
        attributes = list(decision_sub_tree.keys())
        num_attributes = len(attributes)
        if num_attributes == 1 :
            attribute = attributes[0]
            decision_sub_tree = decision_sub_tree[attribute]
            example_attribute_value = example[attribute]
            decision_sub_tree = decision_sub_tree[example_attribute_value]
            if decision_sub_tree == 'T' or decision_sub_tree == 'F' :
                return decision_sub_tree
            else:
                classify_example(example, decision_sub_tree)
    return decision_sub_tree

def loocv_error(input_data) :
    loocv_error = 0
    total_rows = len(input_data)
    for i in range(total_rows) :
        leftout_row = input_data.iloc[i]
        desired_output_leftout_row = input_data.iloc[i,-1]
        training_set = input_data.drop(input_data.index[i])
        learned_decision_tree = decision_tree_learning(training_set, attributes)
        predicted_output_leftout_row = classify_example(leftout_row, learned_decision_tree)
        if predicted_output_leftout_row != desired_output_leftout_row :
            loocv_error += 1
    loocv_error_rate = loocv_error / total_rows
    return loocv_error_rate

def error_rate(input_data, decision_tree) :
    total_rows = len(input_data)
    error = 0
    for i in range(len(input_data)) :
        desired_output = input_data.iloc[i,-1]
        predicted_output = classify_example(input_data.iloc[i,:-1], decision_tree)
        if(predicted_output != desired_output) :
            error +=1
    error_rate = error / total_rows
    return error_rate

learned_decision_tree = decision_tree_learning(data, attributes)
print("Decision tree learned")
print("======================")
print("Training error rate for the learned tree : ", error_rate(data, learned_decision_tree))
print("LOOCV error rate for the given dataset : ", loocv_error(data))


