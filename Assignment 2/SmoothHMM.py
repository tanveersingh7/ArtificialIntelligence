'''

 ARTIFICIAL INTELLIGENCE II - ASSIGNMENT 2
 ------------------------------------------

 Author: Tanveer Singh Virdi
 Date: 08 October, 2019

 SmoothHMM.py

'''

#Importing the necessary libraries
import numpy as np
import sys

num_steps = int(sys.argv[1])
evidence_vector = sys.argv[2:12]
evidence_vector = [int(evidence_vector[i]) for i in range(len(evidence_vector))]

transition_dict = {True: [0.7,0.4],
                   False: [0.3,0.6]}

evidence_dict = {True: [0.9,0.3],
                 False: [0.1,0.7]}

def norm(array) :
    add = array[0] + array[1]
    return array/add


def filtering(t, transition_dict, evidence_dict, evidence) :
    if t == 0 :
        return np.array([0.5,0.5])
    else :
        previous_probability = filtering(t-1, transition_dict, evidence_dict, evidence)
        prediction_step = (np.array(transition_dict[True]))*previous_probability[0] + (1 - np.array(transition_dict[True])) * previous_probability[1]
        updated_probability = norm(np.array(evidence_dict[bool(evidence[t-1])])*prediction_step)

        return updated_probability


def smoothing(k, transition_dict, evidence_dict, evidence) :
    if k == len(evidence) :
        return 1
    else :
        array = smoothing(k+1, transition_dict, evidence_dict, evidence)
        array1 = np.array(evidence_dict[bool(evidence[k])][0])*array*np.array(transition_dict[True])+ np.array(evidence_dict[bool(evidence[k])][1])*array* (1-np.array(transition_dict[True]))
        return array1


output = []
for i in range(1,num_steps+1) :
    smooth_estimate_i = norm(filtering(i, transition_dict, evidence_dict, evidence_vector) * smoothing(i, transition_dict, evidence_dict,evidence_vector))
    print("Smoothing estimate for Xt = ",i," is :", smooth_estimate_i)
    output.append((smooth_estimate_i[0],smooth_estimate_i[1]))

print("-----------------------------------------------------------------------------------------------------")
print("Smoothed estimates of Xt given evidence e1-10 :", output)