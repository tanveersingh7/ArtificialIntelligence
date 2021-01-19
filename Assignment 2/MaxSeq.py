'''

 ARTIFICIAL INTELLIGENCE II - ASSIGNMENT 2
 ------------------------------------------

 Author: Tanveer Singh Virdi
 Date: 10 October, 2019

 MaxSeq.py

'''

#Importing the necessary libraries
import numpy as np
import sys

num_evidence = int(sys.argv[1])
evidence_vector = sys.argv[2:2+num_evidence]
evidence_vector = [int(i) for i in evidence_vector]

print(evidence_vector)

transition_dict = {
                    True: {
                            True : 0.7,
                            False : 0.3
                    },
                    False: {
                            True : 0.4,
                            False : 0.6
                    }
}

evidence_dict = {
                    True: {
                            True : 0.9,
                            False : 0.3
                    },
                    False: {
                            True : 0.1,
                            False : 0.7
                    }
}


true_probability_vector = []
false_probability_vector = []

prior_probability = {
    True : 0.5,
    False : 0.5
}

predictions = []

if evidence_vector[0] == 1 :
    true_probability = prior_probability[True] * evidence_dict[bool(evidence_vector[0])][True]
    true_probability_vector.append(true_probability)
    false_probability = prior_probability[False] * evidence_dict[bool(evidence_vector[0])][False]
    false_probability_vector.append(false_probability)
else :
    true_probability = prior_probability[True] * evidence_dict[bool(evidence_vector[0])][True]
    true_probability_vector.append(true_probability)
    false_probability = prior_probability[False] * evidence_dict[bool(evidence_vector[0])][False]
    false_probability_vector.append(false_probability)


for i in range(1,len(evidence_vector)) :
    prev_true = true_probability_vector[i-1]
    prev_false = false_probability_vector[i-1]
    if evidence_vector[i] == 1 :
       true_probability1 = prev_true * evidence_dict[bool(evidence_vector[i])][True] * transition_dict[True][True]
       true_probability2 = prev_false * evidence_dict[bool(evidence_vector[i])][True] * transition_dict[False][True]
       true_probability_vector.append(max(true_probability1, true_probability2))
       false_probability1 = prev_true * evidence_dict[bool(evidence_vector[i])][False] * transition_dict[True][False]
       false_probability2 = prev_false * evidence_dict[bool(evidence_vector[i])][False] * transition_dict[False][False]
       false_probability_vector.append(max(false_probability1,false_probability2))
    else :
        true_probability1 = prev_true * evidence_dict[bool(evidence_vector[i])][True] * transition_dict[True][True]
        true_probability2 = prev_false * evidence_dict[bool(evidence_vector[i])][True] * transition_dict[False][True]
        true_probability_vector.append(max(true_probability1, true_probability2))
        false_probability1 = prev_true * evidence_dict[bool(evidence_vector[i])][False] * transition_dict[True][False]
        false_probability2 = prev_false * evidence_dict[bool(evidence_vector[i])][False] * transition_dict[False][False]
        false_probability_vector.append(max(false_probability1, false_probability2))


for i in range(len(true_probability_vector)) :
    if (true_probability_vector[i] > false_probability_vector[i]) :
        predictions.append(1)
    else :
        predictions.append(0)

print("Most likely sequence of states X1:10 given evidence e1:10 is : ",predictions)
print("Probability vector of Xt = 1 : ", true_probability_vector)
print("Probability vector of Xt = 0 : ",false_probability_vector)

