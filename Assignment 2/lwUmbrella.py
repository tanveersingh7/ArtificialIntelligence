'''

 ARTIFICIAL INTELLIGENCE II - ASSIGNMENT 2
 ------------------------------------------

 Author: Tanveer Singh Virdi
 Date: 12 October, 2019

 lwUmbrella.py

'''

#Importing the necessary libraries
import numpy as np
import sys

transition_dict = {1: [0.7,0.3],
                   0: [0.3,0.7]}

evidence_dict = {1: [0.9,0.2],
                 0: [0.1,0.8]}

num_samples = int(sys.argv[1])
num_steps = int(sys.argv[2])
evidence_vector = sys.argv[3:13]

evidence = [int(evidence_vector[i]) for i in range(len(evidence_vector))]

def norm(array) :
    add = array[0] + array[1]
    return array/add

#Filtering function
def filtering(t, transition_dict, evidence_dict, evidence) :
    if t == 0 :
        return np.array([0.5,0.5])
    else :
        previous_probability = filtering(t-1, transition_dict, evidence_dict, evidence)
        prediction_step = (np.array(transition_dict[1]))*previous_probability[0] + (1 - np.array(transition_dict[1])) * previous_probability[1]
        updated_probability = norm(np.array(evidence_dict[evidence[t-1]])*prediction_step)
        return updated_probability

filtering_prob_r10 = filtering(10, transition_dict, evidence_dict, evidence)
print("filtering prob R10:", filtering_prob_r10)
print()

print("Number of samples :", num_samples)
print("Number of evidences :", num_steps)
print("Evidence vector :", evidence)
print()


weight_dict = {}
prior_probability = [0.5,0.5]

#Generating a sample based on the probability
def sampling(probability) :
    sample_value = np.random.choice([1,0], p = probability)
    return sample_value

#Likelihood weighting fuction
def likelihood_weighting() :
    weight_dict = {1: 0,
                   0: 0}
    for j in range(num_samples) :
        weight = 1
        sample_i = sampling(prior_probability)
        for i in range(num_steps):
            sample_i = sampling([transition_dict[1][1-sample_i], transition_dict[0][1-sample_i]])
            weight *= evidence_dict[evidence[i]][1 -sample_i]
        weight_dict[sample_i] += weight

    lw_prob_r10 = norm(np.array([float(weight_dict[1]), float(weight_dict[0])]))
    #print("Likelihood weighting estimate for R10 is : ", lw_prob_r10)
    return lw_prob_r10

prob_true_array = []
prob_false_array = []
array = []
num_of_trials = 100
#Likelihood weighting is run repeatedly for 100 trials and the mean and variance is computed over all the trial outputs
for i in range(num_of_trials) :
    output = likelihood_weighting()
    print("Trial number = ",i+1," , the lw estimate of P( r10 | u 1:10) is :", output)
    array.append(output)
    prob_true_array.append(array[i][0])
    prob_false_array.append(array[i][1])

variance = [np.var(prob_true_array), np.var(prob_false_array)]
print("Variance after ",num_of_trials," repeated runs of likelihood weighting :", variance)
mean = [np.mean(prob_true_array), np.mean(prob_false_array)]
print("Mean after ",num_of_trials," repeated runs of the likelihood weighting :", mean)
print("Likelihood weighting estimate of P(r10 | u 1:10 ) is:",mean)









