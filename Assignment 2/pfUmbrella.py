'''

 ARTIFICIAL INTELLIGENCE II - ASSIGNMENT 2
 ------------------------------------------

 Author: Tanveer Singh Virdi
 Date: 12 October, 2019

 pfUmbrella.py

'''

#Importing the necessary libraries
import numpy as np
import sys
import random

transition_dict = {True: [0.7,0.3],
                   False: [0.3,0.7]}

evidence_dict = {True: [0.9,0.2],
                 False: [0.1,0.8]}

num_samples = int(sys.argv[1])
num_steps = int(sys.argv[2])
evidence_vector = sys.argv[3:13]

evidence = [int(evidence_vector[i]) for i in range(len(evidence_vector))]

def norm(array) :
    add = array[0] + array[1]
    return array/add

#filtering function
def filtering(t, transition_dict, evidence_dict, evidence) :
    if t == 0 :
        return np.array([0.5,0.5])
    else :
        previous_probability = filtering(t-1, transition_dict, evidence_dict, evidence)
        prediction_step = (np.array(transition_dict[True]))*previous_probability[0] + (1 - np.array(transition_dict[True])) * previous_probability[1]
        updated_probability = norm(np.array(evidence_dict[evidence[t-1]])*prediction_step)
        return updated_probability


filtering_prob_r10 = filtering(10, transition_dict, evidence_dict, evidence)
print("filtering prob R10:", filtering_prob_r10)
print()

print("Number of particles :", num_samples)
print("Number of evidences :", num_steps)
print("Evidence vector :", evidence)
print()

def sampling(probability):
    sample_value = np.random.choice([True, False], p=probability)
    return sample_value

def particle_filtering() :
    num_particles = num_samples
    particle_dict = {True: 0,
                     False: 0}
    particle_state = np.ones((num_particles), dtype=bool)

    for i in range(num_particles):
        if (i % 2 == 0):
            particle_state[i] = True
        else:
            particle_state[i] = False

    for i in range(num_steps):
        weight_particles = np.ones(num_particles)
        for j in range(num_particles):
            particle_state[j] = sampling(
                [transition_dict[True][1 - int(particle_state[j])], transition_dict[False][1 - int(particle_state[j])]])
            weight_particles[j] *= evidence_dict[bool(evidence[i])][1 - int(particle_state[j])]

        particle_state = random.choices(particle_state, weights=weight_particles, k=num_particles)

    for i in range(num_particles):
        if (particle_state[i] == True):
            particle_dict[True] += 1
        elif (particle_state[i] == False):
            particle_dict[False] += 1


    pf_prob_r10 = norm(np.array([particle_dict[True], particle_dict[False]]))
    return pf_prob_r10

prob_true_array = []
prob_false_array = []
array = []
num_of_trials = 100
#Particle filtering is run repeatedly for 100 trials and the mean and variance is computed over all the trial outputs
for i in range(num_of_trials) :
    output = particle_filtering()
    print("Trial number = ", i + 1, " , the pf estimate of P( r10 | u 1:10 ) is :", output)
    array.append(output)
    prob_true_array.append(array[i][0])
    prob_false_array.append(array[i][1])

variance = [np.var(prob_true_array), np.var(prob_false_array)]
print("Variance of P( r10 | u 1:10) after ", num_of_trials," repeated runs of particle filtering :", variance)
mean = [np.mean(prob_true_array), np.mean(prob_false_array)]
print("Mean of P( r10 | u 1:10) after",num_of_trials," repeated runs of the particle filtering :", mean)
print("Particle filtering estimate of P( r10 | u 1:10) is:",mean)


