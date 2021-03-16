import pandas as pd
import numpy as np
import copy
import math
from random import randint


#Setting up data input for the pandas dataframe
small_dataset = 'CS170_SMALLtestdata__21.txt'
large_dataset = 'CS170_largetestdata__21.txt'
special_dataset = 'CS170_small_special_testdata__95.txt'
main_data = pd.read_table(special_dataset, delim_whitespace=True, header=None).values


def leave_one_out_cross_validation(data,set_of_features,feature):
    #Deepcopy the data into a new dataframe so that we can make changes without affecting the orignal
    test_dataset = copy.deepcopy(data)
    row_len = test_dataset.shape[0]
    column_len = test_dataset.shape[1]
    for j in range(1,column_len):
        if j not in set_of_features and j != feature:
            test_dataset[:,j] = 0
    

    #number of correctly classfied entires
    number_correctly_classfied = 0

    #Going through all the entries and using their features to see what object they classify as
    for i in range(row_len):
        object_to_classify = test_dataset[i,1:]
        label_object_to_classify = test_dataset[i,0]


        nearest_neighbor_distance = 100
        nearest_neighbor_location = 0
        nearest_neighbor_label = 0

        for k in range(row_len):
            if k != i:
                distance = math.sqrt(sum(pow((object_to_classify - test_dataset[k,1:]),2)))
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = test_dataset[nearest_neighbor_location,0]

        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classfied = number_correctly_classfied + 1


    #returning the accuracy of using said new feature
    return (number_correctly_classfied / (row_len))


def search_feature_function(data):
    #set of current features, intialized to zero at the start
    current_set_of_features = set()

    #set of features that create the best accuracy 
    best_set_of_features = set()
    best_accuracy = 0

    for i in range(data.shape[1]):
        print('On the ', str(i), 'th level of the search tree')
        best_accuracy_so_far = 0
        feature_to_add_at_this_level = 0
        for k in range(1,data.shape[1]):
            if k not in current_set_of_features:
                print('--Considering adding the ', str(k), ' feature')
                accuracy = leave_one_out_cross_validation(data,current_set_of_features,k)
                
                if accuracy > best_accuracy_so_far:
                    best_accuracy_so_far = accuracy
                    feature_to_add_at_this_level = k
                    print("Change in accuracy, new accuracy: ",accuracy,"\n")
        current_set_of_features.add(feature_to_add_at_this_level)
        if best_accuracy_so_far > best_accuracy:
            best_set_of_features = copy.deepcopy(current_set_of_features)
            best_accuracy = best_accuracy_so_far
            print('Best accuracy overall',best_accuracy)
            print('Best set of features is: ', best_set_of_features)
        
        print('On this level ', str(i), ', i added feature ',str(feature_to_add_at_this_level))
    print('Final Accuracy: ', best_accuracy)
    print('Final set: ', best_set_of_features)
    return best_set_of_features

ret = search_feature_function(main_data)