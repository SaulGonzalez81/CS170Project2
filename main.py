import pandas as pd
import numpy as np
import copy
import math
from random import randint


#Setting up data input for the pandas dataframe
small_dataset = 'CS170_SMALLtestdata__21.txt'
large_dataset = 'CS170_largetestdata__21.txt'
special_dataset = 'CS170_small_special_testdata__95.txt'
special_dataset2 = 'CS170_small_special_testdata__96.txt'
#main_data = pd.read_table(special_dataset, delim_whitespace=True, header=None).values


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

        nearest_neighbor_distance = 1000
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

    #Search through our dataset while testing the features to find the best accuracy
    for i in range(data.shape[1]):
        print('On the ', str(i), ' level of the search tree')
        best_accuracy_so_far = 0
        feature_to_add_at_this_level = 0
        for k in range(1,data.shape[1]):
            if k not in current_set_of_features:
                print('--Considering adding ', str(k), ' feature')
                accuracy = leave_one_out_cross_validation(data,current_set_of_features,k)
                
                if accuracy > best_accuracy_so_far:
                    best_accuracy_so_far = accuracy
                    feature_to_add_at_this_level = k
                    print("Change in accuracy, new accuracy: ",accuracy,"\n")
        
        #Add the best feature for this level to our set of current features
        current_set_of_features.add(feature_to_add_at_this_level)

        #Update the best_accuracy that we have seen and update the set of features that produce this result
        if best_accuracy_so_far > best_accuracy:
            best_set_of_features = copy.deepcopy(current_set_of_features)
            best_accuracy = best_accuracy_so_far
        
        #Print out the feature that was added at this level if there was a feature added. feature 0 is not allowed.
        if feature_to_add_at_this_level != 0:
            print('On this level ', str(i), ', feature ',str(feature_to_add_at_this_level), ' was added.')
        else:
            print('On this level ', str(i), ', no feature was added.')

    #Print out the set that produces the best accuracy 
    print('Final Best Accuracy: ', best_accuracy)
    print('Final Best Set: ', best_set_of_features)  


def backwards_elimination_search(data):
    #set of current features, initialized to all of them from them
    current_set_of_features = {i for i in range(1,data.shape[1])}

    #set of features that create the best accuracy 
    best_set_of_features = {i for i in range(1,data.shape[1])}
    best_accuracy = 0

    #Search through our dataset while testing the features to find the best accuracy
    for i in range(data.shape[1]):
        print('On the ', str(10-i), ' level of the search tree')
        best_accuracy_so_far = 0
        feature_to_eliminate_at_this_level = 0
        for k in range(1,data.shape[1]):
            if k in current_set_of_features:
                print('--Considering deleting ', str(k), ' feature')
                elim_set_data = copy.deepcopy(current_set_of_features)
                elim_set_data.remove(k)
                accuracy = leave_one_out_cross_validation(data,elim_set_data,0)

                if accuracy > best_accuracy_so_far:
                    best_accuracy_so_far = accuracy
                    feature_to_eliminate_at_this_level = k
                    print("Change in accuracy, new accuracy: ",accuracy,"\n")

        #Determine which feature to delete at the current level and determine if that set produces the best accuracy so far. If so, then we save the accuracy & the set
        if feature_to_eliminate_at_this_level != 0:            
            current_set_of_features.remove(feature_to_eliminate_at_this_level)
            if best_accuracy_so_far > best_accuracy:
                best_set_of_features = copy.deepcopy(current_set_of_features)
                best_accuracy = best_accuracy_so_far
            print('On this level ', str(10-i), ', feature ',str(feature_to_eliminate_at_this_level), ' was eliminated.')
        else:
            print('On this level ', str(10-i), ', no feature was eliminated.')

    #Print out the set that produces the best accuracy 
    print('Final Best Accuracy: ', best_accuracy)
    print('Final Best Set: ', best_set_of_features)    

def main_function():
    print("Hello! ¯\_(ツ)_/¯ Welcome to Bertie Woosters Feature Selection Algorithm!")
    file_input = input('Type in the name of the file to test: ')
    print('\nType the number of the algorithm you want to run.\n')
    print('\t1) Forward Selection\n\t2) Backward Elimination\n')
    search_input = input()
    option = True
    while option:
        if search_input == str(1):
            main_data = pd.read_table(file_input, delim_whitespace=True, header=None).values
            search_feature_function(main_data)
            option = False
        elif search_input == str(2):
            main_data = pd.read_table(file_input, delim_whitespace=True, header=None).values
            backwards_elimination_search(main_data)
            option = False
        else:
            print('Please enter a valid option! :)\n')
            search_input = input()
            option = True
main_function()