import pandas as pd
import copy
import math
from random import randint


#Setting up data input for the pandas dataframe
small_dataset = 'CS170_SMALLtestdata__21.txt'
large_dataset = 'CS170_largetestdata__21.txt'
special_dataset = 'CS170_small_special_testdata__95.txt'
main_data = pd.read_table(special_dataset, delim_whitespace=True, header=None)


def leave_one_out_cross_validation(data,set_of_features,feature):
    #Deepcopy the data into a new dataframe so that we can make changes without affecting the orignal
    test_dataset = copy.deepcopy(data)
    for j in range(1,len(test_dataset.columns)):
        if j not in set_of_features or j != feature:
            test_dataset =test_dataset.drop([(j)])
    

    #number of correctly classfied entires
    number_correctly_classfied = 0

    #Going through all the entries and using their features to see what object they classify as
    for i in range(1,len(test_dataset)-1):
        object_to_classify = test_dataset.loc[1:len(test_dataset.columns)-1]
        label_object_to_classify = test_dataset.iloc[i,1]

        nearest_neighbor_distance = 0
        nearest_neighbor_location = 0
        nearest_neighbor_label = 0

        for k in range(1,len(test_dataset.columns-1)):
            if k != i:
                distance = math.sqrt(sum(object_to_classify - pow((test_dataset.loc[1:len(test_dataset.columns) - 1]),2)))
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data.iloc[nearest_neighbor_location,1]
        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classfied = number_correctly_classfied + 1


    #returning the accuracy of using said new feature
    return (number_correctly_classfied / (len(test_dataset)-1))


def search_feature_function(data):
    #set of current features, intialized to zero at the start
    current_set_of_features = set()
    for i in range(1,len(data)-1):
        print('On the ', str(i), 'th level of the search tree')
        best_accuracy_so_far = 0
        feature_to_add_at_this_level = 0
        for k in range(1,len(data.columns)-1):
            if k not in current_set_of_features:
                print('--Considering adding the ', str(k), ' feature')
                #accuracy = leave_one_out_cross_validation(data,current_set_of_features,k)
                accuracy = randint(1,10)
                if accuracy > best_accuracy_so_far:
                    print("Change in accuracy\n")
                    best_accuracy_so_far = accuracy
                    feature_to_add_at_this_level = k
        current_set_of_features.add(feature_to_add_at_this_level)
        print('On this level ', str(i), ', i added feature ',str(feature_to_add_at_this_level))
        #print(current_set_of_features)
    return current_set_of_features

ret = search_feature_function(main_data)
print(ret)