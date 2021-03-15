import pandas as pd

#Setting up data input for the pandas dataframe
small_dataset = 'CS170_SMALLtestdata__21.txt'
large_dataset = 'CS170_largetestdata__21.txt'
main_data = pd.read_table(small_dataset, delim_whitespace=True, header=None)


#def leave_one_out_cross_validation(data,set_of_features,feature):



def search_feature_function(data):
    #set of current features, intialized to zero at the start
    current_set_of_features = {}
    for i in range(1,len(data)-1):
        print('On the ', str(i), 'th level of the search tree')
        best_accuracy_so_far = 0
        feature_to_add_at_this_level = 0
        for k in range(1,len(data.columns)-1):
            if k not in current_set_of_features:
                print('--Considering adding the ', str(k), ' feature')
                accuracy = leave_one_out_cross_validation(data,current_set_of_features,k+1)

                if accuracy > best_accuracy_so_far:
                    best_accuracy_so_far = accuracy
                    feature_to_add_at_this_level = k
        current_set_of_features.add(k)
        print('On this level ', str(i), ', i added feature ',str(feature_to_add_at_this_level))
