import json
import collections
import pandas as pd
from NaiveBayes import NB
from SGD import SGD

##### Reading the Json file for Training set #####
print("Reading the input data...")
with open('1999a_TrainingData.json', 'r') as content_file:
    json_data = content_file.read()
py_object_train_data = json.loads(json_data, object_pairs_hook=collections.OrderedDict)['TrainingData']
train_data_x = []
train_data_y = []
for key, value in py_object_train_data.items():
    train_data_x.append(value["bodyText"])
    train_data_y.append(value["topics"])

##### Cleaning the Train data, filtering the missing values       #####
train_data_frame = pd.DataFrame(
    {'bodyText': train_data_x,
     'topics': train_data_y,
    })
train_data_frame = train_data_frame[train_data_frame.topics.apply(lambda c: c != [])]
train_data_frame = train_data_frame[train_data_frame.bodyText.apply(lambda c: c != "")]
train_data_x = train_data_frame.bodyText
train_data_y = train_data_frame.topics

##### Reading Json file for Test data       #####
with open('2000a_TrainingData.json', 'r') as content_file:
    json_data = content_file.read()
py_object_test_data = json.loads(json_data, object_pairs_hook=collections.OrderedDict)['TrainingData']
test_data_x = []
test_data_y = []
for key, value in py_object_test_data.items():
    test_data_x.append(value["bodyText"])
    test_data_y.append(value["topics"])

##### Cleaning the Test data, filtering the missing values       #####
test_data_frame = pd.DataFrame(
    {'bodyText': test_data_x,
    'topics': test_data_y,
    })
test_data_frame = test_data_frame[test_data_frame.topics.apply(lambda c: c != [])]
test_data_frame = test_data_frame[test_data_frame.bodyText.apply(lambda c: c != "")]
test_data_x = test_data_frame.bodyText
test_data_y = test_data_frame.topics

def main():
    ##### Calling the Naive Bayes Classifier #####
    #print("\nNAIVE BAYES CLASSIFIER...\n")
    #NB(train_data_x, train_data_y, test_data_x, test_data_y)
    ##### Calling the SGD Classifier #####
    print("\nSGD CLASSIFIER...\n")
    SGD(train_data_x, train_data_y, test_data_x, test_data_y)

if __name__ == "__main__":
    main()