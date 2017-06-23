from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import SGDClassifier

def SGD(train_data_x, train_data_y, test_data_x, test_data_y):

    ##### Generate vectors for words in training data Articles    #####
    print("Generating frequency vectors for the words present in the training set. This will take a while...")
    vectorizer = TfidfVectorizer(min_df=0.001, max_df=0.90, stop_words='english', smooth_idf=True,
                                 norm="l2", sublinear_tf=False, use_idf=True, ngram_range=(1, 3))
    X = vectorizer.fit_transform(train_data_x)

    ##### Generate vectors for words in training data Articles    #####
    xTest = vectorizer.transform(test_data_x)

    ##### Level binarizer for Class variable of Train set     #####
    print("Converting the class variable to binary matrix...")
    mlb_train = MultiLabelBinarizer()
    train_data_y = mlb_train.fit_transform(train_data_y)

    ##### Level binarizer for Class variable of Test set     #####
    mlb_test = MultiLabelBinarizer()
    test_data_y = mlb_test.fit_transform(test_data_y)

    ##### Fitting a One Vs Rest Classifier using the Naive Bayes classifier       #####
    print("Training the model...")
    classifier = OneVsRestClassifier(SGDClassifier(alpha=0.00001)).fit(X, train_data_y)
    classifier
    ##### Getting the Predictions for labels for test data        #####
    print("Getting the predictions for labels on Test Data...")
    y_pred = classifier.predict(xTest)

    ##### Getting predicted labels and Actual labels present in test data for all test samples    #####
    dict_test_y = {}
    dict_test_pred = {}
    for i in range(len(test_data_y)):
        list_pred_y = []
        for j in range(len(list(mlb_train.classes_))):
            if (y_pred[i][j] == 1):
                list_pred_y.append(j)
        dict_test_pred[i] = list_pred_y

    for i in range(len(test_data_y)):
        list_test_y = []
        for j in range(len(list(mlb_test.classes_))):
            if (test_data_y[i][j] == 1):
                list_test_y.append(j)
        dict_test_y[i] = list_test_y

    ##### Calculating the accuracy measures       #####
    print("Checking for Precision and F-Measure...")
    correct_pred = 0
    total_labels = 0
    total_pred = 0
    for i in dict_test_y.keys():
        predicted = []
        tested = []
        for k in dict_test_pred[i]:
            predicted.append(list(mlb_train.classes_)[k])
        for k in dict_test_y[i]:
            tested.append(list(mlb_test.classes_)[k])
        total_labels = total_labels + len(tested)
        total_pred = total_pred + len(predicted)
        correct_pred = correct_pred + len(set(predicted).intersection(tested))

    print("\nTotal labels predicted: " + str(total_pred))
    print("Total labels in Test Data: " + str(total_labels))
    print("Correct predicted labels: " + str(correct_pred))

    precision = float(correct_pred * 100 / total_pred)
    recall = float(correct_pred * 100 / total_labels)
    f_measure = float(((2 * correct_pred * 100) / (total_labels + total_pred)))

    print("\nPrecision: %.2f" %precision)
    print("Recall: %.2f" %recall)
    print("F-Measure: %.2f" %f_measure)