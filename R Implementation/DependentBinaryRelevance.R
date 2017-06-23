#---------------------------- Setting required parameters & loading packages

setwd(dir = "../R Implementation/")

set.seed(100)
gc()
time.start = Sys.time()
print(paste("Application Started at", time.start))

#install.packages("jsonlite")
library(jsonlite)
#install.packages("tm")
library(tm)
#install.packages("mlr")
library(mlr)
library(stringr)

print(paste("Started Processing All Files at", Sys.time()))


#---------------------------- Obtaining list of Matrices from JSONs & Cleaning Data

cleansed_data = NULL
for(filename in list.files()[1:1]){
  print(paste("Processing File: ", filename))
  #Read JSON file
  json_data = fromJSON(filename)
  
  data_table = NULL
  #Transform JSON to a the data_table matrix
  for(i in json_data$TrainingData) {
    data_table = rbind(data_table, c(i[1], i[2], i[3]))
  }
  
  row_names = names(json_data$TrainingData)
  
  data_table = cbind(row_id=row_names, data_table)
  
  #Take a subset of the data
  data_index = sample(seq_len(nrow(data_table)), size = 0.35 * nrow(data_table))
  data_table = data_table[data_index,]
  
  #Remove entries with no categories
  data_table = data_table[data_table[,3] != "list()",]
  
  #Remove Entries with no body text
  data_table = data_table[data_table[,4] != "",]
  
  #Remove Unwanted column(s)
  data_table = data_table[,c(-1,-2)]
  
  #Append cleansed data from current file to cleansed_data
  cleansed_data = rbind(cleansed_data,data_table)
}

print(paste("Complete Processing All Files at", Sys.time()))

remove(json_data)
remove(data_table)
gc()

cleansed_data = as.data.frame(cleansed_data)

print(paste("Cleansed Data generated at", Sys.time()))

#Unlist list in cleansed data frame
cleansed_data[,2] = as.character(cleansed_data[,2])

#Generating train index to split the data set
train_index = sample(seq_len(nrow(cleansed_data)), size = 0.8 * nrow(cleansed_data))


#-----------------------------Generating feature vector based on word frequencies

#Generating a Corpus of the cleansed data
text_corpus = Corpus(VectorSource(as.vector(cleansed_data[,2])))

#Creating a DocumentTermMatrix, and cleansing the data further
dtm <- DocumentTermMatrix(text_corpus, control = list(
  weight = weightTfIdf, 
  tolower = TRUE, 
  removeNumbers = TRUE,
  removePunctuation = TRUE, 
  stripWhitespace = TRUE,
  minWordLength = 2, 
  stopwords = stopwords(kind = "en")))

#Removing the Sparse terms in the text
dtm = removeSparseTerms(dtm, 0.8)

text_feature_vector = as.data.frame(as.matrix(dtm))

#Add an '_' to labels to avoid conflict with topic names
colnames(text_feature_vector) = paste(colnames(text_feature_vector),"_", sep = "")

print(paste("Feature vectors generated at", Sys.time()))


#---------------------------- Binding target variables as columns

#List all unique categories in the training data
all_categories = unique(unlist(cleansed_data[train_index,1]))

#For each category set TRUE if the row belongs to the category else FALSE.
category_vec = NULL
for (category in all_categories ){
  category_match = vapply(cleansed_data[,1], function(x) category %in% unlist(x), logical(1))
  category_vec = cbind(category_vec, category_match)
}
colnames(category_vec) = all_categories

category_vec = as.data.frame(category_vec)

print(paste("Category vectors generated at", Sys.time()))


#---------------------------- Creating datasets for train and test (80-20)

#Generating a data frame by combining the features and the targets
all_data = data.frame(text_feature_vector, category_vec)

#Splitting the data into train and test based on previosuly calculated index
train_data = all_data[train_index, ]
test_data = all_data[-train_index, ]


#---------------------------- Removing variables no longer needed

remove(all_data)
remove(data_index)
remove(category_vec)
remove(cleansed_data)
remove(text_feature_vector)
remove(category_match)
remove(dtm)
remove(text_corpus)
remove(train_index)
remove(fileList)
gc()


# ---------------------------- Creating the DBR model

#Creating the learner for the DBR model
lrn.dbr = makeMultilabelDBRWrapper(makeLearner("classif.rpart", predict.type = "response"))

#Creating the task for the DBR model
train.task = makeMultilabelTask(id = "multi", data = train_data, target = all_categories)

print(paste("Started Training Model at", Sys.time()))

#Creating the model
dbr_model = train(lrn.dbr, train.task)

print(paste("Completed Training Model at", Sys.time()))
gc()

#Predicting on the test set using the model created
predict_result = predict(dbr_model, newdata =  test_data)

#Extracting the predicted labels
label_index = length(all_categories)
predict_data = predict_result$data[,(label_index+1):ncol(predict_result$data)]
all_predicted_labels = colnames(predict_data)
all_predicted_labels = lapply(all_predicted_labels, function(x) substr(x, 10,str_length(x)))


# ---------------------------- Calculating precision of the model

tp = 0
fp = 0
fn = 0
total_pred = 0
test_row_count = nrow(test_data)
for(i in 1:length(all_predicted_labels)){
  #print(paste(i, all_predicted_labels[[i]]))
  for(j in 1:test_row_count){
    if(test_data[j,(ncol(test_data) - length(all_predicted_labels) + i)]){
      if(predict_data[j,i]){
        tp = tp + 1
      } else{
        fn = fn + 1
      }
    } else{
      if(predict_data[j,i]){
        fp = fp + 1
      }
    }
  }
}
precision = tp/(tp + fp) * 100
recall = tp/(tp + fn) * 100
f_measure = (2 * recall * precision) / (recall + precision)
cat("Precision:", precision, "%")
cat("Recall:", recall, "%")
cat("F-Measure:", f_measure, "%")

print(paste("End of Application Run", Sys.time()))
print(paste("Total Execution Time:", Sys.time() - time.start, "minutes"))
