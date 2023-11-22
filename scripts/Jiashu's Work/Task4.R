
## Primary Task ##
require(tidyverse)
require(tidytext)
require(textstem)
require(rvest)
require(qdapRegex)
require(stopwords)
require(tokenizers)
require(pROC)
require("caret")
require("e1071")

# function to parse html and clean text
parse_fn <- function(.html){
  doc <- read_html(.html)
  
  paragraphs <- doc %>%
    html_elements('p') %>%
    html_text2() %>%
    str_c(collapse = ' ') %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all('\'') %>%
    str_replace_all(paste(c('\n', 
                            '[[:punct:]]', 
                            'nbsp', 
                            '[[:digit:]]', 
                            '[[:symbol:]]'),
                          collapse = '|'), ' ') %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>%
    str_replace_all("\\s+", " ")
  
  headers <- doc %>%
    html_elements(c('h1', 'h2', 'h3', 'h4', 'h5', 'h6')) %>%
    html_text2() %>%
    str_c(collapse = ' ') %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all('\'') %>%
    str_replace_all(paste(c('\n', 
                            '[[:punct:]]', 
                            'nbsp', 
                            '[[:digit:]]', 
                            '[[:symbol:]]'),
                          collapse = '|'), ' ') %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>%
    str_replace_all("\\s+", " ")
  
  return(list(paragraphs = paragraphs, headers = headers))
}

# function to apply to claims data
parse_data <- function(.df){
  out <- .df %>%
    filter(str_detect(text_tmp, '<!')) %>%
    rowwise() %>%
    mutate(text_clean = list(parse_fn(text_tmp))) %>%
    unnest_wider(text_clean) 
  return(out)
}

nlp_fn <- function(parse_data.out){
  out <- parse_data_1.out %>% 
    unnest_tokens(output = token, 
                  input = paragraphs,  # Use the parsed paragraphs
                  token = 'words',
                  stopwords = str_remove_all(stop_words$word, 
                                             '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, bclass, token.lem, name = 'n') %>%
    bind_tf_idf(term = token.lem, 
                document = .id,
                n = n) %>%
    pivot_wider(id_cols = c('.id', 'bclass'),
                names_from = 'token.lem',
                values_from = 'tf_idf',
                values_fill = 0)
  return(out)
}

## nlp-model-development:=========================================
## PREPROCESSING

# load raw data
load('data/claims-raw.RData')
load('data/claims-test.RData')

# preprocess (will take a minute or two)
claims_clean <- claims_raw %>%
  parse_data()

# export
save(claims_clean, file = 'data/Jiashu-clean.RData')

## MODEL TRAINING (NN)
######################
library(tidyverse)
library(tidymodels)
library(keras)
library(tensorflow)

# load cleaned data
load('data/Jiashu-clean.RData')

# partition
set.seed(101)
partitions <- claims_clean %>%
  initial_split(prop = 0.9)

############################# DEEP LEARNING & RNN ##############################
##### DATA PREPARATION
# Use the script provided by yoobin to concat the header with the text
train_labels <- training(partitions) %>%
  pull(bclass) %>%
  as.numeric() - 1

train_labels_m <- training(partitions) %>%
  pull(mclass) %>%
  as.numeric() - 1

train_text <- training(partitions) %>%
  rowwise() %>%
  mutate(
    combined_text = str_c(paragraphs, headers, collapse = ' ')
  ) %>%
  select(combined_text) %>%
  pull()

test_labels <- testing(partitions) %>%
  pull(bclass) %>%
  as.numeric() - 1

test_labels_m <- testing(partitions) %>%
  pull(mclass) %>%
  as.numeric() - 1

test_text <- testing(partitions) %>%
  rowwise() %>%
  mutate(
    combined_text = str_c(paragraphs, headers, collapse = ' ')
  ) %>%
  select(combined_text) %>%
  pull()


##### BASIC DEEP LEARNING MODEL
# create a preprocessing layer
preprocess_layer <- layer_text_vectorization(
  standardize = "lower_and_strip_punctuation",
  split = 'whitespace',
  ngrams = NULL,
  max_tokens = NULL,
  output_mode = 'tf_idf'
  
)
preprocess_layer %>% adapt(train_text)

# define NN architecture
model_nn <- keras_model_sequential() %>%
  preprocess_layer() %>%  
  layer_dropout(0.7) %>%
  layer_dense(units = 2048, activation = "relu") %>%
  layer_dropout(0.7) %>%
  layer_dense(units = 2048, activation = "relu") %>%
  layer_dropout(0.7) %>%
  layer_dense(5) %>%
  layer_activation(activation = 'softmax')

# define binary NN architecture
model_nn_b <- keras_model_sequential() %>%
  preprocess_layer() %>%  
  layer_dropout(0.8) %>%
  layer_dense(units = 1024, activation = "relu") %>%
  layer_dropout(0.8) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(1) %>%
  layer_activation(activation = 'sigmoid')

summary(model_nn)

# configure for training
model_nn %>% compile(
  loss = 'sparse_categorical_crossentropy',
  optimizer = "adam",
  metrics = 'accuracy'
)

model_nn_b %>% compile(
  loss = 'binary_crossentropy',
  optimizer = "adam",
  metrics = 'binary_accuracy'
)

# train
history <- model_nn %>%
  fit(train_text, 
      train_labels_m,
      validation_split = 0.2,
      epochs = 30)
history <- model_nn_b %>%
  fit(train_text, 
      train_labels,
      validation_split = 0.2,
      epochs = 10)

preds <- predict(model_nn, test_text) 
class_labels <- factor(test_labels_m) %>% levels()
pred_classes <- factor(max.col(preds), labels = class_labels)
confusionMatrix(pred_classes, factor(test_labels_m))
roc_object <- multiclass.roc(test_labels_m, max.col(preds))
auc(roc_object)

preds <- predict(model_nn_b, test_text) 
class_labels <- factor(test_labels) %>% levels()
pred_classes <- factor(preds>0.5, labels = class_labels)
confusionMatrix(pred_classes, factor(test_labels))
roc_object <- roc(test_labels, preds)
auc(roc_object)

# save the entire model as a SavedModel
save_model_tf(model_nn, "results/model_nn")
save_model_tf(model_nn_b, "results/model_nn_binary")





##### RNN MODEL
# define NN architecture
preprocess_layer <- layer_text_vectorization(
  standardize = "lower_and_strip_punctuation",
  split = 'whitespace',
  ngrams = NULL,
  max_tokens = NULL,
  output_mode = 'int'
  
)
preprocess_layer %>% adapt(train_text)

model_rnn <- keras_model_sequential() %>%
  preprocess_layer %>%
  layer_embedding(input_dim = 40618+1, output_dim = 1024) %>%
  layer_dropout(0.5) %>%
  layer_simple_rnn(units = 32) %>%
  layer_dense(8, activation = "relu") %>%
  layer_dense(units = 1, activation = 'sigmoid')

summary(model_rnn)

# configure for training
model_rnn %>% compile(
  loss = 'binary_crossentropy',
  optimizer = "adam",
  metrics = 'binary_accuracy'
)

# train
history <- model_rnn %>%
  fit(train_text, 
      train_labels,
      validation_split = 0.3,
      epochs = 5)

preds <- predict(model_1, test_text) %>%
  as.numeric()
class_labels <- claims_raw %>% pull(bclass) %>% levels()
pred_classes <- factor(preds > 0.5, labels = class_labels)

pred_df <- clean_df %>%
  bind_cols(bclass.pred = pred_classes) %>%
  select(.id, bclass.pred)
# save the entire model as a SavedModel
save_model_tf(model_rnn, "results/model_rnn")




##### SVM MODEL

preprocess_layer <- layer_text_vectorization(
  standardize = "lower_and_strip_punctuation",
  split = 'whitespace',
  ngrams = NULL,
  max_tokens = NULL,
  output_mode = 'tf_idf'
)
preprocess_layer %>% adapt(train_text)

train_data <- as.data.frame(as.matrix(preprocess_layer(train_text)))
colnames(train_data) <- make.names(colnames(train_data))
test_data <- as.data.frame(as.matrix(preprocess_layer(test_text)))
colnames(test_data) <- make.names(colnames(test_data))


train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 10)
svm_model <- svm(x = train_data, y = factor(train_labels), type = 'nu-classification', kernel = 'radial', probability = TRUE)
predictions <- predict(svm_model, newdata = test_data)
confusionMatrix(predictions, factor(test_labels))


##EXPORT 
# apply preprocessing pipeline
clean_df <- claims_test %>%
  parse_data()

# grab input
x <- clean_df %>%
  rowwise() %>%
  mutate(
    combined_text = str_c(paragraphs, headers, collapse = ' ')
  ) %>%
  select(combined_text) %>%
  pull()

# compute predictions
preds <- predict(model_nn, x) 

preds_b <- predict(model_nn_b, x) 


class_labels <- claims_raw %>% pull(bclass) %>% levels()
class_labels_m <- claims_raw %>% pull(mclass) %>% levels()

pred_classes <- factor(preds_b > 0.5, labels = class_labels)
pred_classes_m <- factor(max.col(preds), labels = class_labels_m)


# export (KEEP THIS FORMAT IDENTICAL)
pred_df <- clean_df %>%
  bind_cols(bclass.pred = pred_classes) %>%
  bind_cols(mclass.pred = pred_classes_m) %>%
  select(.id, bclass.pred, mclass.pred)

save(pred_df, file = 'results/preds-group4-Jiashu.RData')



## Writeup notes
"""
1. Preliminary tasks show that augmenting the html scraping strategy would help. So in this task I 
concatenate headers to the web content and use them both as predictors.

2. For binary classification and multinomial classification, I experimented with a support vector machine
model, a standard deep learning model, and a RNN model.

3. MODEL SELECTION

- For the standard deep learning model, I experimented with various combinations of hyper-parameters, including
batch size, number and size of layers, activation functions, regularization methods, and optimization method.
- I found that a relatively light model with only two layers with a hard regularization (dropout layers) helped
pervent the model from overfitting. Batch size, activation functions, and optimization method were not 
significantly influential. 

- For the SVM model, I found that a combination of a nu-classification machine and a radial-basis kernel yielded 
the best result.
- Also used a 10-fold cross-validation during the training process.

(May or may not include this in the final report)
- RNN took a really long time to train each epoch.
- Tried to change the model architecture multiple times, but the training loss never converged.
- The final accuracy was around 55%.



4. RESULT

***BINARY CLASSFICATION***
1. SVM
          Reference
Prediction   0   1
         0 102  47
         1   7  58
         
Accuracy : 0.7477
Sensitivity : 0.9358          
Specificity : 0.5524 
AUC-ROC: NA

2. BASIC NN
          Reference
Prediction  0  1
         0 91 17
         1 18 88
                                        
Accuracy : 0.8364          
Sensitivity : 0.8349        
Specificity : 0.8381    
AUC-ROC: 0.8638

3. RNN
N/A

***MULTICLASS CLASSIFICATION***
1. BASIC NN
Confusion Matrix and Statistics

          Reference
Prediction  0  1  2  3  4
         0 92  3  3  5  6
         1  6 11  0  0  0
         2  6  1 48  0  0
         3  5  0  0 25  0
         4  0  0  0  0  3                 
Accuracy : 0.8364        
                     Class: 0 Class: 1 Class: 2 Class: 3 Class: 4
Sensitivity            0.8440  0.73333   0.9412   0.8333  0.33333
Specificity            0.8381  0.96985   0.9571   0.9728  1.00000
AUC-ROC: 0.7644



8. The result from class

sensitivity binary         0.621
specificity binary         0.830
accuracy    binary         0.721
roc_auc     binary         0.796





