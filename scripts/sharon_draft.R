require(tidyverse)
require(tidytext)
require(textstem)
require(rvest)
require(qdapRegex)
require(stopwords)
require(tokenizers)

set.seed(313)
setwd("/Users/seoeun/UCSB/PSTAT197/module2-f23-module2-group4")

head(claims_clean)


partitions <- claims_clean %>%
  initial_split(prop=0.8)

train_text <- training(partitions) %>%
  pull(text_clean)
train_labels_b <- training(partitions) %>%
  pull(bclass) %>%
  as.numeric()-1

preprocess_layer <- layer_text_vectorization(
  standardize = NULL,
  split = 'whitespace',
  ngrams = NULL,
  max_tokens = NULL,
  output_mode = 'tf_idf'
)
preprocess_layer %>% adapt(train_text)

## NN
model_b <- keras_model_sequential() %>%
  preprocess_layer() %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 25) %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 25, activation = 'relu') %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 25, activation = 'relu') %>%
  layer_dropout(0.2) %>%
  layer_dense(1) %>%
  layer_activation(activation = 'sigmoid')

summary(model_b)

# configure for training
model_b %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'binary_accuracy'
)

history_b <- model_b %>%
  fit(train_text, train_labels_b, validation_split = 0.3, epochs = 15)

## predict on the testing
test_text <- testing(partitions) %>%
  pull(text_clean)
test_pred_b <- predict(model_b, test_text) 

## prediction df
class_labels_b <- claims_raw %>% pull(bclass) %>% levels()
pred_classes_b <- factor(test_pred_b > 0.5, labels = class_labels_b)

pred_classes_b

pred_df <- testing(partitions) %>%
  bind_cols(bclass.pred = pred_classes_b) %>%
  select(.id, bclass.pred)


############# multiclass #############

train_labels_m <- training(partitions) %>%
  pull(mclass) %>%
  as.numeric()-1

unique(train_labels_m) # 2 0 4 3 1

model_m <- keras_model_sequential() %>%
  preprocess_layer() %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 25) %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 25, activation = 'relu') %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 25, activation = 'relu') %>%
  layer_dropout(0.2) %>%
  layer_dense(5) %>%
  layer_activation(activation = 'softmax')

model_m  %>% compile(
  loss = 'sparse_categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

history_m <- model_m %>%
  fit(train_text, train_labels_m, validation_split = 0.3, epochs = 15)

## predict on the testing
test_pred_m <- predict(model_m, test_text)  
test_pred_m

## prediction df
class_labels_m <- claims_raw %>% pull(mclass) %>% levels()
pred_classes_m <- factor(max.col(test_pred_m), labels = class_labels_m)

pred_df <- pred_df %>%
  bind_cols(mclass.pred = pred_classes_m)
pred_df

save(pred_df, file = 'results/predictions_df.RData')
