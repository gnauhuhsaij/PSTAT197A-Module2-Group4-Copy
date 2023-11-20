require(tidyverse)
require(tidytext)
require(textstem)
require(rvest)
require(qdapRegex)
require(stopwords)
require(tokenizers)

set.seed(313)
load("claims-clean-example.RData")

head(claims_clean)


partitions <- claims_clean %>%
  initial_split(prop=0.8)

train_text <- training(partitions) %>%
  pull(text_clean)
train_labels <- training(partitions) %>%
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
model <- keras_model_sequential() %>%
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

summary(model)

# configure for training
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'binary_accuracy'
)

history <- model %>%
  fit(train_text, train_labels, validation_split = 0.3, epochs = 15)

## predict on the testing
test_text <- testing(partitions) %>%
  pull(text_clean)
test_pred <- predict(model, test_text)

## prediction df
pred_df_b <- testing(partitions) %>%
  select(.id) %>%
  mutate(bclass.pred = (test_pred>0.5))

############# multiclass #############

train_labels_m <- training(partitions) %>%
  pull(mclass) %>%
  as.numeric()-1


model_m <- keras_model_sequential() %>%
  preprocess_layer() %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 25) %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 25, activation = 'relu') %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 25, activation = 'relu') %>%
  layer_dropout(0.2) %>%
  layer_dense(1) %>%
  layer_activation(activation = 'softmax')

model_m  %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

history_m <- model %>%
  fit(train_text, train_labels_m, validation_split = 0.3, epochs = 15)

## predict on the testing
test_pred_m <- predict(model_m, test_text)  ## keep getting 1 for every row

## prediction df
pred_df_m <- testing(partitions) %>%
  select(.id) %>%
  mutate(mclass.pred = (test_pred_m>0.5))

## combine into one data frame
#pred_df <- merge(pred_df_b, pred_df_m, by=".id")
#pred_df
