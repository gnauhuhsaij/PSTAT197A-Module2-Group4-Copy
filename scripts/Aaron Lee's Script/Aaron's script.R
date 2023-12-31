## Load Libraries and Data
require(tidyverse)
require(keras)
require(tensorflow)
source('preprocessing.R')

# Load data
load('claims-raw.RData')
load('claims-test.RData')

## Preprocess Data

# Apply preprocessing pipeline to training data
claims_clean <- claims_raw %>%
  parse_data()

# Apply preprocessing pipeline to test data
clean_df <- claims_test %>%
  parse_data() %>%
  select(.id, text_clean)

save(claims_clean, file = "claims-clean.RData")


## Binary Classification Model
library(tidyverse)
library(tidymodels)
library(keras)
library(tensorflow)

# Load cleaned data
load('claims-clean.RData')

# Partition data
set.seed(123)
partitions <- claims_clean %>%
  initial_split(prop = 0.8)

# Extract training text and labels
train_text <- training(partitions) %>%
  pull(text_clean)
train_labels <- training(partitions) %>%
  pull(bclass) %>%
  as.numeric() - 1

# If having library conflicts
install.packages("keras", type = "source")
library(keras)
install_keras()


# Create preprocessing layer
preprocess_layer <- layer_text_vectorization(
  standardize = NULL,
  split = 'whitespace',
  ngrams = NULL,
  max_tokens = NULL,
  output_mode = 'tf_idf'
)

preprocess_layer %>% adapt(train_text)

# Define NN architecture for binary classification
binary_model <- keras_model_sequential() %>%
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

summary(binary_model)

# Configure for training
binary_model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'binary_accuracy'
)

# Train binary model
history_binary <- binary_model %>%
  fit(train_text, 
      train_labels,
      validation_split = 0.3,
      epochs = 10)

plot(history_binary)

# Save binary model
save_model_tf(binary_model, "results/binary_model")

# Prediction on testing
test_text <- testing(partitions) %>%
  pull(text_clean)

preds_binary <- predict(binary_model, test_text) %>%
  as.numeric()

# Predictions for binary model
# preds_binary1 <- predict(binary_model, clean_df$text_clean) %>%
#    as.numeric()

# Convert to binary labels
class_labels_binary <- claims_raw %>% pull(bclass) %>% levels()
pred_classes_binary <- factor(preds_binary > 0.5, labels = class_labels_binary)
# pred_classes_binary


## Multi-Class Classification Model

# Define NN architecture for multi-class classification
multi_model <- keras_model_sequential() %>%
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

summary(multi_model)

# Configure for training
multi_model %>% compile(
  loss = 'sparse_categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

# Convert multi-class labels to numeric
train_labels_multi <- training(partitions) %>%
  pull(mclass) %>%
  as.factor() %>%
  as.numeric() - 1

# Train multi-class model
history_multi <- multi_model %>%
  fit(train_text, 
      train_labels_multi,
      validation_split = 0.3,
      epochs = 10)

plot(history_multi)

# Save multi-class model
save_model_tf(multi_model, "results/multi_model")

# Prediction on testing
# test_text <- testing(partitions) %>%
#   pull(text_clean)

preds_multi <- predict(multi_model, test_text)

# Predictions for multi-class model
# preds_multi <- predict(multi_model, clean_df$text_clean)
class_labels_multi <- claims_raw %>% pull(mclass) %>% levels()
pred_classes_multi <- factor(max.col(preds_multi), labels = class_labels_multi)
# pred_classes_multi


## Export Predictions

# Create and save binary predictions
pred_df_binary <- testing(partitions) %>%
  bind_cols(bclass.pred = pred_classes_binary) %>%
  select(.id, bclass.pred)

# pred_df_binary

save(pred_df_binary, file = 'results/binary_preds.RData')

# Create and save multi-class predictions
pred_df_multi <- testing(partitions) %>%
  bind_cols(mclass.pred = pred_classes_multi) %>%
  select(.id, mclass.pred)

# pred_df_multi

save(pred_df_multi, file = 'results/multi_preds.RData')

# merge the 2 dataframe

merged_pred <- left_join(pred_df_binary, pred_df_multi, by = ".id")

merged_pred

save(merged_pred, file = 'results/merged_pred.RData')

