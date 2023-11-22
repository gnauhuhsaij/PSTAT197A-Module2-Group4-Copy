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
source('scripts/Jiashu/Task4.R')

# Load the models
model_multi <- load_model_tf('results/model_nn')
model_binary <- load_model_tf('results/model_nn_binary')

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
preds <- predict(model_multi, x) 
preds_b <- predict(model_binary, x) 


class_labels <- claims_raw %>% pull(bclass) %>% levels()
class_labels_m <- claims_raw %>% pull(mclass) %>% levels()

pred_classes <- factor(preds_b > 0.5, labels = class_labels)
pred_classes_m <- factor(max.col(preds), labels = class_labels_m)


# export 
pred_df <- clean_df %>%
  bind_cols(bclass.pred = pred_classes) %>%
  bind_cols(mclass.pred = pred_classes_m) %>%
  select(.id, bclass.pred, mclass.pred)

save(pred_df, file = 'results/preds-group4-Jiashu.RData')


