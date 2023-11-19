
## Preliminary Task 1 ============================================

# Augment the HTML scraping strategy so that header information is captured 
# in addition to paragraph content. 
# Are binary class predictions improved using logistic principal component regression?

# modifying code from preprocessing.R to scrape header information 
require(tidyverse)
require(tidytext)
require(textstem)
require(rvest)
require(qdapRegex)
require(stopwords)
require(tokenizers)

# function to parse html and clean text
parse_fn_1 <- function(.html){
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
parse_data_1 <- function(.df){
  out <- .df %>%
    filter(str_detect(text_tmp, '<!')) %>%
    rowwise() %>%
    mutate(text_clean = list(parse_fn_1(text_tmp))) %>%
    unnest_wider(text_clean) 
  return(out)
}

nlp_fn_1 <- function(parse_data_1.out){
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

# preprocess (will take a minute or two)
claims_clean_1 <- claims_raw %>%
  parse_data_1()

# export
save(claims_clean_1, file = 'data/claims-clean-example_1.RData')

## MODEL TRAINING (NN)
######################
library(tidyverse)
library(tidymodels)
library(keras)
library(tensorflow)

# load cleaned data
load('data/claims-clean-example_1.RData')

# partition
set.seed(110122)
partitions_1 <- claims_clean_1 %>%
  initial_split(prop = 0.8)

# train_text_1 <- training(partitions_1) %>%
#   pull(paragraphs,headers)
train_labels_1 <- training(partitions_1) %>%
  pull(bclass) %>%
  as.numeric() - 1

# training split - concatenate paragraphs and headers into a list
train_text_1 <- training(partitions_1) %>%
  rowwise() %>%
  mutate(
    combined_text = str_c(paragraphs, headers, collapse = ' ')
  ) %>%
  select(combined_text) %>%
  pull()

# create a preprocessing layer
preprocess_layer <- layer_text_vectorization(
  standardize = NULL,
  split = 'whitespace',
  ngrams = NULL,
  max_tokens = NULL,
  output_mode = 'tf_idf'
)
preprocess_layer %>% adapt(train_text_1)

# define NN architecture
model_1 <- keras_model_sequential() %>%
  preprocess_layer() %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 25) %>%
  layer_dropout(0.2) %>%
  layer_dense(1) %>%
  layer_activation(activation = 'sigmoid')

summary(model_1)

# configure for training
model_1 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'binary_accuracy'
)

# train
history <- model_1 %>%
  fit(train_text_1, 
      train_labels_1,
      validation_split = 0.3,
      epochs = 5)

## CHECK TEST SET ACCURACY HERE

# save the entire model as a SavedModel
save_model_tf(model_1, "results/model_1")
