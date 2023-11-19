
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
  read_html(.html) %>%
    html_elements('p, h1, h2, h3, h4, h5, h6') %>%
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
}

# function to apply to claims data
parse_data_1 <- function(.df){
  out <- .df %>%
    filter(str_detect(text_tmp, '<!')) %>%
    rowwise() %>%
    mutate(text_clean = parse_fn_1(text_tmp)) %>%
    unnest(text_clean) 
  return(out)
}

nlp_fn_1 <- function(parse_data_1.out){
  out <- parse_data_1.out %>% 
    unnest_tokens(output = token, 
                  input = text_clean,  # Use the parsed paragraphs
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

# load raw data
load('data/claims-raw.RData')

# preprocess (will take a minute or two)
claims_clean_1 <- parse_data_1(claims_raw)

# tokenization
claims_tokens <- claims_clean_1 %>%
  unnest_tokens(output = token, 
                input = text_clean, 
                token = 'words',
                stopwords = str_remove_all(stop_words$word, '[[:punct:]]'))

# View the resulting data frame
head(claims_tokens)
# frequency measures
claims_tfidf <- claims_tokens %>%
  count(.id, token) %>%
  bind_tf_idf(term = token,
              document = .id,
              n = n) 

claims_df <- claims_tfidf %>%
  left_join(claims_clean_1 %>%
              select(.id, bclass, mclass),
            by = c(".id" = ".id")) %>%
  pivot_wider(id_cols = c(.id, bclass, mclass), 
              names_from = token,
              values_from = tf_idf,
              values_fill = 0)

# export
save(claims_clean_1, file = 'data/claims-clean-example_1.RData')

## MODEL TRAINING (NN)
######################
library(tidyverse)
library(tidymodels)
library(keras)
library(tensorflow)

# logistic regression: ==================================================
source('https://raw.githubusercontent.com/pstat197/pstat197a/main/materials/scripts/package-installs.R')

# packages
library(tidyverse)
library(tidymodels)
library(modelr)
library(Matrix)
library(sparsesvd)
library(glmnet)

url <- 'https://raw.githubusercontent.com/pstat197/pstat197a/main/materials/activities/data/'

# load a few functions for the activity
source(paste(url, 'projection-functions.R', sep = ''))

# load cleaned data
load('data/claims-clean-example_1.RData')

# partition
set.seed(110122)
partitions_1 <- claims_df %>%
  initial_split(prop = 0.8)

# separate DTM from labels
# test_dtm <- testing(partitions_1) %>%
#   select(-.id, -bclass, -mclass)
# test_labels <- testing(partitions_1) %>%
#   select(.id, bclass, mclass)

# same, training set
train_dtm_1 <- training(partitions_1) %>%
  select(-.id, -bclass, -mclass)
train_labels_1 <- training(partitions_1) %>%
  select(.id, bclass, mclass)

####
train_dtm_1 <- train_dtm_1 %>%
  tibble(text = .) %>%
  unnest_tokens(word, text) %>%
  count(.id, word) %>%
  cast_dtm(document = .id, term = word, value = n)

proj_out <- projection_fn(.dtm = train_dtm_1, .prop = 0.7)
train_dtm_projected <- proj_out$data
####

# find projections based on training data
proj_out <- projection_fn(.dtm = train_dtm_1, .prop = 0.7)
train_dtm_projected <- proj_out$data

# how many components were used?
proj_out$n_pc

train <- train_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(train_dtm_projected)

fit <- glm(bclass~., data = train, family="binomial")


