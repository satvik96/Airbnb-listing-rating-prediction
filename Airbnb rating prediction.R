
# Importing --------------------------------------------------------------

library(tidyverse)
library(dplyr)
library(lubridate)
library(ROCR)
library(caret)
library(stringr)
library(randomForest)
library(tm)
library(text2vec)
library(SnowballC)
library(glmnet)
library(vip)
library(ranger)
library(xgboost)

setwd("/Users/lakshitagarg/Downloads")
set.seed(1)

train_x <- read_csv("airbnb_train_x_2023.csv")
train_y <- read_csv("airbnb_train_y_2023.csv")
test_x <- read_csv("airbnb_test_x_2023.csv")

total<-rbind(train_x,test_x) ## Combining test and training dataset for cleaning

# Cleaning and creating variables ----------------------------------------------------------------

#VJ
clean_vj <- total %>%
  select( amenities, 
         availability_30, 
         availability_365, 
         availability_60, 
         availability_90, 
         bathrooms) %>%
  mutate(bathrooms = ifelse(is.na(bathrooms), median(bathrooms, na.rm = TRUE), bathrooms),
         amenities_count=str_count(amenities, ','))

clean_vj <- clean_vj %>% select(-amenities)

total$cleaning_fee <- str_replace_all(total$cleaning_fee, fixed("$"), "")
total$cleaning_fee <- str_replace_all(total$cleaning_fee, fixed(","), "")
total$cleaning_fee <- as.numeric(total$cleaning_fee)

clean_satvik <- total %>% 
  select(bed_type,
         bedrooms,
         cancellation_policy,
         city_name, 
         cleaning_fee
  ) %>%
  mutate(
    bed_type = as.factor(bed_type),
    bedrooms = ifelse(is.na(bedrooms),0,bedrooms),
    cancellation_policy  = as.factor(cancellation_policy),
    city_name  = as.factor(city_name),
    has_cleaning_fee = as.factor(ifelse(is.na(cleaning_fee),"NO", ifelse(cleaning_fee>0,"YES","NO")))
  )

clean_satvik <- clean_satvik %>% select(-cleaning_fee)

curr_date = "2018-12-31"

clean_shweta <- total %>% 
  select(host_acceptance_rate, 
         host_identity_verified,
         host_is_superhost,
         host_response_rate,
         host_since,
         first_review,
         host_listings_count) %>%
  mutate(
    host_identity_verified = as.factor(ifelse(is.na(host_identity_verified),"FALSE",host_identity_verified)),
    host_is_superhost = as.factor(ifelse(is.na(host_is_superhost),"FALSE",host_is_superhost)),
    host_acceptance_rate = parse_number(host_acceptance_rate),
    host_acceptance_rate = ifelse(is.na(host_acceptance_rate), mean(host_acceptance_rate, na.rm=TRUE), host_acceptance_rate),
    host_response_rate = parse_number(host_response_rate),
    host_response_rate = ifelse(is.na(host_response_rate), mean(host_response_rate, na.rm=TRUE), host_response_rate),
    host_since_days = as.numeric(difftime(curr_date, host_since, units = "days")),
    host_since_days = ifelse(is.na(host_since_days), median(host_since_days,na.rm = TRUE), host_since_days),
    host_listings_count = ifelse(is.na(host_listings_count),0, host_listings_count)
  )

clean_shweta <- clean_shweta %>% select(-host_since)

total$price <- str_replace_all(total$price, fixed("$"), "")
total$price <- str_replace_all(total$price, fixed(","), "")
total$price <- as.numeric(total$price)

total$extra_people <- str_replace_all(total$extra_people, fixed("$"), "")
total$extra_people <- as.numeric(total$extra_people)

market_table <- table(total$market)
market_table <- sort(market_table, decreasing = TRUE)
top_markets <- row.names(market_table[1:19])

clean_lg <- total %>% 
  select(instant_bookable,
         price,
         is_location_exact,
         host_response_time,
         accommodates,
         market, 
         extra_people,
         neighbourhood,
         city_name) %>% 
  mutate(
    market = as.factor(ifelse(market %in% top_markets, market, "Other")),
    instant_bookable = as.factor(instant_bookable),
    property_category = as.factor(case_when(total$property_type %in% c("Apartment","Serviced apartment","Loft") ~ "apartment",
                                  total$property_type %in% c("Bed & Breakfast","Boutique hotel","Hostel") ~ "hotel",
                                  total$property_type %in% c("Townhouse","Condominium") ~ "condo",
                                  total$property_type %in% c("Bungalow","House") ~ "house",
                                  TRUE ~ "other")),
    price = ifelse(is.na(price) | price == 0,mean(price,na.rm = TRUE), price),
    price_per_person = price/accommodates,
    is_location_exact = as.factor(is_location_exact),
    host_response_time = as.factor(ifelse(is.na(host_response_time),"Other",host_response_time)),
    extra_people = as.numeric(extra_people)) %>%
   group_by(property_category) %>%
   mutate(ppp_ind = as.factor(ifelse(price_per_person > median(price_per_person),"1","0"))) %>% 
   ungroup() %>% 
   group_by(market) %>% 
   mutate(above_avg = as.factor(ifelse(price > mean(price), "1","0"))) %>% 
   ungroup() %>% 
   mutate(neighbourhood = as.factor(ifelse(is.na(neighbourhood),city_name, neighbourhood)))

clean_lg <- clean_lg %>% select(-c(city_name))

clean_jihan <- total %>%
  select(requires_license,
         room_type) %>%
  mutate(requires_license = as.factor(ifelse(is.na(requires_license), FALSE, requires_license)),
         room_type = as.factor(ifelse(is.na(room_type), "MISSING", room_type)))

total$amenities <- str_replace_all(total$amenities, fixed("{"), "")
total$amenities <- str_replace_all(total$amenities, fixed("}"), "")
total$amenities <- str_replace_all(total$amenities, fixed('"'), "")

## Text mining for Amenities --------------------------------------------------
cleaning_tokenizer <- function(v) {
  v %>%
    space_tokenizer(sep = ',') 
  
}

it_train = itoken(total$amenities, 
                  preprocessor = tolower, #preprocessing by converting to lowercase
                  tokenizer = cleaning_tokenizer, 
                  row_chunks = 1,
                  rowid = TRUE,
                  progressbar = FALSE)

#learn the vocabulary
vocab <- create_vocabulary(it_train)

#prune vocabulary 
vocab_final = prune_vocabulary(vocab, doc_proportion_min = 0.1, doc_proportion_max = 0.85)

#vectorize
vectorizer = vocab_vectorizer(vocab_final)
dtm_train = create_dtm(it_train, vectorizer)
dim(dtm_train)

# this is small enough to be represented as a regular dataframe
cluster_matrix <- data.frame(as.matrix(dtm_train))

#Integrating external dataset --------------------------------------------------------------------

#Zipcode cleaning 
total <- total %>% mutate(zipcode = str_extract(zipcode, "\\d{5}"))

DP03<- read_csv("DP03-Data.csv")
DP03 <- DP03 %>% select(zipcode, DP03_0019E, DP03_0021E, DP03_0022E, DP03_0023E, DP03_0062E)
colnames(DP03) <- c("zipcode","Drive","Transit","Walk","OtherTransp","Income")
DP03 <- DP03 %>% mutate(zipcode = ifelse(is.na(zipcode),zipcode,sprintf("%05s", zipcode)))

total <- merge(total, DP03, by = "zipcode", all.x = TRUE)

total$Income <- str_replace_all(total$Income, fixed("+"), "")
total$Income <- str_replace_all(total$Income, fixed("-"), "0")
total$Income <- str_replace_all(total$Income, fixed(","), "")

clean_external <- total %>% 
  select(zipcode,
         Drive,
         Transit,
         Walk,
         OtherTransp,
         Income,
         city_name) %>% 
  group_by(city_name) %>% 
  mutate(Drive = ifelse(is.na(Drive), mean(Drive, na.rm = TRUE), Drive),
         Transit = ifelse(is.na(Transit), mean(Transit, na.rm = TRUE), Transit),
         Walk = ifelse(is.na(Walk), mean(Walk, na.rm = TRUE), Walk),
         OtherTransp = ifelse(is.na(OtherTransp), mean(OtherTransp, na.rm = TRUE), OtherTransp),
         Income = as.numeric(Income),
         Income = ifelse(is.na(Income) | Income == 0, mean(Income, na.rm = TRUE), Income)) %>% 
  ungroup() 

clean_external <- clean_external %>% select(-city_name)

summary(clean_external)

### Text mining for description --------------------------------------------------------------------
stopwords_en <- stopwords("en")

cleaning_tokenizer_1 <- function(v) {
  v %>%
    removeNumbers %>% #remove all numbers
    removePunctuation %>% #remove all punctuation
    #stemDocument %>%
    word_tokenizer
}

it_train_1 = itoken(total$description,
                  preprocessor = tolower, #preprocessing by converting to lowercase
                  tokenizer = cleaning_tokenizer_1,
                  row_chunks = 1,
                  rowid = TRUE,
                  progressbar = FALSE)

# learn the vocabulary
vocab_1 <- create_vocabulary(it_train_1,stopwords = stopwords_en, ngram = c(1L, 2L))

# prune vocabulary
vocab_final_1 = prune_vocabulary(vocab_1,doc_proportion_max = 0.70, term_count_min = 10000)

#vectorize
vectorizer_1 = vocab_vectorizer(vocab_final_1)
dtm_train_1 = create_dtm(it_train_1, vectorizer_1)
dim(dtm_train_1)

# this is small enough to be represented as a regular dataframe
cluster_matrix_1 <- data.frame(as.matrix(dtm_train_1))

## Function for classification
classify <- function(scores, c){
  classifications <- ifelse(scores > c, 1 , 0) 
  return(classifications) 
}

###################### Feature plotting ###############################

total_x <- cbind(clean_vj, clean_satvik, clean_lg, clean_jihan, clean_shweta, cluster_matrix, clean_external) # Total clean dataset including test and training dataset

#Spliting training and testing data
train_x=total_x[1:99981,]
test_x=total_x[99982:112186,]

# splitting training set into training and validation (70-30)
n <- nrow(train_x)
train_indices <- sample(1:n, size = round(0.7*n), replace = FALSE)
remaining_indices <- setdiff(1:n, train_indices)
val_indices <- remaining_indices

# Spliting into train and valid dataset by target and features variables
train_set_x <- train_x[train_indices, ]
val_set_x <- train_x[val_indices,]
train_set_y <- train_y[train_indices, ]
val_set_y <- train_y[val_indices,]

# Combining target and feature variable for training data 
train <- cbind(train_set_x, train_set_y) %>%
  mutate(perfect_rating_score = as.factor(perfect_rating_score),
         high_booking_rate = as.factor(high_booking_rate))

train_perfect <- train %>% select(-high_booking_rate)

train_perfect_yes <- train_perfect[train_perfect$perfect_rating_score == "YES", ]
train_perfect_no <- train_perfect[train_perfect$perfect_rating_score == "NO", ]

## breakfast
barplot(table(train_perfect_yes$breakfast), main = "Breakfast offered as an ammenity in listings with perfect rating",
        xlab="Yes(1) or No(0)", ylab="Count")
barplot(table(train_perfect_no$breakfast), main = "Breakfast offered as an ammenity in other listings",
        xlab="Yes(1) or No(0)", ylab="Count")



## bathrooms

barplot(table(train_perfect_yes$bathrooms), main = "Distribution of bathrooms",
        xlab="No of bathrooms", ylab="Count")

barplot(table(train_perfect_no$bathrooms), main = "Distribution of bathrooms",
        xlab="No of bathrooms", ylab="Count")




## amenities_count 
hist(train_perfect_yes$amenities_count, main = "Distribution of amenities",
     xlab="No of amenties", ylab="Count")


v2 <- ggplot(train_perfect, mapping=aes(x=perfect_rating_score, y=amenities_count)) +geom_boxplot() + scale_color_grey() + theme_classic() + 
  stat_summary(fun = "meadian", geom = "point", shape = 2,size = 2, color = "white") +
  scale_y_continuous( limits=c(0,50)) +
  labs(title="Distribution across amenities ",
       y='Count', x='Perfect Rating Score', )

v2

## bedrooms

barplot(table(total$bedrooms), main = "Distribution of bedrooms",
        xlab="No of bedrooms", ylab="Count")



## price

#hist(total$price, main = "Price Distribution",xlab = "Price", ylab = "Frequency",xlim= c(0, 5000)) 

library(tidyverse)
library(ggplot2)


v1 <- ggplot(train_perfect, mapping=aes(x=perfect_rating_score, y=price)) +geom_boxplot() + scale_color_grey() + theme_classic() + 
  stat_summary(fun = "meadian", geom = "point", shape = 2,size = 2, color = "white") +
  scale_y_continuous( limits=c(0,300)) +
  labs(title="Per night Price distribution ",
       y='Price (USD)', x='Perfect Rating Score', )

v1


## response rate

table(train_perfect_yes$host_response_time)

barplot(table(train_perfect_yes$host_response_time), main = "Response rate for listings with a perfect rating score",
        ylab="Count")


## accomodates

barplot(table(total$accommodates), main = "Accomodation capacity",
        ylab="Count", xlab="Count of people")

## property category

table(train_perfect$property_category)

barplot(table(train_perfect$property_category), main = "Property category distribution",
        ylab="Count", xlab="Type of property")

# superhost
table(total$host_is_superhost)

barplot(table(total$host_is_superhost), main = "Is the owner a superhost?",
        ylab="Count")


# host_listings_count

barplot(table(total$host_listings_count), 
        main = "Distrubution of listings owned by hosts",
        ylab="Count", xlim = c(0,20))


# city_name
table(total$city_name)

################ Modeling ##################################################

## Linear Regression


total_x_lr <- total_x %>% 
  select(  "availability_30","bathrooms","amenities_count","bedrooms", "price","price_per_person","host_response_rate")


train_x_lr=total_x_lr[1:99981,]
test_x_lr=total_x_lr[99982:112186,]

# splitting training set into training and validation (70-30)
n <- nrow(train_x_lr)
train_indices <- sample(1:n, size = round(0.7*n), replace = FALSE)
remaining_indices <- setdiff(1:n, train_indices)
val_indices <- remaining_indices

# Spliting into train and valid dataset by target and features variables
train_set_x_lr <- train_x_lr[train_indices, ]
val_set_x_lr <- train_x_lr[val_indices,]
train_set_y_lr <- train_y[train_indices, ]
val_set_y_lr <- train_y[val_indices,]

# Combining target and feature variable for training data 
train_lr <- cbind(train_set_x_lr, train_set_y_lr) %>%
  mutate(perfect_rating_score = as.factor(perfect_rating_score),
         high_booking_rate = as.factor(high_booking_rate))

train_perfect_lr <- train_lr %>% select(-high_booking_rate)


train_perfect_lr<-train_perfect_lr %>% mutate(perfect_rating_score=ifelse(perfect_rating_score=="YES",1,0))


model_lr <- lm(perfect_rating_score ~ . , data = train_perfect_lr)


# Make predictions
predictions <- predict(model_lr, newdata = val_set_x)

valid_actuals <- as.factor(ifelse(val_set_y$perfect_rating_score == "YES",1,0))

cutoffs <- seq(0, 1, 0.01)
results_tr <- data.frame(cutoff = cutoffs)
results_va <- data.frame(cutoff = cutoffs)

# Calculate TPR and FPR for each cutoff in Training dataset
for (i in 1:length(cutoffs)) {
  classifications1 <- as.factor(classify(predictions, cutoffs[i]))
  
  valid_actuals <- as.factor(ifelse(val_set_y$perfect_rating_score == "YES",1,0))
  valid_classifications1 <- as.factor(classifications1)
  
  CM_r = confusionMatrix(data = valid_classifications1, #predictions
                         reference = valid_actuals, #actuals
                         positive="1")
  TP_r <- CM_r$table[2,2]
  TN_r <- CM_r$table[1,1]
  FP_r <- CM_r$table[2,1]
  FN_r <- CM_r$table[1,2]
  
  results_va[i,"accuracy"] <- as.numeric(CM_r$overall["Accuracy"])
  results_va[i,"TPR"] <- TP_r/(TP_r+FN_r)
  results_va[i,"FPR"] <- 1-(TN_r/(TN_r+FP_r))
}

plot(results_va$cutoff, results_va$TPR, type = "l", col = "blue", xlab = "Cutoff", ylab = "TPR / FPR")
lines(results_va$cutoff, results_va$FPR, type = "l", col = "red")
lines(results_va$cutoff, results_va$accuracy, type = "l", col = "green")
legend("topright", legend = c("TPR", "FPR","Accuracy"), col = c("blue", "red","green"), lty = 1)
abline(h=0.092)

find_cutoff_r <- results_va %>%
  filter(FPR < 0.092) #Change cutoff threshold here
cutoff = find_cutoff_r$cutoff[1]

accuracy_tr <- find_cutoff_r[1,2]
TPR_tr <- find_cutoff_r[1,3]
FPR_tr <- find_cutoff_r[1,4]

accuracy_tr  #0.6932053
TPR_tr   #0.13971
FPR_tr  #0.07337536


##### Training set validation 

predictions <- predict(model_lr, newdata = train_set_x_lr)
valid_actuals <- as.factor(ifelse(train_set_y_lr$perfect_rating_score == "YES",1,0))
predictions_bin <- ifelse(predictions > 0.35, 1, 0)

CM_1=confusionMatrix(as.factor(predictions_bin), valid_actuals)

#True Positives
TP <- CM_1$table[2,2]
#True Negatives
TN <- CM_1$table[1,1]
#False Positives
FP <- CM_1$table[2,1]
#False Negatives
FN <- CM_1$table[1,2]
accuracy1 <- as.numeric(CM_1$overall["Accuracy"])
TPR_1 <- TP/(TP+FN)
TNR_1 <- TN/(TN+FP)
FPR_1 <- 1-TNR_1


CM_1$overall #69.32
TPR_1 #0.1335
FPR_1 # 0.072


#Logistic Model --------------------------------------------------------------
library(boot)

## Simple train/test split (60:40): 
total_trees <- cbind(clean_vj, clean_satvik, clean_lg, clean_jihan, clean_shweta, clean_external) # Total clean dataset including test and training dataset
total_trees <- total_trees %>% select(-c(neighbourhood,zipcode))

#Spliting training and testing data
train_trees=total_trees[1:99981,]
test_trees=total_trees[99982:112186,]

# splitting training set into training and validation (70-30)
n <- nrow(train_trees)
train_indices <- sample(1:n, size = round(0.6*n), replace = FALSE)
remaining_indices <- setdiff(1:n, train_indices)
val_indices <- remaining_indices

# Spliting into train and valid dataset by target and features variables
train_trees_x <- train_trees[train_indices, ]
val_trees_x <- train_trees[val_indices,]
train_set_y <- train_y[train_indices, ]
val_set_y <- train_y[val_indices,]

# Combining target and feature variable for training data 
train_perfect_trees <- cbind(train_trees_x, train_set_y) %>%
  mutate(perfect_rating_score = as.factor(perfect_rating_score),
         high_booking_rate = as.factor(high_booking_rate))

train_perfect_trees <- train_perfect_trees %>% select(-high_booking_rate)

### Train logistic model
logistic_perfect <- glm(perfect_rating_score ~ ., data = train_perfect_trees, family = "binomial")

#### Top performing features 

# Get the absolute coefficient values
coefficient_magnitudes <- abs(coef(logistic_perfect))

# Sort the coefficients in descending order
sorted_coefficients <- sort(coefficient_magnitudes, decreasing = TRUE)

# Print the top-performing features
top_features <- names(sorted_coefficients)[1:30]  # Replace K with the desired number of top features
print(top_features)

### Prediction on Training Dataset
probs_perfect_train <- predict(logistic_perfect, newdata = train_trees_x, type='response')
pred_full_train <- prediction(probs_perfect_train, train_set_y$perfect_rating_score)

### Prediction on Validation Dataset
probs_perfect_val <- predict(logistic_perfect, newdata = val_trees_x, type='response')
pred_full_val <- prediction(probs_perfect_val, val_set_y$perfect_rating_score)

#Fitting curve on Validation dataset
roc_full_val <- performance(pred_full_val, "tpr", "fpr")
plot(unlist(roc_full_val@alpha.values), unlist(roc_full_val@y.values), col='red', xlab = 'Cutoff', ylab = 'Rate') # red = TPR
lines(unlist(roc_full_val@alpha.values), unlist(roc_full_val@x.values), col='blue') # blue = FPR
abline(h = 0.1)
legend(.8, 1, legend = c('TPR', 'FPR'), fill = c('red', 'blue'))

#Cutoff Selection
tpr_list <- data.frame(tpr = unlist(roc_full_val@y.values))
fpr_list <- data.frame(fpr = unlist(roc_full_val@x.values))
cutoff_list <- data.frame(cutoff = unlist(roc_full_val@alpha.values))
find_cutoff <- data.frame(cbind(tpr_list, fpr_list, cutoff_list))
find_cutoff <- find_cutoff %>%
  filter(fpr_list < 0.092) #Change cutoff threshold here
cutoff = find_cutoff$cutoff[nrow(find_cutoff)]

#Classification for training dataset

classifications1 <- as.factor(classify(probs_perfect_train, cutoff))

train_actuals <- as.factor(ifelse(train_set_y$perfect_rating_score == "YES",1,0))
train_classifications1 <- as.factor(classifications1)

CM_1 = confusionMatrix(data = train_classifications1, #predictions
                       reference = train_actuals, #actuals
                       positive="1")
#True Positives
TP <- CM_1$table[2,2]
#True Negatives
TN <- CM_1$table[1,1]
#False Positives
FP <- CM_1$table[2,1]
#False Negatives
FN <- CM_1$table[1,2]
accuracy1 <- as.numeric(CM_1$overall["Accuracy"])
TPR_1 <- TP/(TP+FN)
TNR_1 <- TN/(TN+FP)
FPR_1 <- 1-TNR_1

CM_1$overall ### 0.74
TPR_1 ### 0.355
FPR_1 ### 0.092

#Classification for validation dataset

classifications1 <- as.factor(classify(probs_perfect_val, cutoff))

valid_actuals <- as.factor(ifelse(val_set_y$perfect_rating_score == "YES",1,0))
valid_classifications1 <- as.factor(classifications1)

CM_1 = confusionMatrix(data = valid_classifications1, #predictions
                       reference = valid_actuals, #actuals
                       positive="1")
#True Positives
TP <- CM_1$table[2,2]
#True Negatives
TN <- CM_1$table[1,1]
#False Positives
FP <- CM_1$table[2,1]
#False Negatives
FN <- CM_1$table[1,2]
accuracy1 <- as.numeric(CM_1$overall["Accuracy"])
TPR_1 <- TP/(TP+FN)
TNR_1 <- TN/(TN+FP)
FPR_1 <- 1-TNR_1

CM_1$overall ##0.74
TPR_1 ##0.36
FPR_1 ##0.0919


# Perform cross-validation using 5-fold -----------------------------------------

n <- nrow(train_perfect_trees)
train_indices_cv <- sample(1:n, size = round(0.25*n), replace = FALSE)
train_trees_cv <- train_perfect_trees[train_indices_cv, ]

# Fold count for cross-validation
folds_cnt <- 5

# Store accuracies of each fold
fold_acc <- numeric(folds_cnt)

# Perform cross-validation and calculate accuracy for each fold
folds <- createFolds(train_trees_cv$perfect_rating_score, k = folds_cnt)

for (i in 1:folds_cnt) {
  train_indices <- unlist(folds[-i])
  test_indices <- folds[[i]]
  
  train_data <- train_trees_cv[train_indices, ]
  test_data <- train_trees_cv[test_indices, ]
  
  model <- train(
    perfect_rating_score ~ .,
    data = train_trees_cv,
    method = "glm",  # Replace with your desired model
    metric = "Accuracy"  # Specify the evaluation metric
  )
  
  predictions <- predict(model, newdata = test_data)
  accuracy <- sum(predictions == test_data$perfect_rating_score) / length(test_data$perfect_rating_score)
  
  fold_acc[i] <- accuracy
}

# Print the accuracies for each fold
cat("Accuracy for each fold:\n")
for (i in 1:folds_cnt) {
  cat("Fold", i, ":", fold_acc[i], "\n")
}

# Calculate and print the average accuracy
avg_accuracy <- mean(fold_acc)
cat("Average Accuracy:", avg_accuracy, "\n")

#Bagging --------------------------------------------------------------

## Subsetting Data: 
n <- nrow(train_perfect_trees)
train_indices_bag <- sample(1:n, size = round(0.20*n), replace = FALSE)
train_trees_bag <- train_perfect_trees[train_indices_bag, ]

# splitting training set into training and validation (70-30)
n <- nrow(train_trees_bag)
train_indices <- sample(1:n, size = round(0.6*n), replace = FALSE)
remaining_indices <- setdiff(1:n, train_indices)
val_indices <- remaining_indices

# Spliting into train and valid dataset by target and features variables
train_bag_x <- train_trees_bag[train_indices, ]
train_bag_x <- train_bag_x %>% select(-perfect_rating_score)
val_bag_x <- train_trees_bag[val_indices,]
val_bag_x <- val_bag_x %>% select(-perfect_rating_score)
train_bag_y <- train_y[train_indices, ]
val_bag_y <- train_y[val_indices,]

# Combining target and feature variable for training data 
train_perfect_bag <- cbind(train_bag_x, train_bag_y) %>%
  mutate(perfect_rating_score = as.factor(perfect_rating_score),
         high_booking_rate = as.factor(high_booking_rate))

train_perfect_bag <- train_perfect_bag %>% select(-high_booking_rate)

bagging_model <- randomForest(perfect_rating_score ~ ., data = train_perfect_bag, ntree = 500, mtry = (ncol(train_perfect_bag)-1))
bag_prob_tr <- predict(bagging_model, newdata = train_bag_x, type = "prob")
bag_prob_tr <- bag_prob_tr[,"YES"]
bag_prob_va <- predict(bagging_model, newdata = val_bag_x, type = "prob")
bag_prob_va <- bag_prob_va[,"YES"]
## Check RF doc

# Create a data frame with different cutoff values
cutoffs <- seq(0, 1, 0.001)
results_tr <- data.frame(cutoff = cutoffs)
results_va <- data.frame(cutoff = cutoffs)

# Calculate TPR and FPR for each cutoff in Training dataset
for (i in 1:length(cutoffs)) {
  classifications1 <- as.factor(classify(bag_prob_tr, cutoffs[i]))
  
  valid_actuals <- as.factor(ifelse(train_bag_y$perfect_rating_score == "YES",1,0))
  valid_classifications1 <- as.factor(classifications1)
  
  CM_r = confusionMatrix(data = valid_classifications1, #predictions
                         reference = valid_actuals, #actuals
                         positive="1")
  TP_r <- CM_r$table[2,2]
  TN_r <- CM_r$table[1,1]
  FP_r <- CM_r$table[2,1]
  FN_r <- CM_r$table[1,2]
  
  results_tr[i,"accuracy"] <- as.numeric(CM_r$overall["Accuracy"])
  results_tr[i,"TPR"] <- TP_r/(TP_r+FN_r)
  results_tr[i,"FPR"] <- 1-(TN_r/(TN_r+FP_r))
}

# Check TPR, FPR and accuracy for best cutoff with FPR<0.092 in Training Dataset
find_cutoff_tr <- results_tr %>%
  filter(FPR < 0.092) #Change cutoff threshold here

cutoff = find_cutoff_tr$cutoff[1]
accuracy_tr <- find_cutoff_tr[1,2]
TPR_tr <- find_cutoff_tr[1,3]
FPR_tr <- find_cutoff_tr[1,4]

accuracy_tr #0.936
TPR_tr #1
FPR_tr #0.0904

# Calculate TPR and FPR for each cutoff in Validation dataset
for (i in 1:length(cutoffs)) {
  classifications1 <- as.factor(classify(bag_prob_va, cutoffs[i]))
  
  valid_actuals <- as.factor(ifelse(val_bag_y$perfect_rating_score == "YES",1,0))
  valid_classifications1 <- as.factor(classifications1)
  
  CM_r = confusionMatrix(data = valid_classifications1, #predictions
                         reference = valid_actuals, #actuals
                         positive="1")
  
  #True Positives
  TP_r <- CM_r$table[2,2]
  #True Negatives
  TN_r <- CM_r$table[1,1]
  #False Positives
  FP_r <- CM_r$table[2,1]
  #False Negatives
  FN_r <- CM_r$table[1,2]
  
  results_va[i,"accuracy"] <- as.numeric(CM_r$overall["Accuracy"])
  results_va[i,"TPR"] <- TP_r/(TP_r+FN_r)
  results_va[i,"TNR"] <- TN_r/(TN_r+FP_r)
  results_va[i,"FPR"] <- 1-(TN_r/(TN_r+FP_r))
}

# Check TPR, FPR and accuracy for best cutoff with FPR<0.092 in Validation Dataset
find_cutoff_va <- results_va %>%
  filter(FPR < 0.092) #Change cutoff threshold here

cutoff_va = find_cutoff_va$cutoff[1]
accuracy_va <- find_cutoff_va[1,2]
TPR_va <- find_cutoff_va[1,3]
FPR_va <- find_cutoff_va[1,4]

accuracy_va #0.6655
TPR_va #0.095
FPR_va #0.90

## Best set of performing features 
vip(bagging_model)

# Fitting curve to analyze the generalization performance 
plot(results_va$cutoff, results_va$TPR, type = "l", col = "blue", xlab = "Cutoff", ylab = "TPR / FPR")
lines(results_va$cutoff, results_va$FPR, type = "l", col = "red")
lines(results_va$cutoff, results_va$accuracy, type = "l", col = "green")
legend("topright", legend = c("TPR", "FPR","Accuracy"), col = c("blue", "red","green"), lty = 1)
abline(h = 0.092)

##Ridge model -------------------------------------------------------------

library(tidyverse)
library(caret)
library(glmnet)

accuracy <- function(classifications, actuals){
  correct_classifications <- ifelse(classifications == actuals, 1, 0)
  acc <- sum(correct_classifications)/length(classifications)
  return(acc)
}

dummy <- dummyVars( ~ . , data=train_trees_x, fullRank = TRUE)
x <- as.matrix(predict(dummy, newdata =train_trees_x))
y <- as.matrix(ifelse(train_set_y[,1] == 'YES', 1, 0))

dummy <- dummyVars( ~ . , data=val_trees_x, fullRank = TRUE)
val_x <- as.matrix(predict(dummy, newdata =val_trees_x))
val_y <- as.matrix(ifelse(val_set_y[,1] == 'YES', 1, 0))

grid_ridge <- 10^seq(-5,-1,length=100) #determined by default plots above
accs_ridge <- rep(0, length(grid))
TPR_ridge <- rep(0, length(grid))
FPR_ridge <- rep(0, length(grid))

for(i in c(1:length(grid_ridge))){
  lam = grid_ridge[i] #current value of lambda
  
  #train a ridge model with lambda = lam
  glmout <- glmnet(x, y, family = "binomial", alpha = 0, lambda = lam)
  
  #make predictions as usual
  preds <- suppressWarnings(predict(glmout, newx = val_x, type = "response"))
  
  #classify and compute accuracy
  classifications <- ifelse(preds > cutoff_va, 1, 0)
  
  CM_random <- table(classifications, val_y)
  
  #True Positives
  TP <- CM_random[2,2]
  #True Negatives
  TN <- CM_random[1,1]
  #False Positives
  FP <- CM_random[2,1]
  #False Negatives
  FN <- CM_random[1,2]
  
  accs_ridge[i] <- (TP + TN)/ (TP+TN+FP+FN)
  TPR <- TP/(TP+FN)
  TNR <- TN/(TN+FP)
  FPR <- 1-TNR
  
  # Store the accuracy in the list
  TPR_ridge[i] <- TPR
  FPR_ridge[i] <- FPR
  print(i) #, included to check running progress
}

#plot fitting curve - easier to read if we plot logs
plot(log10(grid_ridge), TPR_ridge, xlab = 'Log(Lambda)', ylab = 'TPR') #as lambda approaches 0, TPR increases. Therefore, ordinary least squares works as good or better
title('Ridge')

# get best-performing lambda
ridge_validation_index <- which.max(TPR_ridge)
ridge_lambda <- grid_ridge[ridge_validation_index]
ridge_lambda

accs_ridge[lasso_validation_index]
TPR_ridge[lasso_validation_index]
FPR_ridge[lasso_validation_index]

## Lasso model -----------------------------------------------------------

#Lasso
grid_lasso <- 10^seq(-6,-2,length=100)
accs_lasso <- rep(0, length(grid))
TPR_lasso <- rep(0, length(grid))
FPR_lasso <- rep(0, length(grid))

for(i in c(1:length(grid_lasso))){
  lam = grid_lasso[i] #current value of lambda
  
  #train a lasso model with lambda = lam
  glmout <- glmnet(x, y, family = "binomial", alpha = 1, lambda = lam)
  
  #make predictions as usual
  preds <- suppressWarnings(predict(glmout, newx = val_x, type = "response"))
  
  #classify and compute accuracy
  classifications <- ifelse(preds > cutoff_va, 1, 0)
  
  CM_random <- table(classifications, val_y)
  
  #True Positives
  TP <- CM_random[2,2]
  #True Negatives
  TN <- CM_random[1,1]
  #False Positives
  FP <- CM_random[2,1]
  #False Negatives
  FN <- CM_random[1,2]
  
  accs_lasso[i] <- (TP + TN)/ (TP+TN+FP+FN)
  TPR <- TP/(TP+FN)
  TNR <- TN/(TN+FP)
  FPR <- 1-TNR
  
  # Store the accuracy in the list
  TPR_lasso[i] <- TPR
  FPR_lasso[i] <- FPR
  print(i)
}

#plot fitting curve - easier to read if we plot logs
plot(log10(grid_lasso), TPR_lasso, xlab = 'Log(Lambda)', ylab = 'TPR') #
title('Lasso')

# get best-performing lambda
lasso_validation_index <- which.max(TPR_lasso)
lasso_lambda <- grid_lasso[lasso_validation_index]
lasso_lambda

accs_lasso[lasso_validation_index] #0.701
TPR_lasso[lasso_validation_index] #0.301
FPR_lasso[lasso_validation_index] #0.062

##Ranger model --------------------------------------------------------------

# Modeling
total_x <- cbind(clean_vj, clean_satvik, clean_lg, clean_jihan, clean_shweta, cluster_matrix, clean_external) # Total clean dataset including test and training dataset

#Spliting training and testing data
train_x=total_x[1:99981,]
test_x=total_x[99982:112186,]

# splitting training set into training and validation (70-30)
n <- nrow(train_x)
train_indices <- sample(1:n, size = round(0.7*n), replace = FALSE)
remaining_indices <- setdiff(1:n, train_indices)
val_indices <- remaining_indices

# Spliting into train and valid dataset by target and features variables
train_set_x <- train_x[train_indices, ]
val_set_x <- train_x[val_indices,]
train_set_y <- train_y[train_indices, ]
val_set_y <- train_y[val_indices,]

# Combining target and feature variable for training data 
train <- cbind(train_set_x, train_set_y) %>%
  mutate(perfect_rating_score = as.factor(perfect_rating_score),
         high_booking_rate = as.factor(high_booking_rate))

train_perfect <- train %>% select(-high_booking_rate)


set.seed(1)

## Simple Train/validation split
rf_model_ranger <- ranger(perfect_rating_score ~ ., data = train_perfect,
                 mtry=sqrt(ncol(train_perfect) -1), num.trees=900,
                 importance="impurity",
                 probability = TRUE)

## Predictions on Training data 
rf_preds_ranger_tr <- predict(rf_model_ranger, data=train_set_x)$predictions[,2]

## predictions on Validation Data
rf_preds_ranger_va <- predict(rf_model_ranger, data=val_set_x)$predictions[,2]

# Create a data frame with different cutoff values
cutoffs <- seq(0, 1, 0.001)
results_tr <- data.frame(cutoff = cutoffs)
results_va <- data.frame(cutoff = cutoffs)

# Calculate TPR and FPR for each cutoff in Training dataset
for (i in 1:length(cutoffs)) {
  classifications1 <- as.factor(classify(rf_preds_ranger_tr, cutoffs[i]))
  
  valid_actuals <- as.factor(ifelse(train_set_y$perfect_rating_score == "YES",1,0))
  valid_classifications1 <- as.factor(classifications1)
  
  CM_r = confusionMatrix(data = valid_classifications1, #predictions
                         reference = valid_actuals, #actuals
                         positive="1")
  TP_r <- CM_r$table[2,2]
  TN_r <- CM_r$table[1,1]
  FP_r <- CM_r$table[2,1]
  FN_r <- CM_r$table[1,2]
  
  results_tr[i,"accuracy"] <- as.numeric(CM_r$overall["Accuracy"])
  results_tr[i,"TPR"] <- TP_r/(TP_r+FN_r)
  results_tr[i,"FPR"] <- 1-(TN_r/(TN_r+FP_r))
}

# Check TPR, FPR and accuracy for best cutoff with FPR<0.092 in Training Dataset
find_cutoff_tr <- results_tr %>%
  filter(FPR < 0.092) #Change cutoff threshold here

cutoff = find_cutoff_tr$cutoff[1]
accuracy_tr <- find_cutoff_tr[1,2]
TPR_tr <- find_cutoff_tr[1,3]
FPR_tr <- find_cutoff_tr[1,5]

accuracy_tr #0.93
TPR_tr #1
FPR_tr #0.0917

# Calculate TPR and FPR for each cutoff in Validation dataset
for (i in 1:length(cutoffs)) {
  classifications1 <- as.factor(classify(rf_preds_ranger_va, cutoffs[i]))
  
  valid_actuals <- as.factor(ifelse(val_set_y$perfect_rating_score == "YES",1,0))
  valid_classifications1 <- as.factor(classifications1)
  
  CM_r = confusionMatrix(data = valid_classifications1, #predictions
                         reference = valid_actuals, #actuals
                         positive="1")
  
  #True Positives
  TP_r <- CM_r$table[2,2]
  #True Negatives
  TN_r <- CM_r$table[1,1]
  #False Positives
  FP_r <- CM_r$table[2,1]
  #False Negatives
  FN_r <- CM_r$table[1,2]
  
  results_va[i,"accuracy"] <- as.numeric(CM_r$overall["Accuracy"])
  results_va[i,"TPR"] <- TP_r/(TP_r+FN_r)
  results_va[i,"TNR"] <- TN_r/(TN_r+FP_r)
  results_va[i,"FPR"] <- 1-(TN_r/(TN_r+FP_r))
}

# Check TPR, FPR and accuracy for best cutoff with FPR<0.092 in Validation Dataset
find_cutoff_va <- results_va %>%
  filter(FPR < 0.092) #Change cutoff threshold here

cutoff_va = find_cutoff_va$cutoff[1]
accuracy_va <- find_cutoff_va[1,2]
TPR_va <- find_cutoff_va[1,3]
FPR_va <- find_cutoff_va[1,5]

accuracy_va #0.7638
TPR_va #0.4212
FPR_va #0.0917

## Best set of performing features 
vip(rf_model_ranger)

# Fitting curve to analyze the generalization performance 
plot(results_va$cutoff, results_va$TPR, type = "l", col = "blue", xlab = "Cutoff", ylab = "TPR / FPR")
lines(results_va$cutoff, results_va$FPR, type = "l", col = "red")
lines(results_va$cutoff, results_va$accuracy, type = "l", col = "green")
legend("topright", legend = c("TPR", "FPR","Accuracy"), col = c("blue", "red","green"), lty = 1)
abline(h = 0.092)

## Tuning hyperparameter: ntree on Validation dataset
ntrees_range <- seq(100, 1000, by = 100)

# Create an empty list to store the results
accuracy_list <- list()
TPR_list <- list()
FPR_list <- list()

# Loop over the range of ntrees and fit a random forest model for each value

for (ntrees in ntrees_range) {
  # Fit the model
  rf_model_ranger <- ranger(perfect_rating_score ~ ., data = train_perfect,
                            mtry=sqrt(ncol(train_perfect) -1), num.trees=ntrees,
                            importance="impurity",
                            probability = TRUE)
  
  rf_preds_ranger <- predict(rf_model_ranger, data=val_set_x)$predictions[,2]
  rf_classifications <- ifelse(rf_preds_ranger>0.5, "YES", "NO")
  
  # Calculate the accuracy, TPR and FPR
  CM_random <- table(rf_classifications, val_set_y$perfect_rating_score)
  
  #True Positives
  TP <- CM_random[2,2]
  #True Negatives
  TN <- CM_random[1,1]
  #False Positives
  FP <- CM_random[2,1]
  #False Negatives
  FN <- CM_random[1,2]
  
  accuracy_rf <- (TP + TN)/ (TP+TN+FP+FN)
  TPR_rf <- TP/(TP+FN)
  TNR_rf <- TN/(TN+FP)
  FPR_rf <- 1-TNR_rf
  
  # Store the accuracy in the list
  accuracy_list[[as.character(ntrees)]] <- accuracy_rf
  TPR_list[[as.character(ntrees)]] <- TPR_rf
  FPR_list[[as.character(ntrees)]] <- FPR_rf
}

accuracy_list
TPR_list
FPR_list

######## Learning curve #########################

library(ggplot2)

#Define Training Sizes
train_sizes <- seq(0.1, 1, 0.1)

plot_learning_curve <- function(train_probs, valid_probs, train_sizes) {
  train_errors <- 1 - train_probs
  valid_errors <- 1 - valid_probs
  
  train_size_perc <- train_sizes * 100
  
  # Calculate mean training and validation errors for each train size
  train_errors_mean <- sapply(train_size_perc, function(size) mean(train_errors[1:round(size)]))
  valid_errors_mean <- sapply(train_size_perc, function(size) mean(valid_errors[1:round(size)]))
  
  # Plot learning curve
  plot(train_size_perc, train_errors_mean, type = "l", col = "blue",
       xlab = "Train Size (%)", ylab = "Error Rate", ylim = c(0.6, 1),
       main = "Learning Curve")
  lines(train_size_perc, valid_errors_mean, type = "l", col = "red")
  
  legend("topright", legend = c("Train Error", "Validation Error"),
         col = c("blue", "red"), lty = 1)
}

plot_learning_curve(rf_preds_ranger_tr, rf_preds_ranger_va, train_sizes)

#Relearn the model on whole of training set 

train_perfect_whole <- cbind(train_x,train_y) %>% mutate(perfect_rating_score = as.factor(perfect_rating_score))
train_perfect_whole <- train_perfect_whole %>% select(-high_booking_rate)

ranger_model_f <- rf_model_ranger <- ranger(perfect_rating_score ~ ., data = train_perfect_whole,
                                        mtry=sqrt(ncol(train_perfect) -1), num.trees=900,
                                        importance="impurity",
                                        probability = TRUE)


# Create Final Output: Ranger -----------------------------------------------------

ranger_preds_full <- predict(ranger_model_f, data=test_x)$predictions[,2]
classifications_perfect <- ifelse(ranger_preds_full>cutoff, "YES", "NO")
summary(as.factor(classifications_perfect))
write.table(classifications_perfect, "perfect_rating_score_group6_w8.csv", row.names = FALSE)

