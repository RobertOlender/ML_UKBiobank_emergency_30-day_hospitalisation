#Import libraries.
library(readr)
library(dplyr)
library(tidyverse)
library(forcats)
library(tidymodels)
library(haven)
library(smotefamily)
library(ROSE)
library(randomForest)
library(caret)
library(e1071)
library(rpart)
library(pROC)

set.seed(7895)

#Load in data.
df <- read_csv("[Path_to_data_file].csv")

####DATA PREP####

#re-order to preferred order.
#reorder columns to desired order
desired_columns <- c("ParticipantID", "sex", "age_at_recruitment", "ethnic_background", "townsend_deprivation_index", "dbi", "dizziness_3month", "urinary_frequency_or_bladder_irritability_3month", "pain_or_discomfort_3month", "mobility_issues", "hospitalised_1_year_prior_to_index_date", "hospitalised_30_days")
other_columns <- setdiff(names(df), desired_columns)
df <- df[, c(desired_columns, sort(other_columns))]
remove(desired_columns, other_columns)

#convert all relevant variables to factors.
df$sex <- as.factor(df$sex)
df$age_at_recruitment <- as.factor(df$age_at_recruitment)
df$ethnic_background <- as.factor(df$ethnic_background)

columns_to_update <- 6:40
df[, columns_to_update] <- lapply(df[, columns_to_update], as.factor)
remove(columns_to_update)

#re-factor age_at_recruitment.
df <- df %>%
  mutate(age_at_recruitment = fct_recode(age_at_recruitment,
                                         "71-73" = "73",
                                         "71-73" = "72",
                                         "71-73" = "71",
                                         "68-70" = "70",
                                         "68-70" = "69",
                                         "68-70" = "68",
                                         "65-67" = "67",
                                         "65-67" = "66",
                                         "65-67" = "65"))

#re-factor the townsend deprivation index.
generate_labels <- function(quartiles) {
  labels <- c()
  for (i in 2:5) {
    labels <- c(labels, paste0(round(quartiles[i-1], 2), "-", round(quartiles[i], 2)))
  }
  return(labels)
}

min_val <- min(df$townsend_deprivation_index, na.rm = TRUE)
max_val <- max(df$townsend_deprivation_index, na.rm = TRUE)
quartiles <- seq(min_val, max_val, length.out = 5)
labels <- generate_labels(quartiles)
df$townsend_deprivation_index <- cut(df$townsend_deprivation_index, breaks = quartiles, labels = labels, include.lowest = TRUE)
remove(labels, max_val, min_val, quartiles, generate_labels)

#remove participants for whom hospitalised_1_year_prior_to_index_date = "1"
df <- df %>%
  filter(!(hospitalised_1_year_prior_to_index_date == "1"))

#remove unnecessary columns
df <- select(df, -hospitalised_1_year_prior_to_index_date)
df <- select(df, -ParticipantID)
df <- select(df, -fragility)
df <- select(df, -ethnic_background)
df <- select(df, -townsend_deprivation_index)

#re-level the factors to set the reference level.
df$sex <- relevel(df$sex, ref = "Male")
df$age_at_recruitment <- relevel(df$age_at_recruitment, ref = "65-67")
df$dbi <- relevel(df$dbi, ref = "0")
df$dizziness_3month <- relevel(df$dizziness_3month, ref = "0")
df$urinary_frequency_or_bladder_irritability_3month <- relevel(df$urinary_frequency_or_bladder_irritability_3month, ref = "0")
df$pain_or_discomfort_3month <- relevel(df$pain_or_discomfort_3month, ref = "0")
df$mobility_issues <- relevel(df$mobility_issues, ref = "0")
df$hospitalised_30_days <- relevel(df$hospitalised_30_days, ref = "0")
df$acute_kidney_injury <- relevel(df$acute_kidney_injury, ref = "0")
df$alzheimers_dementia <- relevel(df$alzheimers_dementia, ref = "0")
df$asthma <- relevel(df$asthma, ref = "0")
df$atrial_fibrillation_or_flutter <- relevel(df$atrial_fibrillation_or_flutter, ref = "0")
df$cancer_haematological <- relevel(df$cancer_haematological, ref = "0")
df$cancer_lung <- relevel(df$cancer_lung, ref = "0")
df$cancer_other <- relevel(df$cancer_other, ref = "0")
df$cardiac_disease <- relevel(df$cardiac_disease, ref = "0")
df$chronic_liver_disease <- relevel(df$chronic_liver_disease, ref = "0")
df$copd <- relevel(df$copd, ref = "0")
df$crohns_disease <- relevel(df$crohns_disease, ref = "0")
df$depression <- relevel(df$depression, ref = "0")
df$diabetes <- relevel(df$diabetes, ref = "0")
df$falls <- relevel(df$falls, ref = "0")
df$fractures <- relevel(df$fractures, ref = "0")
df$hazardous_alcohol_drinking <- relevel(df$hazardous_alcohol_drinking, ref = "0")
df$hearing_loss <- relevel(df$hearing_loss, ref = "0")
df$heart_failure <- relevel(df$heart_failure, ref = "0")
df$hypertension <- relevel(df$hypertension, ref = "0")
df$inflammatory_bowel_disease <- relevel(df$inflammatory_bowel_disease, ref = "0")
df$multiple_sclerosis <- relevel(df$multiple_sclerosis, ref = "0")
df$myocardial_infarction <- relevel(df$myocardial_infarction, ref = "0")
df$psychosis_schizophrenia_bipolar <- relevel(df$psychosis_schizophrenia_bipolar, ref = "0")
df$rheumatoid_arthritis <- relevel(df$rheumatoid_arthritis, ref = "0")
df$smoking <- relevel(df$smoking, ref = "0")
df$stroke <- relevel(df$stroke, ref = "0")
df$ulcerative_colitis <- relevel(df$ulcerative_colitis, ref = "0")

####SMOTE####

#SMOTE (minority class="hospitalised_30_days = Y", majority class="hospitalised_30_days = N").
df_balanced <- ovun.sample(hospitalised_30_days~., data=df, method = "over", seed=7895)$data
table(df_balanced$hospitalised_30_days)

#Partition the data into a train set and a test set in a 70:30 ratio.
ind_sample <- sample(2, nrow(df_balanced), replace=TRUE, prob = c(0.7,0.3))
train <- df_balanced[ind_sample==1, ]
test <- df_balanced[ind_sample==2, ]

####RF MODEL####

control <- trainControl(method = "repeatedcv",
                        number = 50,
                        repeats = 2,
                        allowParallel = TRUE)
metric = "Accuracy"
tuneGrid = expand.grid(.mtry = (6)) 

#create a Random Forest model using the train dataset.
rf_model <- train(hospitalised_30_days~., 
                  data = train, 
                  method = "rf", 
                  metric = metric, 
                  tuneGrid = tuneGrid, 
                  trControl = control)

#Predict 30-day hospitalisation using the test dataset.
pred_rf <- predict(rf_model, test) 

#Confusion matrix for the test dataset.
confusionMatrix(data = pred_rf, reference = test$hospitalised_30_days,positive='1')

#Generate a variable importance plot.
varImp(rf_model)
plot(varImp(rf_model), top=25)


#Generate an AUC-ROC.
prediction_for_ROC <- predict(rf_model, test, type = "prob")
ROC_rf <- roc(response = test$hospitalised_30_days, predictor = prediction_for_ROC[,2])
ROC_rf
ROC_rf_AUC <- auc(ROC_rf)
ROC_rf_AUC_95ci <- ci.auc(ROC_rf)
print(ROC_rf_AUC_95ci)

plot(ROC_rf, 
     main = "ROC Curve for the Random Fo+-rest model: 30-day-hosp", 
     col = "cadetblue4",
     lty = 1,
     lwd = 2,
     asp = NA,
     axes = TRUE,
     print.auc = TRUE)