# Load libraries

library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(class)

# Read dataset

df <- read.csv("Dataset.csv")

# Explore dataset

head(df)
tail(df)
dim(df)
str(df)
summary(df)
colSums(is.na(df))

# Check duplicate records

sum(duplicated(df))
df <- distinct(df)

# Check numerical columns

numeric_columns <- names(df)[sapply(df, is.numeric)]

# Initial boxplot

df %>%
  select(all_of(numeric_columns)) %>%
  pivot_longer(cols = everything(),
               names_to = "Feature",
               values_to = "Value") %>%
  ggplot(aes(x = Feature, y = Value)) +
  geom_boxplot() +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Boxplot of Numerical Features")

# Handle missing values

mode_value <- function(x) {
  values <- na.omit(x)
  if (length(values) == 0) {
    return(NA)
  }
  names(sort(table(values), decreasing = TRUE))[1]
}

for (col in names(df)) {
  if (is.numeric(df[[col]])) {
    df[[col]][is.na(df[[col]])] <- median(df[[col]], na.rm = TRUE)
  } else {
    df[[col]][is.na(df[[col]])] <- mode_value(df[[col]])
  }
}

# Check missing values after preprocessing

colSums(is.na(df))

# Exploratory Data Analysis

# Gender distribution

if ("Gender" %in% names(df)) {
  ggplot(df, aes(x = Gender, fill = Gender)) +
    geom_bar() +
    theme_minimal() +
    labs(title = "Gender Distribution")
}

# Marital status distribution

if ("Married" %in% names(df)) {
  ggplot(df, aes(x = Married, fill = Married)) +
    geom_bar() +
    theme_minimal() +
    labs(title = "Marital Status Distribution")
}

# Education distribution

if ("Education" %in% names(df)) {
  ggplot(df, aes(x = Education, fill = Education)) +
    geom_bar() +
    theme_minimal() +
    labs(title = "Education Distribution")
}

# Loan status distribution

ggplot(df, aes(x = Loan_Status, fill = Loan_Status)) +
  geom_bar() +
  theme_minimal() +
  labs(title = "Loan Approval Status Distribution",
       x = "Loan Status",
       y = "Count")

# Statistical analysis

summary(df)

df %>%
  select(where(is.numeric)) %>%
  summarise(
    across(
      everything(),
      list(
        Mean = ~mean(.),
        Median = ~median(.),
        SD = ~sd(.),
        Minimum = ~min(.),
        Maximum = ~max(.)
      )
    )
  )

# Correlation analysis

correlation_data <- df %>%
  select(where(is.numeric))

correlation_matrix <- cor(correlation_data)

print(correlation_matrix)

# Correlation heatmap

correlation_df <- as.data.frame(as.table(correlation_matrix))

ggplot(correlation_df,
       aes(Var1, Var2, fill = Freq)) +
  geom_tile() +
  scale_fill_gradient2(
    low = "blue",
    mid = "white",
    high = "red",
    midpoint = 0
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Correlation Heatmap",
       x = "",
       y = "")

# Feature Engineering

if (all(c("ApplicantIncome", "CoapplicantIncome") %in% names(df))) {
  df$Total_Income <- df$ApplicantIncome + df$CoapplicantIncome
  df$ApplicantIncomelog <- log1p(df$ApplicantIncome)
  df$Total_Income_log <- log1p(df$Total_Income)
}

if ("LoanAmount" %in% names(df)) {
  df$LoanAmountlog <- log1p(df$LoanAmount)
}

if ("Loan_Amount_Term" %in% names(df)) {
  df$Loan_Amount_Term_log <- log1p(df$Loan_Amount_Term)
}

# Distribution of transformed features

if ("ApplicantIncomelog" %in% names(df)) {
  ggplot(df, aes(x = ApplicantIncomelog)) +
    geom_histogram(bins = 30, fill = "skyblue", color = "black") +
    theme_minimal() +
    labs(title = "Applicant Income Distribution")
}

if ("LoanAmountlog" %in% names(df)) {
  ggplot(df, aes(x = LoanAmountlog)) +
    geom_histogram(bins = 30, fill = "lightgreen", color = "black") +
    theme_minimal() +
    labs(title = "Loan Amount Distribution")
}

if ("Loan_Amount_Term_log" %in% names(df)) {
  ggplot(df, aes(x = Loan_Amount_Term_log)) +
    geom_histogram(bins = 30, fill = "orange", color = "black") +
    theme_minimal() +
    labs(title = "Loan Amount Term Distribution")
}

if ("Total_Income_log" %in% names(df)) {
  ggplot(df, aes(x = Total_Income_log)) +
    geom_histogram(bins = 30, fill = "purple", color = "black") +
    theme_minimal() +
    labs(title = "Total Income Distribution")
}

# Boxplots of transformed features

transformed_columns <- intersect(
  c("ApplicantIncomelog",
    "LoanAmountlog",
    "Loan_Amount_Term_log",
    "Total_Income_log"),
  names(df)
)

if (length(transformed_columns) > 0) {
  df %>%
    select(all_of(transformed_columns)) %>%
    pivot_longer(cols = everything(),
                 names_to = "Feature",
                 values_to = "Value") %>%
    ggplot(aes(x = Feature, y = Value)) +
    geom_boxplot(fill = "lightblue") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = "Boxplot of Transformed Features")
}

# Loan status comparison

if ("Education" %in% names(df)) {
  ggplot(df, aes(x = Education, fill = Loan_Status)) +
    geom_bar(position = "dodge") +
    theme_minimal() +
    labs(title = "Loan Approval by Education")
}

if ("Property_Area" %in% names(df)) {
  ggplot(df, aes(x = Property_Area, fill = Loan_Status)) +
    geom_bar(position = "dodge") +
    theme_minimal() +
    labs(title = "Loan Approval by Property Area")
}

if ("Gender" %in% names(df)) {
  ggplot(df, aes(x = Gender, fill = Loan_Status)) +
    geom_bar(position = "dodge") +
    theme_minimal() +
    labs(title = "Loan Approval by Gender")
}

# Remove unnecessary original columns

columns_to_remove <- intersect(
  c("ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Total_Income",
    "Loan_ID"),
  names(df)
)

df <- df %>%
  select(-all_of(columns_to_remove))

# Encode target variable

df$Loan_Status <- trimws(as.character(df$Loan_Status))

df$Loan_Status <- factor(
  df$Loan_Status,
  levels = c("N", "Y")
)

# Encode categorical variables

categorical_columns <- names(df)[
  sapply(df, function(x) is.character(x) || is.factor(x))
]

categorical_columns <- setdiff(categorical_columns, "Loan_Status")

for (col in categorical_columns) {
  df[[col]] <- as.integer(factor(df[[col]])) - 1
}

# Check processed dataset

str(df)
head(df)
summary(df)

# Prepare features and target variable

X <- df %>%
  select(-Loan_Status)

y <- df$Loan_Status

# Remove rows with missing target values, if any

valid_rows <- !is.na(y)

X <- X[valid_rows, , drop = FALSE]
y <- droplevels(y[valid_rows])

# Convert features to numeric

X <- as.data.frame(
  lapply(X, function(x) as.numeric(as.character(x)))
)

# Replace any remaining missing or infinite feature values

for (col in names(X)) {
  X[[col]][!is.finite(X[[col]])] <- median(
    X[[col]][is.finite(X[[col]])],
    na.rm = TRUE
  )
}

# Split training and testing data

set.seed(123)

train_index <- createDataPartition(
  y,
  p = 0.75,
  list = FALSE
)

X_train <- X[train_index, , drop = FALSE]
X_test <- X[-train_index, , drop = FALSE]

y_train <- droplevels(y[train_index])
y_test <- factor(y[-train_index], levels = levels(y))

# Balance training data using upsampling

train_data <- data.frame(
  X_train,
  Loan_Status = y_train
)

train_data <- upSample(
  x = train_data %>% select(-Loan_Status),
  y = train_data$Loan_Status,
  yname = "Loan_Status"
)

X_train_balanced <- train_data %>%
  select(-Loan_Status)

y_train_balanced <- train_data$Loan_Status

# Logistic Regression

model1 <- glm(
  Loan_Status ~ .,
  data = train_data,
  family = binomial
)

logistic_probability <- predict(
  model1,
  newdata = X_test,
  type = "response"
)

logistic_prediction <- ifelse(
  logistic_probability >= 0.5,
  "Y",
  "N"
)

logistic_prediction <- factor(
  logistic_prediction,
  levels = levels(y_test)
)

# Decision Tree

model2 <- rpart(
  Loan_Status ~ .,
  data = train_data,
  method = "class"
)

decision_prediction <- predict(
  model2,
  newdata = X_test,
  type = "class"
)

decision_prediction <- factor(
  decision_prediction,
  levels = levels(y_test)
)

# Random Forest

set.seed(123)

model3 <- randomForest(
  Loan_Status ~ .,
  data = train_data,
  ntree = 500,
  importance = TRUE
)

random_forest_prediction <- predict(
  model3,
  newdata = X_test
)

random_forest_prediction <- factor(
  random_forest_prediction,
  levels = levels(y_test)
)

# KNN

# Scale data using training data parameters

train_mean <- sapply(X_train_balanced, mean)
train_sd <- sapply(X_train_balanced, sd)

train_sd[train_sd == 0] <- 1

X_train_scaled <- scale(
  X_train_balanced,
  center = train_mean,
  scale = train_sd
)

X_test_scaled <- scale(
  X_test,
  center = train_mean,
  scale = train_sd
)

# Apply KNN

knn_prediction <- knn(
  train = X_train_scaled,
  test = X_test_scaled,
  cl = y_train_balanced,
  k = 5
)

knn_prediction <- factor(
  knn_prediction,
  levels = levels(y_test)
)

# Classification Report Function

classification_report <- function(actual, predicted) {

  actual <- factor(actual, levels = c("N", "Y"))
  predicted <- factor(predicted, levels = c("N", "Y"))

  cm <- confusionMatrix(
    predicted,
    actual,
    positive = "Y"
  )

  print(cm)

  precision <- cm$byClass["Precision"]
  recall <- cm$byClass["Recall"]
  f1 <- cm$byClass["F1"]
  accuracy <- cm$overall["Accuracy"]

  return(
    data.frame(
      Accuracy = as.numeric(accuracy),
      Precision = as.numeric(precision),
      Recall = as.numeric(recall),
      F1_Score = as.numeric(f1)
    )
  )
}

# Evaluate Logistic Regression

cat("Logistic Regression Results\n")

logistic_report <- classification_report(
  y_test,
  logistic_prediction
)

print(logistic_report)

# Evaluate Decision Tree

cat("Decision Tree Results\n")

decision_report <- classification_report(
  y_test,
  decision_prediction
)

print(decision_report)

# Evaluate Random Forest

cat("Random Forest Results\n")

random_forest_report <- classification_report(
  y_test,
  random_forest_prediction
)

print(random_forest_report)

# Evaluate KNN

cat("KNN Results\n")

knn_report <- classification_report(
  y_test,
  knn_prediction
)

print(knn_report)

# Compare all models

model_comparison <- data.frame(
  Model = c(
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "KNN"
  ),
  Accuracy = c(
    logistic_report$Accuracy,
    decision_report$Accuracy,
    random_forest_report$Accuracy,
    knn_report$Accuracy
  ),
  Precision = c(
    logistic_report$Precision,
    decision_report$Precision,
    random_forest_report$Precision,
    knn_report$Precision
  ),
  Recall = c(
    logistic_report$Recall,
    decision_report$Recall,
    random_forest_report$Recall,
    knn_report$Recall
  ),
  F1_Score = c(
    logistic_report$F1_Score,
    decision_report$F1_Score,
    random_forest_report$F1_Score,
    knn_report$F1_Score
  )
)

print(model_comparison)

# Accuracy comparison graph

ggplot(model_comparison,
       aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Model Accuracy Comparison",
       x = "Machine Learning Models",
       y = "Accuracy") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

# Precision comparison graph

ggplot(model_comparison,
       aes(x = Model, y = Precision, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Model Precision Comparison",
       x = "Machine Learning Models",
       y = "Precision") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

# Recall comparison graph

ggplot(model_comparison,
       aes(x = Model, y = Recall, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Model Recall Comparison",
       x = "Machine Learning Models",
       y = "Recall") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

# F1 score comparison graph

ggplot(model_comparison,
       aes(x = Model, y = F1_Score, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Model F1 Score Comparison",
       x = "Machine Learning Models",
       y = "F1 Score") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

# Combined performance comparison

model_comparison_long <- model_comparison %>%
  pivot_longer(
    cols = c(Accuracy, Precision, Recall, F1_Score),
    names_to = "Metric",
    values_to = "Score"
  )

ggplot(model_comparison_long,
       aes(x = Model, y = Score, fill = Metric)) +
  geom_bar(stat = "identity",
           position = "dodge") +
  theme_minimal() +
  labs(title = "Overall Model Performance Comparison",
       x = "Machine Learning Models",
       y = "Performance Score") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

# Random Forest feature importance

importance(model3)

varImpPlot(
  model3,
  main = "Random Forest Feature Importance"
)

# Decision Tree visualization

rpart.plot(
  model2,
  main = "Decision Tree for Loan Approval Prediction",
  extra = 104,
  fallen.leaves = TRUE
)

# Cross-validation for Logistic Regression

set.seed(123)

control <- trainControl(
  method = "cv",
  number = 5
)

logistic_cv <- train(
  Loan_Status ~ .,
  data = train_data,
  method = "glm",
  family = binomial,
  trControl = control
)

print(logistic_cv)

# Cross-validation for Decision Tree

set.seed(123)

decision_cv <- train(
  Loan_Status ~ .,
  data = train_data,
  method = "rpart",
  trControl = control
)

print(decision_cv)

# Save trained model and feature columns

saveRDS(
  model3,
  "loan_model.rds"
)

saveRDS(
  names(X_train),
  "feature_columns.rds"
)

# Final output

cat("Project execution completed successfully.\n")
cat("Random Forest model saved as loan_model.rds\n")
cat("Feature columns saved as feature_columns.rds\n")

print(model_comparison)
