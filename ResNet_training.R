# Dependencies
install.packages("wrapr")
install.packages("RWeka")
install.packages("httr")
install.packages("caret")
install.packages("dplyr")
install.packages("magrittr")
install.packages("data.table")
install.packages("devtools")
install.packages("reticulate")
install_github("rstudio/keras")
install.packages("kable")
install.packages("kableExtra")

# Enable GPU computing and install Keras
install_keras(tensorflow = "2.10-gpu")

# Keras installation check
reticulate::py_config()
reticulate::py_module_available("keras")

require("kable")
require("kableExtra")

# Working directory and source import
setwd("D:/PW/Warsztaty\ z\ Technik\ Uczenia\ Maszynowego/FastTrack4")
source("./ResNet_models.R")

# Models training examples

# Uncut data training
to[model <-
     model, history <-
     history, evaluation <-
     evaluation, confusion_matrix <-
     confusion_matrix] <-
  train_model_uncut(
    dataset = "CricketX",
    epochs = 300,
    batch_size = 32,
    format = "arff"
  )

# Cut and fetched data training
to[model <-
     model, history <-
     history, evaluation <-
     evaluation, confusion_matrix <-
     confusion_matrix] <-
  train_model_cut(
    TRUE,
    dataset = "Car",
    epochs = 300,
    batch_size = 32,
    format = "txt"
  )

# Cut and loaded from hard-drive data training
to[model <-
     model, history <-
     history, evaluation <-
     evaluation, confusion_matrix <-
     confusion_matrix] <-
  train_model_cut(
    FALSE,
    epochs = 300,
    batch_size = 32,
    train_path = "./cut_data/data_cut_TRAIN.txt",
    test_path = "./cut_data/data_cut_TEST.txt"
  )

# Tests and metrics
# Trace
cut1_acc <- rep(0, 3)
for (i in 1:3) {
  to[model1_cut <-
       model, history1_cut <-
       history, evaluation1_cut <-
       evaluation, confusion1_cut <-
       confusion_matrix] <-
    train_model_cut(
      TRUE,
      dataset = "Trace",
      epochs = 300,
      batch_size = 32,
      format = "arff"
    )
  cut1_acc[i] <- evaluation1_cut[2]
}
cut1_sd <- sd(cut1_acc)
cut1_results <- mean(cut1_acc)

uncut1_acc <- rep(0, 3)
for (i in 1:3) {
  to[model1_uncut <-
       model, history1_uncut <-
       history, evaluation1_uncut <-
       evaluation, confusion1_uncut <-
       confusion_matrix] <-
    train_model_uncut(
      dataset = "Trace",
      epochs = 300,
      batch_size = 32,
      format = "arff"
    )
  uncut1_acc[i] <- evaluation1_uncut[2]
}
uncut1_sd <- sd(uncut1_acc)
uncut1_results <- mean(uncut1_acc)

# Car
cut2_acc <- rep(0, 3)
for (i in 1:3) {
  to[model2_cut <-
       model, history2_cut <-
       history, evaluation2_cut <-
       evaluation, confusion2_cut <-
       confusion_matrix] <-
    train_model_cut(
      TRUE,
      dataset = "Car",
      epochs = 600,
      batch_size = 32,
      format = "arff"
    )
  cut2_acc[i] <- evaluation2_cut[2]
}
cut2_sd <- sd(cut2_acc)
cut2_results <- mean(cut2_acc)

uncut2_acc <- rep(0, 3)
for (i in 1:3) {
  to[model2_uncut <-
       model, history2_uncut <-
       history, evaluation2_uncut <-
       evaluation, confusion2_uncut <-
       confusion_matrix] <-
    train_model_uncut(
      dataset = "Car",
      epochs = 600,
      batch_size = 32,
      format = "arff"
    )
  uncut2_acc[i] <- evaluation2_uncut[2]
}
uncut2_sd <- sd(uncut2_acc)
uncut2_results <- mean(uncut2_acc)

# Rock (only one class in test set)
cut3_acc <- rep(0, 3)
for (i in 1:3) {
  to[model3_cut <-
       model, history3_cut <-
       history, evaluation3_cut <-
       evaluation, confusion3_cut <-
       confusion_matrix] <-
    train_model_cut(
      TRUE,
      dataset = "Rock",
      epochs = 300,
      batch_size = 32,
      format = "txt"
    )
  cut3_acc[i] <- evaluation3_cut[2]
}
cut3_sd <- sd(cut3_acc)
cut3_results <- mean(uncut2_acc)

uncut3_acc <- rep(0, 3)
for (i in 1:3) {
  to[model3_uncut <-
       model, history3_uncut <-
       history, evaluation3_uncut <-
       evaluation, confusion3_uncut <-
       confusion_matrix] <-
    train_model_uncut(
      dataset = "Rock",
      epochs = 300,
      batch_size = 32,
      format = "txt"
    )
  uncut3_acc[i] <- evaluation3_uncut[2]
}
uncut3_sd <- sd(history3_uncut$metrics$accuracy)
uncut3_results <- mean(uncut3_acc)

# Colposcopy
cut4_acc <- rep(0, 3)
for (i in 1:3) {
  to[model4_cut <-
       model, history4_cut <-
       history, evaluation4_cut <-
       evaluation, confusion4_cut <-
       confusion_matrix] <-
    train_model_cut(
      TRUE,
      dataset = "Colposcopy",
      epochs = 300,
      batch_size = 32,
      format = "arff"
    )
  cut4_acc[i] <- evaluation4_cut[2]
}
cut4_sd <- sd(cut4_acc)
cut4_results <- mean(cut4_acc)

uncut4_acc <- rep(0, 3)
for (i in 1:3) {
  to[model4_uncut <-
       model, history4_uncut <-
       history, evaluation4_uncut <-
       evaluation, confusion4_uncut <-
       confusion_matrix] <-
    train_model_uncut(
      dataset = "Colposcopy",
      epochs = 300,
      batch_size = 32,
      format = "arff"
    )
  uncut4_acc[i] <- evaluation4_uncut[2]
}
uncut4_sd <- sd(uncut4_acc)
uncut4_results <- mean(uncut4_acc)

# Adiac
cut5_acc <- rep(0, 3)
for (i in 1:3) {
  to[model5_cut <-
       model, history5_cut <-
       history, evaluation5_cut <-
       evaluation, confusion5_cut <-
       confusion_matrix] <-
    train_model_cut(
      TRUE,
      dataset = "Adiac",
      epochs = 300,
      batch_size = 32,
      format = "arff"
    )
  cut5_acc[i] <- evaluation5_cut[2]
}
cut5_sd <- sd(cut5_acc)
cut5_results <- mean(cut5_acc)

uncut5_acc <- rep(0, 3)
for (i in 1:3) {
  to[model5_uncut <-
       model, history5_uncut <-
       history, evaluation5_uncut <-
       evaluation, confusion5_uncut <-
       confusion_matrix] <-
    train_model_uncut(
      dataset = "Adiac",
      epochs = 300,
      batch_size = 32,
      format = "arff"
    )
  uncut5_acc[i] <- evaluation5_uncut[2]
}
uncut5_sd <- sd(uncut5_acc)
uncut5_results <- mean(uncut5_acc)

# CricketX
cut6_acc <- rep(0, 3)
for (i in 1:3) {
  to[model6_cut <-
       model, history6_cut <-
       history, evaluation6_cut <-
       evaluation, confusion6_cut <-
       confusion_matrix] <-
    train_model_cut(
      TRUE,
      dataset = "CricketX",
      epochs = 300,
      batch_size = 32,
      format = "arff"
    )
  cut6_acc[i] <- evaluation6_cut[2]
}
cut6_sd <- sd(cut6_acc)
cut6_results <- mean(cut6_acc)

uncut6_acc <- rep(0, 3)
for (i in 1:3) {
  to[model6_uncut <-
       model, history6_uncut <-
       history, evaluation6_uncut <-
       evaluation, confusion6_uncut <-
       confusion_matrix] <-
    train_model_uncut(
      dataset = "CricketX",
      epochs = 300,
      batch_size = 32,
      format = "arff"
    )
  uncut6_acc[i] <- evaluation6_uncut[2]
}
uncut6_sd <- sd(uncut6_acc)
uncut6_results <- mean(uncut6_acc)

# Confusion matrices
rownames(confusion1_cut) <- 1:dim(confusion1_cut)[1]
colnames(confusion1_cut) <- 1:dim(confusion1_cut)[2]
table <- confusion1_cut %>%
  kable(row.names = TRUE) %>%
  kable_styling(
    bootstrap_options = c("striped", "hover", "condensed"),
    full_width = FALSE
  )
cat(table)

rownames(confusion1_uncut) <- 1:dim(confusion1_uncut)[1]
colnames(confusion1_uncut) <- 1:dim(confusion1_uncut)[2]
table <- confusion1_uncut %>%
  kable(row.names = TRUE) %>%
  kable_styling(
    bootstrap_options = c("striped", "hover", "condensed"),
    full_width = FALSE
  )
cat(table)

rownames(confusion2_cut) <- 1:dim(confusion2_cut)[1]
colnames(confusion2_cut) <- 1:dim(confusion2_cut)[2]
table <- confusion2_cut %>%
  kable(row.names = TRUE) %>%
  kable_styling(
    bootstrap_options = c("striped", "hover", "condensed"),
    full_width = FALSE
  )
cat(table)

rownames(confusion2_uncut) <- 1:dim(confusion2_uncut)[1]
colnames(confusion2_uncut) <- 1:dim(confusion2_uncut)[2]
table <- confusion2_uncut %>%
  kable(row.names = TRUE) %>%
  kable_styling(
    bootstrap_options = c("striped", "hover", "condensed"),
    full_width = FALSE
  )
cat(table)

rownames(confusion3_cut) <- 1:dim(confusion3_cut)[1]
colnames(confusion3_cut) <- 1:dim(confusion3_cut)[2]
table <- confusion3_cut %>%
  kable(row.names = TRUE) %>%
  kable_styling(
    bootstrap_options = c("striped", "hover", "condensed"),
    full_width = FALSE
  )
cat(table)

rownames(confusion3_uncut) <- 1:dim(confusion3_uncut)[1]
colnames(confusion3_uncut) <- 1:dim(confusion3_uncut)[2]
table <- confusion3_uncut %>%
  kable(row.names = TRUE) %>%
  kable_styling(
    bootstrap_options = c("striped", "hover", "condensed"),
    full_width = FALSE
  )
cat(table)

rownames(confusion4_cut) <- 1:dim(confusion4_cut)[1]
colnames(confusion4_cut) <- 1:dim(confusion4_cut)[2]
table <- confusion4_cut %>%
  kable(row.names = TRUE) %>%
  kable_styling(
    bootstrap_options = c("striped", "hover", "condensed"),
    full_width = FALSE
  )
cat(table)

rownames(confusion4_uncut) <- 1:dim(confusion4_uncut)[1]
colnames(confusion4_uncut) <- 1:dim(confusion4_uncut)[2]
table <- confusion4_uncut %>%
  kable(row.names = TRUE) %>%
  kable_styling(
    bootstrap_options = c("striped", "hover", "condensed"),
    full_width = FALSE
  )
cat(table)

rownames(confusion5_cut) <- 1:dim(confusion5_cut)[1]
colnames(confusion5_cut) <- 1:dim(confusion5_cut)[2]
table <- confusion5_cut %>%
  kable(row.names = TRUE) %>%
  kable_styling(
    bootstrap_options = c("striped", "hover", "condensed"),
    full_width = FALSE
  )
cat(table)

rownames(confusion5_uncut) <- 1:dim(confusion5_uncut)[1]
colnames(confusion5_uncut) <- 1:dim(confusion5_uncut)[2]
table <- confusion5_uncut %>%
  kable(row.names = TRUE) %>%
  kable_styling(
    bootstrap_options = c("striped", "hover", "condensed"),
    full_width = FALSE
  )
cat(table)

rownames(confusion6_cut) <- 1:dim(confusion6_cut)[1]
colnames(confusion6_cut) <- 1:dim(confusion6_cut)[2]
table <- confusion6_cut %>%
  kable(row.names = TRUE) %>%
  kable_styling(
    bootstrap_options = c("striped", "hover", "condensed"),
    full_width = FALSE
  )
cat(table)

rownames(confusion6_uncut) <- 1:dim(confusion6_uncut)[1]
colnames(confusion6_uncut) <- 1:dim(confusion6_uncut)[2]
table <- confusion6_uncut %>%
  kable(row.names = TRUE) %>%
  kable_styling(
    bootstrap_options = c("striped", "hover", "condensed"),
    full_width = FALSE
  )
cat(table)

# Results table
df <- data.frame(
  cut_accuracy = c(
    cut1_results,
    cut2_results,
    cut4_results,
    cut5_results,
    cut6_results
  ),
  uncut_accuracy = c(
    uncut1_results,
    uncut2_results,
    uncut4_results,
    uncut5_results,
    uncut6_results
  ),
  cut_sd = c(cut1_sd, cut2_sd, cut4_sd, cut5_sd, cut6_sd),
  uncut_sd = c(uncut1_sd, uncut2_sd, uncut4_sd, uncut5_sd, uncut6_sd)
)

row.names(df) <-
  c("Trace", "Car", "Colposcopy", "Adiac", "CricketX")

table <- kable(
  df,
  row.names = TRUE,
  col.names = c("Cut accuracy", "Uncut accuracy", "Cut sd", "Uncut sd"),
  format = "html",
  caption = "",
  align = "c",
  colWidths = c("3cm", "3cm", "3cm"),
  booktabs = TRUE
) %>%
  kable_styling(
    bootstrap_options = c("striped", "hover", "condensed"),
    full_width = FALSE
  )

cat(table)
