require(keras)
require(caret)
require(dplyr)
require(magrittr)
require(data.table)

source("./ResNet_data_utilities.R")

convert_to_df <- function(list) {
  nm <- list
  nm <- lapply(nm, as.matrix)
  n <- max(sapply(nm, nrow))
  return(do.call(cbind, lapply(nm, function (x)
    rbind(
      x, matrix(, n - nrow(x), ncol(x))
    ))))
}

data_preprocessing <- function(data, isTxt) {
  trainTarget <- 1
  testTarget <- 1
  if (!isTxt) {
    trainTarget = length(data[[1]])
    testTarget = length(data[[2]])
  }
  train <- as.data.frame(do.call(cbind, data[[1]]))
  train <- train[sample(1:nrow(train)),]
  test <- as.data.frame(do.call(cbind, data[[2]]))
  test <- test[sample(1:nrow(train)),]
  train_x <- train[, -trainTarget]
  test_x <- test[, -testTarget]
  train_y <- as.numeric(train[, trainTarget]) - 1
  test_y <- as.numeric(test[, testTarget]) - 1
  
  # cut -1 from the end of the series, fill -1 with median or sth
  temp <- as.data.frame(lapply(train_x, function(x) {
    x == -1
  }))
  min_blank <- mean(rowSums(temp))
  train_x <- train_x[, 1:(length(train_x[1, ]) - min_blank)]
  test_x <- test_x[, 1:(length(test_x[1, ]) - min_blank)]
  
  for (i in 1:length(train_x[, 1])) {
    median_train <-
      median(train_x[i, ][train_x[i, ] != -1 & !is.na(train_x[i, ])])
    train_x[i, ][train_x[i, ] == -1] = median_train
    train_x[i, ][is.na(train_x[i, ])] = median_train
  }
  
  for (i in 1:length(test_x[, 1])) {
    median_test <-
      median(test_x[i, ][test_x[i, ] != -1 & !is.na(test_x[i, ])])
    test_x[i, ][test_x[i, ] == -1] = median_test
    test_x[i, ][is.na(test_x[i, ])] = median_test
  }
  
  train_y[is.na(train_y)] = train_y[1]
  test_y[is.na(test_y)] = test_y[1]
  
  train_x <- as.matrix(train_x)
  dim(train_x) <- c(dim(train_x)[1], dim(train_x)[2], 1)
  test_x <- as.matrix(test_x)
  dim(test_x) <- c(dim(test_x)[1], dim(test_x)[2], 1)
  
  train_y_categorical <- to_categorical(train_y)
  test_y_categorical <- to_categorical(test_y)
  
  # Returns list: (train x, train y, test x, test y, number of classes, input shape)
  return(list(
    train_x,
    train_y_categorical,
    test_x,
    test_y_categorical,
    length(unique(train_y)),
    c(dim(train_x)[2], 1)
  ))
}

residual_block <-
  function(input_tensor,
           filters,
           kernel_size = 3,
           dilation_rate = 1,
           dropout_rate = 0.0) {
    x <- input_tensor
    identity <- input_tensor
    
    # First convolution layer
    x <-
      x %>% layer_conv_1d(
        filters,
        kernel_size = kernel_size,
        padding = "same",
        dilation_rate = dilation_rate,
        use_bias = FALSE
      ) %>%
      layer_batch_normalization() %>%
      layer_activation(activation = "relu") %>%
      layer_dropout(rate = dropout_rate)
    
    # Second convolution layer
    x <-
      x %>% layer_conv_1d(
        filters,
        kernel_size = kernel_size,
        padding = "same",
        dilation_rate = dilation_rate,
        use_bias = FALSE
      ) %>%
      layer_batch_normalization() %>%
      layer_dropout(x, rate = dropout_rate)
    
    # Shortcut connection
    if (dim(input_tensor)[2] != dim(x)[2]) {
      identity <-
        identity <-
        layer_conv_1d(
          identity,
          filters,
          kernel_size = 1,
          padding = "same",
          use_bias = FALSE
        ) %>%
        layer_batch_normalization()
    }
    
    x <- x %>%  layer_add(identity) %>%
      layer_activation(activation = "relu")
    
    return(x)
  }

resnet <-
  function(input_shape,
           num_classes,
           num_filters = 64,
           num_blocks = 3,
           kernel_size = 3,
           dilation_rate = 1,
           dropout_rate = 0.0) {
    input <- layer_input(shape = input_shape)
    
    # Initial convolution layer
    x <-
      input %>% layer_conv_1d(
        num_filters,
        kernel_size = 8,
        padding = "same",
        use_bias = FALSE
      ) %>%
      layer_batch_normalization() %>%
      layer_activation(activation = "relu")
    
    # Residual blocks
    for (i in 1:length(num_blocks)) {
      for (j in 1:num_blocks[i]) {
        if (j == 1 && i != 1) {
          # x <- x %>% residual_block(x, num_filters * 2^(i - 1), kernel_size = kernel_size * 2^(i - 1), dilation_rate = dilation_rate * 2^(i - 1), dropout_rate = dropout_rate)
          x <- x %>% residual_block(64)
          
        } else {
          # x <- x %>% residual_block(x, num_filters * 2^(i - 1), kernel_size = kernel_size * 2^(i - 1), dilation_rate = dilation_rate * 2^(i - 1), dropout_rate = dropout_rate)
          x <- x %>% residual_block(num_filters)
        }
      }
    }
    
    # Final layers
    x <- x %>% layer_global_average_pooling_1d() %>%
      layer_dense(units = num_classes, activation = "softmax")
    
    return(keras_model(input, x))
  }

resnet_hfwaz <-
  function(input_shape,
           num_classes,
           num_filters = 64,
           num_blocks = 3,
           kernel_size = 3,
           dilation_rate = 1,
           dropout_rate = 0.0) {
    input <- layer_input(shape = input_shape)
    
    block_1 <-
      input %>% layer_conv_1d(
        num_filters,
        kernel_size = 8,
        padding = "same",
        use_bias = FALSE
      ) %>%
      layer_batch_normalization() %>%
      layer_activation(activation = "relu")
    
    
    block_1 <-
      block_1 %>% layer_conv_1d(
        num_filters,
        kernel_size = 5,
        padding = "same",
        use_bias = FALSE
      ) %>%
      layer_batch_normalization() %>%
      layer_activation(activation = "relu") %>%
      layer_conv_1d(
        num_filters,
        kernel_size = 3,
        padding = "same",
        use_bias = FALSE
      ) %>%
      layer_batch_normalization()
    
    shortcut_1 <-
      input %>% layer_conv_1d(
        num_filters,
        kernel_size = 1,
        padding = "same",
        use_bias = FALSE
      ) %>%
      layer_batch_normalization()
    
    block_1 <- layer_add(list(shortcut_1, block_1)) %>%
      layer_activation(activation = "relu")
    
    block_2 <-
      block_1 %>% layer_conv_1d(
        num_filters * 2,
        kernel_size = 8,
        padding = "same",
        use_bias = FALSE
      ) %>%
      layer_batch_normalization() %>%
      layer_activation(activation = "relu") %>%
      layer_conv_1d(
        num_filters * 2,
        kernel_size = 5,
        padding = "same",
        use_bias = FALSE
      ) %>%
      layer_batch_normalization() %>%
      layer_activation(activation = "relu") %>%
      layer_conv_1d(
        num_filters * 2,
        kernel_size = 3,
        padding = "same",
        use_bias = FALSE
      ) %>%
      layer_batch_normalization()
    
    shortcut_2 <-
      block_1 %>% layer_conv_1d(
        num_filters * 2,
        kernel_size = 1,
        padding = "same",
        use_bias = FALSE
      ) %>%
      layer_batch_normalization()
    
    block_2 <- layer_add(list(shortcut_2, block_2)) %>%
      layer_activation(activation = "relu")
    
    block_3 <-
      block_2 %>% layer_conv_1d(
        num_filters * 2,
        kernel_size = 8,
        padding = "same",
        use_bias = FALSE
      ) %>%
      layer_batch_normalization() %>%
      layer_activation(activation = "relu") %>%
      layer_conv_1d(
        num_filters * 2,
        kernel_size = 5,
        padding = "same",
        use_bias = FALSE
      ) %>%
      layer_batch_normalization() %>%
      layer_activation(activation = "relu") %>%
      layer_conv_1d(
        num_filters * 2,
        kernel_size = 3,
        padding = "same",
        use_bias = FALSE
      ) %>%
      layer_batch_normalization()
    
    shortcut_3 <- block_2  %>% layer_batch_normalization()
    
    block_3 <- layer_add(list(shortcut_3, block_3)) %>%
      layer_activation(activation = "relu")
    
    # Final layers
    block_3 <- block_3 %>% layer_global_average_pooling_1d() %>%
      layer_dense(units = num_classes, activation = "softmax")
    
    return(keras_model(input, block_3))
  }

train_model_cut <-
  function(download,
           dataset = "BeetleFly",
           format = "txt",
           train_path = "",
           test_path = "",
           epochs = 100,
           batch_size = 32) {
    data <- get_data(download, dataset, format, train_path, test_path)
    if (format == "txt") {
      isTxt <- TRUE
    } else
      isTxt <- FALSE
    temp <- data_preprocessing(data, isTxt)
    train_x <- temp[[1]]
    train_y <- temp[[2]]
    test_x <- temp[[3]]
    test_y <- temp[[4]]
    class_number <- temp[[5]]
    input_shape <- temp[[6]]
    
    # Create and compile the model
    model <- resnet_hfwaz(input_shape, class_number)
    summary(model)
    
    callbacks <- list(
      callback_model_checkpoint(
        "best_model.h5",
        save_best_only = TRUE,
        monitor = "val_accuracy"
      ),
      callback_reduce_lr_on_plateau(
        monitor = "val_loss",
        factor = 0.5,
        patience = 20,
        min_lr = 0.0001
      )
      #callback_early_stopping(monitor = "val_loss", patience = 50, verbose = 1)
    )
    
    # Train the model
    history <- model %>% compile(
      loss = "categorical_crossentropy",
      optimizer = optimizer_adam(),
      metrics = c("accuracy"),
    ) %>% fit(
      train_x,
      train_y,
      batch_size = batch_size,
      epochs = epochs,
      validation_data = list(test_x, test_y),
      callbacks = callbacks
    )
    
    load_model_weights_hdf5(object = model,
                            filepath = paste(getwd(), "/best_model.h5", sep = ''))
    
    # Evaluate the model
    eval <- model %>% evaluate(test_x, test_y)
    prediction <- predict(model, test_x)
    prediction <- max.col(prediction)
    labels <- 1:length(test_y[1, ])
    test_y_decoded <- labels[max.col(test_y)]
    confusion_matrix <- table(test_y_decoded, prediction)
    
    sd(prediction)
    
    ret <- list(model, history, eval, confusion_matrix)
    names(ret) <-
      c("model", "history", "evaluation", "confusion_matrix")
    
    return(ret)
  }

train_model_uncut <-
  function(dataset = "BeetleFly",
           format = "txt",
           epochs = 100,
           batch_size = 32) {
    d <- read_data(dataset, format)
    if (format == "txt") {
      isTxt <- TRUE
    } else
      isTxt <- FALSE
    t <- data_preprocessing(d, isTxt)
    
    train_x <- t[[1]]
    train_y <- t[[2]]
    test_x <- t[[3]]
    test_y <- t[[4]]
    class_number <- t[[5]]
    input_shape <- t[[6]]
    
    
    model <- resnet_hfwaz(input_shape, class_number)
    
    callbacks <- list(
      callback_model_checkpoint(
        "best_model.h5",
        save_best_only = TRUE,
        monitor = "val_loss"
      ),
      callback_reduce_lr_on_plateau(
        monitor = "val_loss",
        factor = 0.5,
        patience = 20,
        min_lr = 0.0001
      )
      #callback_early_stopping(monitor = "val_loss", patience = 50, verbose = 1)
    )
    
    # Train the model
    history <- model %>% compile(
      loss = "categorical_crossentropy",
      optimizer = optimizer_adam(),
      metrics = c("accuracy"),
    ) %>% fit(
      train_x,
      train_y,
      batch_size = batch_size,
      epochs = epochs,
      validation_data = list(test_x, test_y),
      callbacks = callbacks
    )
    
    load_model_weights_hdf5(object = model,
                            filepath = paste(getwd(), "/best_model.h5", sep = ''))
    
    # Evaluate the model
    eval <- model %>% evaluate(test_x, test_y)
    prediction <- predict(model, test_x)
    prediction <- max.col(prediction)
    labels <- 1:length(test_y[1, ])
    test_y_decoded <- labels[max.col(test_y)]
    confusion_matrix <- table(test_y_decoded, prediction)
    
    sd(prediction)
    
    ret <- list(model, history, eval, confusion_matrix)
    names(ret) <-
      c("model", "history", "evaluation", "confusion_matrix")
    
    return(ret)
  }
