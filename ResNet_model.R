library(keras)
library(caret)
library(dplyr) # For the %>% operator

layer_shortcut <- function(object, ninput, noutput, stride) {
  if(ninput == noutput)
    object %>% 
    layer_lambda(function(x) x)
  else{
    object <- object %>%
      layer_average_pooling_1d(1, stride)
    
    a <- object %>%
      layer_lambda(function(x) x)
    
    b <- object %>% 
      layer_lambda(., function(x) k_zeros_like(x))
    
    layer_concatenate(c(a, b))
  }
}

layer_basic_block <- function(object, ninput, noutput, stride) {
  a <- object %>%	
    layer_conv_2d(noutput, 3, stride, 'same', kernel_initializer = 'lecun_normal') %>%
    layer_batch_normalization() %>%
    layer_activation('relu') %>%
    layer_conv_2d(noutput, 3, 1, 'same', kernel_initializer = 'lecun_normal') %>%
    layer_batch_normalization()
  
  b <- object %>%
    layer_shortcut(ninput, noutput, stride)
  
  layer_add(c(a, b))
}


build_block <- function(object, ninput, noutput, count, stride) {
  for(i in 1:count)
    object <- object %>% 
      layer_basic_block(if(i == 1) ninput else noutput,
                        noutput, 
                        if(i == 1) stride else 1
      )
  object
}

build_resnet <- function(input_shape, classes, depth = 20) { 
  n <- (depth - 2) / 6
  
  input <- layer_input(shape=input_shape)
  
  output <- input %>%
    layer_conv_1d(16, 3, 1, 'same', kernel_initializer = 'lecun_normal') %>%
    layer_batch_normalization() %>%
    layer_activation('relu') %>%
    build_block(16, 16, n, 1) %>% # Primer conjunto
    build_block(16, 32, n, 2) %>% # Segundo conjunto
    build_block(32, 64, n, 2) %>% # Tercer conjunto
    layer_average_pooling_1d(8, 1) %>%
    layer_flatten() %>%
    layer_dense(10) %>%
    layer_activation_softmax()
  
  return(keras_model(input, output))
}

validation <- function(depth, x, y, input_shape, classes, batch_size, epochs, 
                                       y_tags=NULL, k = 5, callbacks = NULL, ...) {
  folds <- createFolds(y = if(is.null(y_tags)) y else y_tags, 
                       k = k, list = F) # Stratified
  histories <- list()
  for(f in 1:k){
    print(paste(f, 'of', k))
    
    model.aux <- build_resnet(depth=depth, input_shape=input_shape, classes=class_number) %>% compile(
      optimizer=optimizer_sgd(lr=0.1, momentum=0.9, decay=0.0001), 
      ...
    ) 
    ind <- which(folds == f)
    # x_train <- x[-ind,,]
    # y_train <- y[-ind]
    # x_valid <- x[ind,,]
    # y_valid <- y[ind]
    
    histories[[f]] <- model.aux %>% fit(
      x, y,
      epochs = epochs,
      batch_size = batch_size,
      validation_data = list(x, y),
      verbose = 0,
      callbacks = c(callback_reduce_lr_on_plateau(verbose=0, patience=10, factor=0.1))
    )
  }
  histories
}

model <- build_resnet(input_shape=input_shape, classes=class_number)

model.cv <- validation(20, input_shape=input_shape, classes=class_number,
                         train_x, train_y, batch_size=5,
                         epochs=10, y_tags=tags_y, k=5,
                         loss='categorical_crossentropy',
                         metrics=c('accuracy')
)

# Compiling and training the model
model %>%
  compile(
    optimizer=optimizer_sgd(lr=0.1, momentum=0.9, decay=0.0001),
    loss='categorical_crossentropy', metrics=c('accuracy')
  ) %>%
  fit(
    train_x, train_y, validation_split=0.2,
    verbose=0, batch_size=5, epochs=10,
    callbacks = c(callback_reduce_lr_on_plateau(verbose=0, patience=10, factor=0.1))
  )

# Getting and plotting the predictions
predictions <- predict(model, x_test)
print(paste('Predictions:', paste0(max.col(predictions), collapse=' ')))
print(paste('Real values:', paste0(max.col(y_test), collapse=' ')))

###########################
layer_shortcut <- function(object, ninput, noutput, stride) {
  if(ninput == noutput)
    object %>% 
    layer_lambda(function(x) x)
  else{
    object <- object %>%
      layer_average_pooling_1d(1, stride)
    
    a <- object %>%
      layer_lambda(function(x) x)
    
    b <- object %>% 
      layer_lambda(., function(x) k_zeros_like(x))
    
    layer_concatenate(c(a, b))
  }
}

layer_basic_block <- function(object, ninput, noutput, stride) {
  a <- object %>%	
    layer_conv_1d(noutput, 3, stride, 'same', kernel_initializer = 'lecun_normal') %>%
    layer_batch_normalization() %>%
    layer_activation('relu') %>%
    layer_conv_1d(noutput, 3, 1, 'same', kernel_initializer = 'lecun_normal') %>%
    layer_batch_normalization()
  
  b <- object %>%
    layer_shortcut(ninput, noutput, stride)
  
  layer_add(c(a, b)) #%>%
  #layer_activation('relu') QUITAMOS ESTO
}


build_block <- function(object, ninput, noutput, count, stride) {
  for(i in 1:count)
    object <- object %>% 
      layer_basic_block(if(i == 1) ninput else noutput,
                        noutput, 
                        if(i == 1) stride else 1
      )
  object
}

build_resnet_cifar10 <- function(depth = 20) { 
  n <- (depth - 2) / 6
  
  input <- layer_input(shape=input_shape)
  
  output <- input %>%
     layer_conv_1d(16, 3, 1, 'same', kernel_initializer = 'lecun_normal') %>%
     layer_batch_normalization() %>%
     layer_activation('relu') #%>%
    # build_block(16, 16, n, 1) %>% # Primer conjunto
    # build_block(16, 32, n, 2) %>% # Segundo conjunto
    # build_block(32, 64, n, 2) %>% # Tercer conjunto
    # layer_average_pooling_1d(8, 1) %>%
    # layer_flatten() %>%
    # layer_dense(10) %>%
    #layer_activation_softmax()		
  
  keras_model(input, output)
}


model <- build_resnet_cifar10(20)

# Compiling and training the model
model %>%
  compile(
    optimizer=optimizer_sgd(lr=0.1, momentum=0.9, decay=0.0001),
    loss='categorical_crossentropy', metrics=c('accuracy')
  ) %>%
  fit(
    train_x, train_y, validation_split=0.2,
    verbose=0, batch_size=5, epochs=10,
    callbacks = c(callback_reduce_lr_on_plateau(verbose=0, patience=10, factor=0.1))
  )



# Doing cross validation (it concatenates all the results)
model.cv <- do.cross.validation.resnet(20,
                                       x_train, y_train, batch_size=5, 
                                       epochs=10, y_tags=y_tags, k=5,
                                       loss='categorical_crossentropy', 
                                       metrics=c('accuracy') 
)

