library(keras)
neural_network_classification_train <- function(input_dim, N_pathway, k, y, parameters) {

  main_inputs <- list()
  sample_weights <- layer_dense(units = 1, use_bias = FALSE, input_shape = input_dim)
  intermediate_outputs <- list()
  for (i in 1:N_pathway) {
    main_inputs[[i]] <-layer_input(shape = input_dim, dtype = "float32")
    intermediate_outputs[[i]] <- main_inputs[[i]] %>% sample_weights
  }
  
  
  
  score_layer <- layer_concatenate(intermediate_outputs)
  
  output_layer <- score_layer %>% 
    layer_dense(units = 1, activation = 'sigmoid', use_bias = TRUE, input_shape = N_pathway, kernel_constraint = constraint_nonneg(),kernel_regularizer = regularizer_l1(l=0.02) )
 
  
  
  model <- keras_model(inputs = main_inputs, outputs = output_layer)
  
  
  model %>% compile(

    optimizer = optimizer_adam(),
    loss = 'binary_crossentropy',
    metrics = c('AUC')
  )
  k_list <- list()
  for (i in 1:N_pathway){
    k_list[[i]] <- k[,,i] 
  }
  y[y == -1] <- 0 
  model %>% fit(k_list, y, epochs = 10000, batch_size = parameters$bs, shuffle = TRUE)
 # model %>% save_model_hdf5("my_model.h5")
return(model)
}


