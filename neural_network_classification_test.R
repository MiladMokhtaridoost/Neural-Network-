library(keras)
neural_network_classification_test <- function(kt, model, n) {
  kt_list <- list()
  for (i in 1:n){
  kt_list[[i]] <- kt[,,i] 
 }
  prediction <- model %>% predict_on_batch(kt_list)
return(prediction)
  }

