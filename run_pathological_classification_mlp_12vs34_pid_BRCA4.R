library(AUC)
library(keras)
data_path <- "./data"
result_path <- "./result_12vs34_pid"
cohorts<- c("TCGA-BRCA")
#cohorts <- c("TCGA-BRCA", "TCGA-COAD", "TCGA-ESCA", "TCGA-HNSC", "TCGA-KICH",
#             "TCGA-KIRC", "TCGA-KIRP", "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC",
#            "TCGA-PAAD", "TCGA-READ", "TCGA-STAD", "TCGA-TGCT", "TCGA-THCA")
#cohorts <- c("TCGA-ACC",  "TCGA-BLCA", "TCGA-BRCA", "TCGA-COAD", "TCGA-ESCA",
#             "TCGA-HNSC", "TCGA-KICH", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LIHC",
#             "TCGA-LUAD", "TCGA-LUSC","TCGA-MESO", "TCGA-READ", "TCGA-SKCM","TCGA-STAD", "TCGA-THCA", "TCGA-UVM")

#args <- commandArgs(trailingOnly = TRUE)
#cohorts <- cohorts[as.numeric(args[[1]]) %% length(cohorts) + 1]


source("classification_helper.R")
source("neural_network_classification_train_AUC.R")
source("neural_network_classification_test.R")

#cohorts <- c("TCGA-BRCA", "TCGA-COAD", "TCGA-ESCA", "TCGA-HNSC", "TCGA-KICH",
#             "TCGA-KIRC", "TCGA-KIRP", "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC",
#            "TCGA-PAAD", "TCGA-READ", "TCGA-STAD", "TCGA-TGCT", "TCGA-THCA")
#cohorts <- c("TCGA-KICH")


#data_path <- "C:/Users/KATY/Desktop/arezou/gsbc-master/data"
#result_path <- "C:/Users/KATY/Desktop/arezou/gsbc-master/Result"
#data_path <- "./data"
#result_path <- "./results_1_vs_234"
pathway <- "pid"

for (cohort in cohorts) {
  if (dir.exists(sprintf("%s/%s", result_path, cohort)) == FALSE) {
    dir.create(sprintf("%s/%s", result_path, cohort)) 
  }
 replication=82 
#  for (replication in 69:71) {
    if (file.exists(sprintf("%s/%s/mlp_pathway_%s_measure_AUROC_replication_%d_result.RData", result_path, cohort, pathway, replication)) == FALSE) {
      load(sprintf("%s/%s.RData", data_path, cohort))
      
      common_patients <- intersect(rownames(TCGA$clinical)[which(is.na(TCGA$clinical$pathologic_stage) == FALSE)], rownames(TCGA$mrna))
      
      X <- log2(TCGA$mrna[common_patients,] + 1)
      y <- rep(NA, length(common_patients))
      
     # y[TCGA$clinical[common_patients, "pathologic_stage"] %in% c("Stage I",  "Stage IA",  "Stage IB",  "Stage IC")] <- +1
     # y[TCGA$clinical[common_patients, "pathologic_stage"] %in% c("Stage II", "Stage IIA", "Stage IIB", "Stage IIC",
     #                                                             "Stage III", "Stage IIIA", "Stage IIIB", "Stage IIIC",
     #                                                             "Stage IV",  "Stage IVA",  "Stage IVB",  "Stage IVC")] <- -1
 
      y[TCGA$clinical[common_patients, "pathologic_stage"] %in% c("Stage I",  "Stage IA",  "Stage IB",  "Stage IC",
                                                                 "Stage II", "Stage IIA", "Stage IIB", "Stage IIC")] <- +1
      y[TCGA$clinical[common_patients, "pathologic_stage"] %in% c("Stage III", "Stage IIIA", "Stage IIIB", "Stage IIIC",
                                                                 "Stage IV",  "Stage IVA",  "Stage IVB",  "Stage IVC")] <- -1
     
      valid_patients <- which(is.na(y) == FALSE)
      valid_features <- as.numeric(which(apply(X[valid_patients,], 2, sd) != 0))
      X <- X[valid_patients, valid_features]
      y <- y[valid_patients]
      
      negative_indices <- which(y == -1)
      positive_indices <- which(y == +1)
      
      batch_size_set <- c(25, 50,75,100)
      #epsilon <- 1e-5
      fold_count <- 4
      train_ratio <- 0.8
      iteration_count <- 10000
      
################################
      max_iteration <- 100
      sample_per_gradient_update <- 10 
################################
      
      pathways <- read_pathways(pathway)
      gene_names <- sort(unique(unlist(sapply(1:length(pathways), FUN = function(x) {pathways[[x]]$symbols}))))
      X <- X[, which(colnames(X) %in% gene_names)]
      
      auroc_matrix <- matrix(NA, nrow = fold_count, ncol = length(batch_size_set), dimnames = list(1:fold_count, sprintf("%g", batch_size_set)))
      
      set.seed(1606 * replication)
      train_negative_indices <- sample(negative_indices, ceiling(train_ratio * length(negative_indices)))
      train_positive_indices <- sample(positive_indices, ceiling(train_ratio * length(positive_indices)))
      
      negative_allocation <- sample(rep(1:fold_count, ceiling(length(train_negative_indices) / fold_count)), length(train_negative_indices))
      positive_allocation <- sample(rep(1:fold_count, ceiling(length(train_positive_indices) / fold_count)), length(train_positive_indices))
      for (fold in 1:fold_count) {
        train_indices <- c(train_negative_indices[which(negative_allocation != fold)], train_positive_indices[which(positive_allocation != fold)])
        test_indices <- c(train_negative_indices[which(negative_allocation == fold)], train_positive_indices[which(positive_allocation == fold)])
        
        X_train <- X[train_indices,]
        X_test <- X[test_indices,]
        X_train <- scale(X_train)
        X_test <- (X_test - matrix(attr(X_train, "scaled:center"), nrow = nrow(X_test), ncol = ncol(X_test), byrow = TRUE)) / matrix(attr(X_train, "scaled:scale"), nrow = nrow(X_test), ncol = ncol(X_test), byrow = TRUE)
        
        N_train <- nrow(X_train)
        N_test <- nrow(X_test)
        N_pathway <- length(pathways)
        K_train <- array(0, dim = c(N_train, N_train, N_pathway))
        K_test <- array(0, dim = c(N_test, N_train, N_pathway))
        for (m in 1:N_pathway) {
          feature_indices <- which(colnames(X_train) %in% pathways[[m]]$symbols)
          D_train <- pdist(X_train[, feature_indices], X_train[, feature_indices])
          D_test <- pdist(X_test[, feature_indices], X_train[, feature_indices])
          sigma <- mean(D_train)
          K_train[,,m] <- exp(-D_train^2 / (2 * sigma^2))
          K_test[,,m] <- exp(-D_test^2 / (2 * sigma^2))
        }
        
        y_train <- y[train_indices]
        y_test <- y[test_indices]
        
        for (bs in batch_size_set) {
          print(sprintf("running fold = %d, bs = %g", fold, bs))
          parameters <- list()
          parameters$bs <- bs
          #parameters$epsilon <- epsilon
          #parameters$iteration_count <- iteration_count
          
          parameters$max_iteration <- max_iteration
          parameters$sample_per_gradient_update <- sample_per_gradient_update
          
          model <- neural_network_classification_train(N_train, N_pathway, K_train, y_train, parameters)
          prediction <- neural_network_classification_test(K_test, model, N_pathway)
          auroc_matrix[fold, sprintf("%g", bs)] <- auc(roc(prediction, as.factor(1 * (y_test == +1))))
        }
      }
      
      bs_star_AUROC <- batch_size_set[max.col(t(colMeans(auroc_matrix)), ties.method = "last")]    
      
      train_indices <- c(train_negative_indices, train_positive_indices)
      test_indices <- setdiff(1:length(y), train_indices)
      
      X_train <- X[train_indices,]
      X_test <- X[test_indices,]
      X_train <- scale(X_train)
      X_test <- (X_test - matrix(attr(X_train, "scaled:center"), nrow = nrow(X_test), ncol = ncol(X_test), byrow = TRUE)) / matrix(attr(X_train, "scaled:scale"), nrow = nrow(X_test), ncol = ncol(X_test), byrow = TRUE)
      
      N_train <- nrow(X_train)
      N_test <- nrow(X_test)
      N_pathway <- length(pathways)
      K_train <- array(0, dim = c(N_train, N_train, N_pathway))
      K_test <- array(0, dim = c(N_test, N_train, N_pathway))
      for (m in 1:N_pathway) {
        feature_indices <- which(colnames(X_train) %in% pathways[[m]]$symbols)
        D_train <- pdist(X_train[, feature_indices], X_train[, feature_indices])
        D_test <- pdist(X_test[, feature_indices], X_train[, feature_indices])
        sigma <- mean(D_train)
        K_train[,,m] <- exp(-D_train^2 / (2 * sigma^2))
        K_test[,,m] <- exp(-D_test^2 / (2 * sigma^2))
      }
      
      y_train <- y[train_indices]
      y_test <- y[test_indices]
      
      parameters <- list()
      parameters$bs <- bs_star_AUROC
      #parameters$epsilon <- epsilon
      parameters$iteration_count <- iteration_count
      
      model <- neural_network_classification_train(N_train, N_pathway, K_train, y_train, parameters)
      prediction <- neural_network_classification_test(K_test, model, N_pathway)
      result <- list()
      result$AUROC <- auc(roc(prediction, as.factor(1 * (y_test == +1))))
       save_model_hdf5(model, file = sprintf("%s/%s/mlp_pathway_%s_measure_AUROC_replication_%d_model.h5", result_path, cohort, pathway, replication))
     # save("model", file = sprintf("%s/%s/mlp_pathway_%s_measure_AUROC_replication_%d_model.RData", result_path, cohort, pathway, replication))
      save("result", file = sprintf("%s/%s/mlp_pathway_%s_measure_AUROC_replication_%d_result.RData", result_path, cohort, pathway, replication))
    } 
  }
#}
