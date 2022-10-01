# Dependencies
library(tidyverse)
library(C50)
library(mice)
library(partykit)
library(caret)
library(pROC)
library(data.table)
library(gridExtra)

# Loading training and testing data. Label is also cast into factor datatype and testing data is scaled
data_train <- read.csv('~/Desktop/data_transformed_train.csv')[,-1]%>% 
             mutate(Go_Trial = as.factor(Go_Trial)) 
data_test <- read.csv('~/Desktop/data_transformed_test.csv')[,-1] %>% 
             mutate(Go_Trial = as.factor(Go_Trial)) 
data_test <- data_test %>% mutate_at(colnames(data_test %>% select(-Go_Trial)),~ scale(., center = F))

# Simulate missing values using MCAR with probability of 0.1 and save to CSV
set.seed(123)
data_train_mcar <- ampute(data_train %>% select(-Go_Trial), prop = 0.1, mech = "MCAR")$amp %>% mutate(Go_Trial = data_train$Go_Trial)
summary(data_train_mcar)
#write.csv(data_train_mcar, '~/Desktop/data_transformed_train_mcar.csv')

# Loading training set with missing values CSV
data_train_mcar <- read.csv('~/Desktop/data_transformed_train_mcar.csv')[,-1]%>% 
             mutate(Go_Trial = as.factor(Go_Trial))
summary(data_train_mcar)

# Predictors used as well as label (Go_Trial)
colnames(data_train)

#' Instantiate an empty dataframe filled with NAs to save results for each value of n_Trials on training, validation, and testing
#' 
#' @param n_Trials A numeric vector.
#' @return A dataframe of dimension (length(n_Trials)x6).
create_results_df <- function(n_Trials = c(20, 40, 60, 80)){
    df <-  data.frame(matrix(NA, nrow = length(n_Trials), ncol = 6), row.names = n_Trials)
    colnames(df) <- c('Training_mean', 'Training_sd', 'Validation_mean', 'Validation_sd', 'Testing_mean', 'Testing_sd')
    return(df)
}

#' Split a dataset df into k folds
#' 
#' @param df A dataframe.
#' @param k An integer.
#' @param seed An integer, optional.
#' @return A list of dataframes of length k.
k_split <- function(df,k, seed = 1){
    set.seed(seed)
    grp <- rep(1:k, length.out = nrow(df))
    df |>
      mutate(grp = sample(grp, nrow(df), replace = F)) |>
      group_split(grp)|>
      map(\(d) select(d, -grp))
  }

#' Calculate precision from confusion matrix as TP/(TP + FP)
#' 
#' @param confusion_matrix A matrix of dimension 2x2.
#' @return A numeric value in the range [0,1], NA if TP = FP = 0.
calc_precision <- function(confusion_matrix){
    if(confusion_matrix[2,2] == 0 & confusion_matrix[2,1] == 0){
        return(NA)
    }
    return(confusion_matrix[2,2]/(confusion_matrix[2,2] + confusion_matrix[2,1]))
}

#' Calculate recall from confusion matrix as TP/(TP + FN)
#' 
#' @param confusion_matrix A matrix of dimension 2x2.
#' @return A numeric value in the range [0,1], NA if TP = FN = 0.
calc_recall <- function(confusion_matrix){
    if(confusion_matrix[2,2] == 0 & confusion_matrix[1,2] == 0){
        return(NA)
    }
    return(confusion_matrix[2,2]/(confusion_matrix[2,2] + confusion_matrix[1,2]))
}

#' Calculate specificity from confusion matrix as TN/(TN + FP)
#' 
#' @param confusion_matrix A matrix of dimension 2x2.
#' @return A numeric value in the range [0,1], NA if TN = FP = 0.
calc_specificity <- function(confusion_matrix){
    if(confusion_matrix[1,1] == 0 & confusion_matrix[2,1] == 0){
        return(NA)
    }
    return(confusion_matrix[1,1]/(confusion_matrix[1,1] + confusion_matrix[2,1]))
}

#' Calculate AUC score from model prediction ROC curve
#' 
#' @param pred A numeric vector of model predicitions.
#' @param true A numeric vector of ground truths.
#' @param quiet A boolean indicating if process should be silent, optional (default is FALSE).
#' @return A numeric value in the range [0,1].
calc_auc <- function(pred, true, quiet = FALSE){
    return(auc(roc(true, pred, quiet = quiet)))
}

#' Calculate F1-score from model precision and recall
#' 
#' @param precision A numeric value.
#' @param recall A numeric value.
#' @return A numeric value in the range [0,1], NA if precision = recall = NA (although TP = FN = FP = 0 is unlikely).
calc_f1 <- function(precision, recall){
    if(is.na(precision) & is.na(recall)){
        return(NA)
    }
    return(2*(precision * recall)/(precision + recall))
}

#' Perform k-fold cross validation using C5.0 with an option to tune number of boosting trials without early stopping
#' 
#' @param data_train A numeric dataframe for training.
#' @param data_test A numeric dataframe for testing.
#' @param ycol An integer indicating index of ground truth column, optional (default is 21).
#' @param window_sizes A numeric vector with values for 'trials' to try, optional (default is c(20, 40, 60, 80)).
#' @param k_folds An integer indicating number of CV folds, optional (default is 10).
#' @param n_iter An integer indicating number of times to repeat experiment, optional (default is 30).
#' @param start_seed An integer indicating the starting seed for each value in window_sizes, optional (default is 123).
#' @return A list of results including:
#'        trees being a dataframe of dimension of n_iter*k x length(window_sizes), 
#'        a dataframe of dimension length(window_sizes)x6 for each precision, recall, specificity, accuracy, AUC, and F1, 
#'        confusion_matrices being a list of length 3 with each element being a list of length n_iter*k,
#'        n_Trials in a numeric vector,
#'        best containing an integer being the best value from window_sizes,
#'        rulesets being a list of length 3 with each element being a list of length n_iter*k,
#'        ruleset_sizes being a dataframe of dimension of n_iter*k x length(window_sizes),
#'        predictor_usage being a dataframe of dimension of length(window_sizes) x (ncol(data_train)-1)
cross_validate_C50 <- function(data_train, 
                               data_test,
                               ycol = 21, 
                               window_sizes = c(20, 40, 60, 80), 
                               k_folds = 10, 
                               n_iter = 30, 
                               start_seed = 123){
    trees_w <- list()
    rulesets_w <- list()
    ruleset_sizes_w <- list()
    predictor_usage <- data.frame(matrix(0, nrow = length(window_sizes), ncol = ncol(data_train) - 1))
    colnames(predictor_usage) <- colnames(data_train[,-ycol])

    confusion_matrices_w <- list()
    precision_agg <- create_results_df(window_sizes)
    recall_agg <- create_results_df(window_sizes)
    specificity_agg <- create_results_df(window_sizes)
    auc_agg <- create_results_df(window_sizes)
    f1_agg <- create_results_df(window_sizes)
    acc_agg <- create_results_df(window_sizes)

    for(w in window_sizes){
        print("=======================================================================================")
        print("=======================================================================================")
        print(paste("Trials:", w))
        print("=======================================================================================")
        print("=======================================================================================")

        trees <- list()
        rulesets <- list()
        ruleset_sizes <- c()
        idw <- which(window_sizes == w)
        confusion_matrices <- list()
        precision <- data.frame(matrix(NA, nrow = k_folds*n_iter, ncol = 3))
        recall <- data.frame(matrix(NA, nrow = k_folds*n_iter, ncol = 3))
        specificity <- data.frame(matrix(NA, nrow = k_folds*n_iter, ncol = 3))
        auc <- data.frame(matrix(NA, nrow = k_folds*n_iter, ncol = 3))
        f1 <- data.frame(matrix(NA, nrow = k_folds*n_iter, ncol = 3))
        acc <- data.frame(matrix(NA, nrow = k_folds*n_iter, ncol = 3))
        
        count <- 0
        for(i in 1:n_iter){
            print("================================================================")
            print(paste("Iteration:", i))
            print("================================================================")

            kf <- k_split(data_train, k = k_folds, seed = start_seed + i - 1)

            for(k in 1:k_folds){
              print("----------------------------")  
              print(paste("Fold:", k))
              print("----------------------------")

              count <- count + 1
              
              # split data to validation and training and scale each
              scaled_split_val <- as.data.frame(kf[[k]]) %>% 
                           mutate_at(colnames(data_train[,-ycol]),~ scale(., center = F))
              scaled_split_tr <- as.data.frame(bind_rows(kf[-k])) %>% 
                           mutate_at(colnames(data_train[,-ycol]),~ scale(., center = F))
              
              tree <- C5.0(as.formula(paste(colnames(scaled_split_tr)[ycol], " ~ .")), data = scaled_split_tr, trials = w, control = C5.0Control(earlyStopping = FALSE, seed = start_seed + i))
              ruleset <- partykit:::.list.rules.party(C50:::as.party.C5.0(tree))
              trees[[count]] <- tree
              rulesets[[count]] <- ruleset
              ruleset_sizes <- c(ruleset_sizes, unname(sapply(ruleset, FUN = str_count, '&') + 1))
              predictor_usage[idw,] <- predictor_usage[idw,] + (data.frame(t(sapply(ruleset, FUN = str_count, colnames(data_train[,-ycol])))) %>% summarize_all(sum))

              pred_tr <- predict(tree, newdata = scaled_split_tr[,-ycol])
              pred_val <- predict(tree, newdata = scaled_split_val[,-ycol])
              pred_te <- predict(tree, newdata = data_test[,-ycol])

              confusion_matrices[['train']][[count]] <- confusionMatrix(scaled_split_tr[,ycol], pred_tr)$table
              confusion_matrices[['val']][[count]] <- confusionMatrix(scaled_split_val[,ycol], pred_val)$table
              confusion_matrices[['test']][[count]] <- confusionMatrix(data_test[,ycol], pred_te)$table
                
              precision[count, 1] <- calc_precision(confusion_matrices[['train']][[count]])
              precision[count, 2] <- calc_precision(confusion_matrices[['val']][[count]])
              precision[count, 3] <- calc_precision(confusion_matrices[['test']][[count]])

              recall[count, 1] <- calc_recall(confusion_matrices[['train']][[count]])
              recall[count, 2] <- calc_recall(confusion_matrices[['val']][[count]])
              recall[count, 3] <- calc_recall(confusion_matrices[['test']][[count]])

              specificity[count, 1] <- calc_specificity(confusion_matrices[['train']][[count]])
              specificity[count, 2] <- calc_specificity(confusion_matrices[['val']][[count]])
              specificity[count, 3] <- calc_specificity(confusion_matrices[['test']][[count]])

              auc[count, 1] <- calc_auc(predict(tree, newdata = scaled_split_tr[,-ycol], type = 'prob')[,2], scaled_split_tr[,ycol])
              auc[count, 2] <- calc_auc(predict(tree, newdata = scaled_split_val[,-ycol], type = 'prob')[,2], scaled_split_val[,ycol])
              auc[count, 3] <- calc_auc(predict(tree, newdata = data_test[,-ycol], type = 'prob')[,2], data_test[,ycol])

              f1[count, 1] <- calc_f1(precision[count, 1], recall[count, 1])
              f1[count, 2] <- calc_f1(precision[count, 2], recall[count, 2])
              f1[count, 3] <- calc_f1(precision[count, 3], recall[count, 3])

              acc[count, 1] <- mean(pred_tr == scaled_split_tr[,ycol])
              acc[count, 2] <- mean(pred_val == scaled_split_val[,ycol])
              acc[count, 3] <- mean(pred_te == data_test[,ycol])
            }
        }
        trees_w[[as.character(w)]] <- trees
        rulesets_w[[as.character(w)]] <- rulesets
        ruleset_sizes_w[[as.character(w)]] <- ruleset_sizes
        confusion_matrices_w[[as.character(w)]] <- confusion_matrices
        
        #' Aggregate metric results as mean and SD for each of n_Trials values and the 3 sets of instances while ignoring NAs
        #' 
        #' @param res A numeric dataframe of dimension n_iter*k x 3.
        #' @return A numeric dataframe of dimension 1x6
        agg <- function(res){
            return(res %>% summarize(across(everything(), list(mean = ~mean(.,na.rm = TRUE), sd = ~sd(.,na.rm = TRUE)))))
        }
        
        rownames(predictor_usage) <- window_sizes
        precision_agg[idw,] <- agg(precision)
        recall_agg[idw,] <- agg(recall) 
        specificity_agg[idw,] <- agg(specificity)
        auc_agg[idw,] <- agg(auc)
        f1_agg[idw,] <- agg(f1)
        acc_agg[idw,] <- agg(acc)
    }
    return(list(trees = as.data.frame(do.call(cbind, trees_w)), 
                rulesets = rulesets_w, 
                ruleset_sizes = as.data.frame(do.call(cbind, ruleset_sizes_w)), 
                predictor_usage = predictor_usage,
                confusion_matrices = confusion_matrices_w,
                precision = precision_agg, 
                recall = recall_agg, 
                specificity = specificity_agg, 
                auc = auc_agg, 
                f1 = f1_agg,
                acc = acc_agg,
                n_Trials = window_sizes,
                best = window_sizes[which(acc_agg[,5] == max(acc_agg[,5]))]))
}    

#' Plot metric results as mean and SD for each of n_Trials values and the 3 sets of instances as line plots as well as ruleset sizes and predictor usage as a boxplot and bargraph respectively
#' 
#' @param res A numeric dataframe of dimension length(res$window_sizes) x 3.
#' @return None
plot_metrics <- function(res){
    
    
    #' Format df to be of long format and compatible with plotting functions
    #' 
    #' @param df A numeric dataframe of dimension length(res$window_sizes) x 3.
    #' @return A numeric dataframe of dimension length(res$window_sizes)*3 x 4.
    format_df <- function(df, sfx = "_mean"){
        window_sizes <- rownames(df)
        res_m <- df %>% select(ends_with(sfx)) %>% mutate(n_Trials = window_sizes)
        colnames(res_m) <- gsub(sfx, "", colnames(res_m))
        return(res_m %>% pivot_longer(cols = colnames(res_m)[-c(ncol(res_m))], names_to = 'Set', values_to = gsub("_", "", sfx)))
    }
    
    #' plot a single metric line plot with errorbars, grouped by set
    #' 
    #' @param metric_results A numeric dataframe of dimension length(res$window_sizes)*3 x 4.
    #' @param title A string to be used as plot title.
    #' @return A ggplot
    plot_metric <- function(metric_results, title){
        df <- format_df(metric_results) %>% mutate(sd = format_df(metric_results, "_sd")$sd)
        options(repr.plot.width = 20, repr.plot.height = 20)
        ggplot(df, aes(x = n_Trials, y = mean, group = Set, col = Set)) +
        geom_line(aes(linetype = Set), size = 1) +
        geom_point(aes(shape = Set), size = 4) +
        geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = 0.2, position = position_dodge(0.05)) +
        ggtitle(title) +
        theme_bw() +
        theme(text = element_text(size = 20))
    }
    
    pr <- plot_metric(res$precision, "Precision")
    sp <- plot_metric(res$specificity, "Specificity")
    rc <- plot_metric(res$recall, "Recall")
    acc <- plot_metric(res$acc, "Accuracy")
    auc <- plot_metric(res$auc, "AUC")
    f1 <- plot_metric(res$f1, "F1-Score")
    
    rs_szs <- ggplot(res$ruleset_sizes %>% pivot_longer(cols = colnames(res$ruleset_sizes), names_to = 'n_Trials', values_to = 'Ruleset_size')) +
        geom_boxplot(aes(x = n_Trials, y = Ruleset_size)) +
        ggtitle('Ruleset sizes by number of boosting trials') +
        theme_bw()
    
    prd_cnt <- ggplot(res$predictor_usage %>% rownames_to_column('n_Trials') %>% pivot_longer(cols = colnames(res$predictor_usage), names_to = 'Predictor', values_to = 'N_Occurance'), aes(x = N_Occurance, y = reorder(Predictor, N_Occurance))) +
        geom_col(aes(fill = n_Trials)) +
        ggtitle('Predictor number of occurances filled by number of boosting trials') +
        theme_bw()
    
    grid.arrange(pr, sp, rc, acc, auc, f1, rs_szs, prd_cnt, ncol = 3, nrow = 3)
}

# 10-fold CV for 30 iterations using the MCAR dataset to tune n_Trials
start_seed <- 123
n_Trials <- c(20, 40, 60, 80)
k_folds <- 10
n_iter <- 30

res_mcar <- cross_validate_C50(data_train = data_train_mcar,
                         data_test = data_test,
                         ycol = 21,
                         window_sizes = n_Trials,
                         k_folds = k_folds,
                         n_iter = n_iter,
                         start_seed = start_seed)

plot_metrics(res_mcar)

print('Precision')
res_mcar$precision
print('Specificity')
res_mcar$specificity
print('Recall')
res_mcar$recall
print('Accuracy')
res_mcar$acc
print('AUC')
res_mcar$auc
print('F1-Score')
res_mcar$f1
print('Scores for n_Trials = best')
tab <- bind_rows(res_mcar$precision[which(res_mcar$n_Trials == res_mcar$best),], 
                 res_mcar$recall[which(res_mcar$n_Trials == res_mcar$best),], 
                 res_mcar$specificity[which(res_mcar$n_Trials == res_mcar$best),], 
                 res_mcar$acc[which(res_mcar$n_Trials == res_mcar$best),], 
                 res_mcar$auc[which(res_mcar$n_Trials == res_mcar$best),], 
                 res_mcar$f1[which(res_mcar$n_Trials == res_mcar$best),])
rownames(tab) <- c("Precision", "Recall", "Specificity", "Accuracy", "AUC", "F1-Score")
tab
print('Predictor Occurances in Rulesets')
res_mcar$predictor_usage

# 10-fold CV for 30 iterations using the original dataset and the best value for n_Trials from the tuning 
# step with the MCAR dataset with respect to validation accuracy
res <- cross_validate_C50(data_train = data_train,
                         data_test = data_test,
                         ycol = 21,
                         window_sizes = res_mcar$best,
                         k_folds = k_folds,
                         n_iter = n_iter,
                         start_seed = start_seed)

plot_metrics(res)

tab <- bind_rows(res$precision, res$recall, res$specificity, res$acc, res$auc, res$f1)
rownames(tab) <- c("Precision", "Recall", "Specificity", "Accuracy", "AUC", "F1-Score")
tab
res$predictor_usage
