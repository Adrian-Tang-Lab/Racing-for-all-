# -------------------------
# 加载必要的包
# -------------------------
library(caret)
library(glmnet)
library(xgboost)
library(randomForest)
library(Boruta)
library(e1071)
library(doParallel)
# 设置随机种子
set.seed(42)

library(readxl)

# 文件路径
file_path <- ""

# 读取数据
data <- read_excel(file_path)

# 确保 `treatment` 列为因子类型
data$treatment <- as.factor(data$treatment)

# 检查目标变量是否正确加载
if (!is.factor(data$treatment)) {
  stop("目标变量 'treatment' 不是因子类型，请检查数据格式！")
}

# 数据划分为训练集和测试集
set.seed(42)
train_index <- createDataPartition(data$treatment, p = 0.8, list = FALSE)

X_train <- data[train_index, -which(colnames(data) %in% c("treatment", "Sample"))] # 去掉 `treatment` 和 `Sample`
y_train <- data$treatment[train_index]

X_test <- data[-train_index, -which(colnames(data) %in% c("treatment", "Sample"))]
y_test <- data$treatment[-train_index]

# 检查目标变量类型
if (!is.factor(y_train) || !is.factor(y_test)) {
  stop("训练集或测试集的目标变量不是因子类型，请检查代码！")
}

# 标准化特征
preproc <- preProcess(X_train, method = c("center", "scale"))
X_train <- predict(preproc, X_train)
X_test <- predict(preproc, X_test)

# 检查数据
print(head(X_train))
print(levels(y_train)) # 检查因子类别
# -------------------------
# 设置多核并行
# -------------------------
num_cores <- detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)
# -------------------------
# 初始化结果存储与日志
# -------------------------
results <- list()
error_log <- c()
output_dir <- "D:/Sepsis/risk_prediction_results/"
if (!dir.exists(output_dir)) dir.create(output_dir)

# -------------------------
# 模型训练和特征选择
# -------------------------

# XGBoost
tryCatch({
  xgb_grid <- expand.grid(
    nrounds = c(50, 100, 150),
    max_depth = c(3, 6, 9),
    eta = c(0.01, 0.1, 0.2),
    gamma = c(0, 1, 5),
    colsample_bytree = c(0.6, 0.8, 1.0),
    min_child_weight = c(1, 3, 5),
    subsample = c(0.6, 0.8, 1.0)
  )
  xgb_model <- train(
    X_train, y_train,
    method = "xgbTree",
    trControl = cv_control,
    tuneGrid = xgb_grid
  )
  results$xgb <- xgb_model
}, error = function(e) {
  error_log <<- c(error_log, paste("xgb failed:", e$message))
})

# 随机森林
tryCatch({
  rf_grid <- expand.grid(
    mtry = c(1, sqrt(ncol(X_train)), ncol(X_train)),
    splitrule = c("gini", "extratrees"),
    min.node.size = c(1, 5, 10)
  )
  rf_model <- train(
    X_train, y_train,
    method = "ranger",
    trControl = cv_control,
    tuneGrid = rf_grid,
    importance = "impurity"
  )
  results$rf <- rf_model
}, error = function(e) {
  error_log <<- c(error_log, paste("rf failed:", e$message))
})

# LASSO-RFE
tryCatch({
  lasso_rfe_control <- rfeControl(
    functions = rfFuncs,
    method = "cv",
    number = 3,
    verbose = TRUE,
    allowParallel = TRUE
  )
  lasso_rfe <- rfe(
    X_train, y_train,
    sizes = c(5, 10, 15),
    rfeControl = lasso_rfe_control
  )
  results$lasso_rfe <- lasso_rfe
}, error = function(e) {
  error_log <<- c(error_log, paste("lasso_rfe failed:", e$message))
})

# Elastic Net
tryCatch({
  elastic_net_grid <- expand.grid(
    alpha = seq(0, 1, length = 5),
    lambda = 10^seq(-3, 1, length = 10)
  )
  elastic_net_model <- train(
    X_train, y_train,
    method = "glmnet",
    trControl = cv_control,
    tuneGrid = elastic_net_grid
  )
  results$elastic_net <- elastic_net_model
}, error = function(e) {
  error_log <<- c(error_log, paste("elastic_net failed:", e$message))
})

# Boruta
tryCatch({
  boruta_model <- Boruta(
    x = as.data.frame(X_train),
    y = y_train,
    doTrace = 2,
    maxRuns = 100
  )
  final_boruta <- TentativeRoughFix(boruta_model)
  boruta_selected <- getSelectedAttributes(final_boruta, withTentative = TRUE)
  results$boruta <- list(model = boruta_model, selected_features = boruta_selected)
}, error = function(e) {
  error_log <<- c(error_log, paste("boruta failed:", e$message))
})

# SVM-RFE
tryCatch({
  svm_rfe_control <- rfeControl(
    functions = caretFuncs,
    method = "cv",
    number = 3,
    verbose = TRUE,
    allowParallel = TRUE
  )
  svm_rfe <- rfe(
    x = X_train,
    y = y_train,
    sizes = c(5, 10, 15),
    rfeControl = svm_rfe_control
  )
  results$svm_rfe <- svm_rfe
}, error = function(e) {
  error_log <<- c(error_log, paste("svm_rfe failed:", e$message))
})

# -------------------------
# 保存特征重要性结果
# -------------------------
for (model_name in names(results)) {
  tryCatch({
    model <- results[[model_name]]
    output_path <- paste0(output_dir, "top_features_", model_name, ".txt")
    
    if (model_name %in% c("xgb", "rf")) {
      # XGBoost 或 随机森林
      importance <- varImp(model)$importance
      write.table(
        importance,
        file = output_path,
        sep = "\t", quote = FALSE, col.names = TRUE, row.names = TRUE
      )
    } else if (model_name == "elastic_net") {
      # Elastic Net
      importance <- varImp(model)$importance
      write.table(
        importance,
        file = output_path,
        sep = "\t", quote = FALSE, col.names = TRUE, row.names = TRUE
      )
    } else if (model_name == "lasso_rfe") {
      # LASSO-RFE
      write.table(
        model$variables,
        file = output_path,
        sep = "\t", quote = FALSE, col.names = TRUE, row.names = FALSE
      )
    } else if (model_name == "boruta") {
      # Boruta
      boruta_importance <- attStats(model$model)
      write.table(
        boruta_importance,
        file = output_path,
        sep = "\t", quote = FALSE, col.names = TRUE, row.names = TRUE
      )
    } else if (model_name == "svm_rfe") {
      # SVM-RFE
      write.table(
        model$variables,
        file = output_path,
        sep = "\t", quote = FALSE, col.names = TRUE, row.names = FALSE
      )
    }
  }, error = function(e) {
    error_log <<- c(error_log, paste("importance extraction failed for", model_name, ":", e$message))
  })
}

# -------------------------
# 保存错误日志
# -------------------------
if (length(error_log) > 0) {
  writeLines(error_log, con = paste0(output_dir, "error_log.txt"))
  cat("Errors occurred during processing. Check error_log.txt for details.\n")
} else {
  cat("No errors occurred during processing.\n")
}

stopCluster(cl)