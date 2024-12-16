# 加载必要的包
library(movMF)
library(cluster) # 用于计算 Silhouette Score
library(ggplot2) # 用于绘图
library(reticulate)

# 激活 Python 环境并加载数据
use_condaenv("p2v", required = TRUE)
reticulate::py_run_string("
import os
import pickle
print('Current working directory:', os.getcwd())
with open('data/2010s/vectors.pkl', 'rb') as f:
    vectors = pickle.load(f)
")
vectors <- py$vectors

# 确保数据为矩阵
vectors <- as.matrix(vectors)

# 检查数据是否包含异常值
if (any(is.na(vectors))) {
  cat("Data contains NA values. Removing rows with NA.\n")
  vectors <- na.omit(vectors)
}

if (any(rowSums(vectors^2) == 0)) {
  cat("Data contains zero vectors. Removing zero vectors.\n")
  vectors <- vectors[rowSums(vectors^2) > 0, ]
}

if (any(is.nan(vectors)) || any(is.infinite(vectors))) {
  stop("Data contains NaN or infinite values. Please check preprocessing.")
}

# 定义 K 值范围
k_values <- seq(10, min(100, nrow(vectors)), by = 10)

# 初始化存储结果的向量
inertia <- numeric(length(k_values))
silhouette_scores <- numeric(length(k_values))

# 遍历每个 K 值
for (k in k_values) {
  cat("Fitting model for k =", k, "\n")
  
  # 模型拟合
  model <- movMF(vectors, k = k)
  
  # 计算 inertia
  inertia[which(k_values == k)] <- -logLik(model)
  
  # 获取簇标签
  labels <- apply(model$theta %*% t(vectors), 2, which.max)
  
  # 计算 Silhouette Score
  if (length(unique(labels)) > 1) {
    silhouette_scores[which(k_values == k)] <- mean(silhouette(labels, dist(vectors))[, 3])
  } else {
    silhouette_scores[which(k_values == k)] <- NA
    cat("Silhouette calculation skipped for k =", k, "due to single cluster.\n")
  }
  
  cat("Finished fitting model for k =", k, "\n")
}

# 检查有效数据
valid_indices <- !is.na(inertia) & !is.na(silhouette_scores)
k_values <- k_values[valid_indices]
inertia <- inertia[valid_indices]
silhouette_scores <- silhouette_scores[valid_indices]

# 确保有有效数据
if (length(k_values) > 0) {
  # 绘制 Elbow 方法图
  elbow_plot <- ggplot(data.frame(k = k_values, inertia = inertia), aes(x = k, y = inertia)) +
    geom_line() +
    geom_point() +
    labs(title = "Elbow Method", x = "Number of clusters (k)", y = "Inertia") +
    theme_minimal()
  
  # 保存 Elbow 方法图
  ggsave("clustering/movMF_elbow_plot.png", plot = elbow_plot, width = 8, height = 6)
  
  # 绘制 Silhouette Scores 图
  silhouette_plot <- ggplot(data.frame(k = k_values, silhouette = silhouette_scores), aes(x = k, y = silhouette)) +
    geom_line() +
    geom_point() +
    labs(title = "Silhouette Scores", x = "Number of clusters (k)", y = "Silhouette Score") +
    theme_minimal()
  
  # 保存 Silhouette Scores 图
  ggsave("clustering/movMF_silhouette_plot.png", plot = silhouette_plot, width = 8, height = 6)
  
  # 输出图表
  print(elbow_plot)
  print(silhouette_plot)
} else {
  cat("No valid data for plotting.\n")
}
