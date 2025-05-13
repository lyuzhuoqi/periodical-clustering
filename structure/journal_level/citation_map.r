library(readr)
library(igraph)
library(edgebundle)
library(ggplot2)
library(dplyr)

# 配色方案
cluster2color <- c(
    '0' = '#8FA329',
    '1' = '#D26B04',
    '2' = '#FF5C29',
    '3' = '#A679FF',
    '4' = '#0099FF',
    '5' = '#FF7C80',
    '6' = '#FFCC00',
    '7' = '#7030A0',
    '8' = '#000099',
    '9' = '#92D050',
    '10' = '#FC9320',
    '11' = '#9A0000',
    '12' = '#FE0000',
    '13' = '#375623',
    '14' = '#D20000',
    '15' = '#FBFF57',
    '16' = '#CC00FF',
    '17' = '#336699',
    '18' = '#85D6FF',
    '19' = '#6C0000',
    '20' = '#9900CC',
    '21' = '#0000F2',
    '22' = '#CCB3FF',
    '23' = '#16A90F',
    '24' = '#187402',
    '25' = '#66FF66'
)

# 读取节点和边
edges <- read_csv("structure/journal_level/edge.csv", col_types = cols())
nodes <- read_csv("structure/journal_level/node.csv", col_types = cols())

# 如果有权重，按权重降序筛选；没有的话可以按其它列或原始顺序
edges <- edges %>%
  group_by(source) %>%
  arrange(desc(weight), .by_group = TRUE) %>%
  slice_head(n = 20) %>%
  ungroup()

# 保留出现在边表中的所有节点
nodes <- nodes[nodes$name %in% unique(c(edges$source, edges$target)), ]

cat(sprintf("Number of nodes after filtering: %d\n", nrow(nodes)))
cat(sprintf("Number of edges after filtering: %d\n", nrow(edges)))

# 读取布局
layout_df <- read_csv("structure/journal_level/layout.csv", col_types = cols())
# 保证节点顺序与layout一致
nodes <- nodes[match(layout_df$name, nodes$name), ]
stopifnot(all(nodes$name == layout_df$name))

# 生成布局矩阵
layout <- as.matrix(layout_df[, c("x", "y")])

# 创建igraph对象
cat("Creating igraph object\n")
g <- graph_from_data_frame(d=edges, vertices=nodes, directed=TRUE)
cat("Graph created\n")

# 边捆绑
cat("Creating edge bundles\n")
fbundle <- edge_bundle_path(g, layout)
cat("Edge bundles created\n")

# 节点数据框（含颜色）
nodes_plot <- layout_df %>%
  mutate(
    color = cluster2color[as.character(nodes$kmeans_label)],
    x = x,
    y = y
  )

cat("Plotting\n")
# 可视化
p <- ggplot() +
  geom_path(data = fbundle, aes(x, y, group = group), color = "grey70", size = 0.1, alpha = 0.05) +
  geom_point(data = nodes_plot, aes(x = x, y = y, color = color), size = 0.1, alpha = 0.5, show.legend = FALSE) +
  scale_color_identity() +
  theme_void() +
  ggtitle("Journal-Level Citation Map")

ggsave("structure/journal_level/journal_citation_map.svg", plot = p, width = 10, height = 10, dpi = 300)
cat("Plot saved as journal_citation_map_bundle.svg")