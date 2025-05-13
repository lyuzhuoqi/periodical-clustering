library(readr)
library(igraph)
library(tidygraph)
library(ggraph)
library(dplyr)

# cluster2color字典
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

edges <- read_csv("structure/journal_level/edge_filtered.csv", col_types = cols())
nodes <- read_csv("structure/journal_level/node_filtered.csv", col_types = cols())

cat(sprintf("Number of nodes: %d\n", nrow(nodes)))
cat(sprintf("Number of edges: %d\n", nrow(edges)))

cat("Creating igraph object\n")
g <- graph_from_data_frame(d=edges, vertices=nodes, directed=TRUE)
# 检查是否存在自环
if (any(is.loop(g))) {
    warning("The graph contains self-loops. Removing self-loops.")
    g <- simplify(g, remove.loops = TRUE)
}
cat("Graph created\n")

layout_file <- "structure/journal_level/neulay_results/2d/fdl_iter_01001.csv"

cat("Reading layout from file\n")
layout_df <- read_csv(layout_file, col_names = FALSE, col_types = cols())
colnames(layout_df) <- c("x", "y")  # 添加列名
layout <- as.matrix(layout_df)

# 构造tidygraph对象，并添加颜色
tg <- as_tbl_graph(g) %>%
  activate(nodes) %>%
  mutate(
    x = layout_df$x,
    y = layout_df$y,
    color = cluster2color[as.character(kmeans_label)]
  )

cat("Plotting\n")
# 可视化
p <- ggraph(tg, layout = "manual", x = x, y = y) +
  geom_node_point(aes(color = color), size = 0.1, alpha = 1, show.legend = FALSE) +
  # geom_edge_link(alpha = 0.05, colour = "grey70", width = 0.05) +
  geom_edge_bundle_minimal0(# directed=TRUE, 
                            edge_alpha = 0.05,
                            edge_width = 0.05,
                            edge_colour = "grey70") +
  theme_void() +
  ggtitle("Journal-Level Citation Network")

ggsave("structure/journal_level/journal_citation_map_no_bundle.svg", plot = p, width = 10, height = 10, dpi = 300)
cat("Plot saved as journal_citation_map_no_bundle.svg\n")