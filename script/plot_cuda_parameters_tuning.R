library(ggplot2)
library(here)

save_svg_plot = function(plot,
                         name = "test",
                         folder = "./",
                         type = "standard",
                         width = 7,
                         height = 7,
                         pointsize = 12)
{
  final_name = paste(folder, name, ".svg", sep = "")
  if (type == "standard")
  {
    svg(
      final_name,
      width = width,
      height = height,
      pointsize = pointsize
    )
    replayPlot(plot)
    dev.off()
  } else if (type == "ggplot")
  {
    #ggsave(nome_finale, plot, width = width, height = height)
    svg(
      final_name,
      width = width,
      height = height,
      pointsize = pointsize
    )
    print(plot)
    dev.off()
  }
}

folder = paste(here(), "/script", sep = "")
setwd(folder)
names = c("citeseer", "cora", "pubmed")
plots = vector("list", length = 3)
i = 1
for (name in names)
{
  folder_name = paste("../output/tuning_cuda_", name, ".txt", sep = "")
  data = read.csv(folder_name, sep = "")
  
  x_axis_labels <-
    min(data$num_blocks_factor):max(data$num_blocks_factor)
  title = name
  
  plot = ggplot(data = data,
                aes(
                  x = num_blocks_factor,
                  y = avg_epoch_training_time,
                  color = factor(num_threads)
                )) +
    geom_line(linewidth = 1.5) +
    labs(x = "Multiplicative factor to obtain the number of blocks", y = "Average training time per epoch", color = "Number of threads") +
    theme_minimal() +
    ggtitle(title) +
    scale_x_continuous(labels = x_axis_labels, breaks = x_axis_labels) +
    theme(
      axis.text = element_text(size = rel(1.2)),
      axis.title = element_text(size = rel(1.5)),
      plot.title = element_text(
        face = "bold",
        size = rel(2),
        hjust = 0.5
      ),
      legend.text = element_text(size = rel(1.2)),
      legend.title = element_text(size = rel(1.2)),
      legend.position = c(1, 1),
      legend.justification = c(1, 1)
    )
  name = paste("tuning_cuda_", name, sep = "")
  save_svg_plot(plot,
                name = name,
                folder = "../output/plot/",
                type = "ggplot")
}

