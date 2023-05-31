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

folder = paste(here(), "/parallel/script", sep = "")
setwd(folder)

cpu = read.csv("../output/performance_cpu.txt", sep = "")
gpu = read.csv("../output/performance_gpu.txt", sep = "")

performance = data.frame(
  time_cpu = cpu$time,
  time_gpu = gpu$time,
  dataset = cpu$dataset
)

mean_performance = aggregate(. ~ dataset, data = performance, FUN = mean)

temp1 = data.frame(
  time = mean_performance$time_cpu,
  dataset = mean_performance$dataset,
  type = rep("cpu", length(mean_performance$dataset))
)

temp2 = data.frame(
  time = mean_performance$time_gpu,
  dataset = mean_performance$dataset,
  type = rep("gpu", length(mean_performance$dataset))
)

mean_performance = rbind(temp1, temp2)
mean_performance$dataset = factor(mean_performance$dataset)
mean_performance$type = factor(mean_performance$type)
mean_performance$time = round(mean_performance$time, 3)

plot = ggplot(mean_performance, aes(x = dataset, y = time, fill = type)) +
  geom_bar(stat = "identity",
           width = .8,
           position = "dodge") +
  labs(x = "", y = "Training time per epoch [ms]", fill = "Architecture") +
  geom_text(aes(label = time),
            position = position_dodge(width = 0.8),
            vjust = -0.5) +
  theme_minimal() +
  ggtitle("Performance comparison") +
  theme(
    axis.text = element_text(size = rel(1.4)),
    axis.title = element_text(size = rel(1.6)),
    plot.title = element_text(
      face = "bold",
      size = rel(2.2),
      hjust = 0.5
    ),
    legend.text = element_text(size = rel(1.4)),
    legend.title = element_text(size = rel(1.4))
  )

save_svg_plot(plot,
              name = "performance_plot",
              folder = "../output/plot/",
              type = "ggplot")