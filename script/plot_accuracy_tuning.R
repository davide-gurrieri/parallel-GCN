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
#setwd(folder)
names = c("citeseer", "cora", "pubmed")
plots = vector("list", length = length(names))
data = vector("list", length = length(names))
data_no_feature = vector("list", length = length(names))
data2 = vector("list", length = length(names))

i = 1
for (name in names)
{
  folder_name_no_feature = paste("../output/tuning_accuracy_no_feature_", name, ".txt", sep = "")
  data_no_feature[[i]] = read.csv(folder_name_no_feature, sep = "")
  folder_name = paste("../output/tuning_accuracy_", name, ".txt", sep = "")
  data[[i]] = read.csv(folder_name, sep = "")
  folder_name2 = paste("../output/tuning_accuracy_second_", name, ".txt", sep = "")
  data2[[i]] = read.csv(folder_name2, sep = "")
  i = i+1
}
