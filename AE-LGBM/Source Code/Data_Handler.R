# setwd("Ecoli")

path1 <- "CT_a_Hpylo.csv"
path2 <- "CT_b_Hpylo.csv"
path3 <- "MMI_a.csv"
path4 <- "MMI_b.csv"

CT_deep_a <- read.csv(path1)
CT_deep_b <- read.csv(path2)
MMI_a <- read.csv(path3)
MMI_b <- read.csv(path4)

total_features_a <- cbind(CT_deep_a, MMI_a)
total_features_b <- cbind(CT_deep_b, MMI_b)

write.csv(total_features_a, "total_features_a.csv", row.names = F)
write.csv(total_features_b, "total_features_b.csv", row.names = F)

getwd()
path1
path2
path3
path4