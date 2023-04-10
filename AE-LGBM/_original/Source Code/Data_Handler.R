#setwd("C:/Protein/")

MMI_a<-read.csv("MMI_a.csv")
MMI_b<-read.csv("MMI_b.csv")
CT_deep_a<-read.csv("CT_Deep_a_human.csv")
CT_deep_b<-read.csv("CT_Deep_b_huamn.csv")

total_features_a<-cbind(CT_deep_a,MMI_a)
total_features_b<-cbind(CT_deep_b,MMI_b)

write.csv(total_features_a, "total_features_a.csv", row.names = F)
write.csv(total_features_b, "total_features_b.csv", row.names = F)