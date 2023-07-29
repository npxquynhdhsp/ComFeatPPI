# setwd("Celeg")
# getwd()

path1 <- "Hpylo_PPI.csv"
path2 <- "_Hpylo.csv"

set <- list(set_1 <- c("A", "G", "V"),
            set_2 <- c("I", "L", "F", "P"),
            set_3 <- c("Y", "M", "T", "U", "S"),
            set_4 <- c("H", "N", "Q", "W"),
            set_5 <- c("R", "K"),
            set_6 <- c("D", "E"),
            set_7 <- c("C"))

amino_class <- cbind(unlist(set),
                     c(rep(1, 3), rep(2, 4), rep(3, 5), rep(4, 4), rep(5, 2), rep(6, 2), rep(7, 1)))
amino_class
permute <- NULL; for (i in 1:7) { for (j in 1:7) { for (k in 1:7) { permute <- c(permute, as.numeric(paste0(i, j, k, collapse = ""))) } } }
permute
value <- c("a", "b")

for (chain in 1:2) {
  Matine <- read.csv(path1)
  feature <- NULL
  for (k in 1:length(Matine[, chain])) {
    protein <- strsplit(as.character(Matine[, chain][k]), "")[[1]]
    feat <- rep(0, length(permute))
    for (i in 1:(length(protein) - 2)) {
      A <- as.numeric(paste0(amino_class[which(amino_class[, 1] == protein[i]), 2],
                             amino_class[which(amino_class[, 1] == protein[i + 1]), 2],
                             amino_class[which(amino_class[, 1] == protein[i + 2]), 2], collapse = ""))
      feat[which(permute == A)] <- feat[which(permute == A)] + 1
    }
    feature <- rbind(feature, feat)
  }

  write.csv(rbind(feature),
            paste0("CT_", value[chain], path2, collapse = ""),
            row.names = F)
}

getwd()
path1
path2
