# ###############################################################
# modifier: thnhan
# fine
# ###############################################################

library(keras)
library(tensorflow)
library(reticulate)

# library(e1071)

setwd("cua_thnhan")
label <- read.csv("label.csv")

feat_a <- as.matrix(read.csv("total_features_a.csv"))
feat_b <- as.matrix(read.csv("total_features_b.csv"))

k <- 208  # Number of nodes in hidden layer

####################Chain A

#Initialize the Autoencoder for chain A
model_AE1 <- keras_model_sequential()
model_AE1 %>%
  layer_dense(units = k, activation = 'sigmoid', input_shape = ncol(feat_a)) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = ncol(feat_a), activation = 'sigmoid', input_shape = k)
summary(model_AE1)

# Constructing Autoencoder
model_AE1 %>%
  compile(loss = 'mean_squared_error',
          optimize = 'nadam',
          metrics = "accuracy")

#Fitting the Autoencoder
history <- model_AE1 %>%
  fit(feat_a, feat_a, epoch = 200,
      batch_size = 50, validation_split = 0.2)

#Extracting the hidden/condensed layer for chain A
a_100 <- keras_model(inputs = model_AE1$input,
                     outputs = get_layer(model_AE1, index = 1)$output)
red.feat.a <- predict(a_100, feat_a)

#################### Chain B

#Initialize the Autoencoder for chain B
model_AE2 <- keras_model_sequential()
model_AE2 %>%
  layer_dense(units = k, activation = 'sigmoid', input_shape = ncol(feat_b)) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = ncol(feat_b), activation = 'sigmoid', input_shape = k)
summary(model_AE2)

# Constructing Autoencoder
model_AE2 %>%
  compile(loss = 'mean_squared_error',
          optimize = 'nadam',
          metrics = "accuracy")

#Fitting the Autoencoder
history <- model_AE2 %>%
  fit(feat_b, feat_b, epoch = 200,
      batch_size = 50, validation_split = 0.2)

#Extracting the hidden/condensed layer for chain B
b_100 <- keras_model(inputs = model_AE2$input,
                     outputs = get_layer(model_AE2, index = 1)$output)
red.feat.b <- predict(b_100, feat_b)

####################Save Model
model_AE1 %>% save_model_hdf5(paste0("AE_Human_a_.h5", collapse = ""))
model_AE2 %>% save_model_hdf5(paste0("AE_Human_b_.h5", collapse = ""))
##############

gbm_data <- cbind(red.feat.a, red.feat.b)
gbm_data <- as.data.frame(cbind(gbm_data, label))
write.csv(gbm_data[sample(seq_len(nrow(gbm_data))),], "gbm_Human.csv", row.names = F) # shuffle and save
