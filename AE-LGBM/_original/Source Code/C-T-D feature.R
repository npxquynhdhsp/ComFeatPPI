#setwd("C:/Protein")
pair.p<-read.csv("Positive Human PPI.csv", stringsAsFactors = F)
pair.n<-read.csv("Negative Huamn PPI.csv", stringsAsFactors = F)

MMI_class<-list(Hyd_Polar_1<- list(c("R","K","E","D","Q","N"), c("G","A","S","T","U","P","H","Y"), c("C","L","V","I","M","F","W"))
                ,Hyd_Polar_2<- list(c("Q","S","T","U","N","G","D","E"), c("R","A","H","C","K","M","V"), c("L","Y","P","F","I","W"))
                ,Hyd_Polar_3<- list(c("Q","N","G","S","W","T","U","D","E","R","A"), c("H","M","C","K","V"), c("L","P","F","Y","I"))
                ,Hyd_Polar_4<- list(c("K","P","D","E","S","N","Q","T","U"), c("G","R","H","A"), c("Y","M","F","W","L","C","V","I"))
                ,Hyd_Polar_5<- list(c("K","D","E","Q","P","S","R","N","T","U","G"), c("A","H","Y","M","L","V"), c("F","I","W","C"))
                ,Hyd_Polar_6<- list(c("R","D","K","E","N","Q","H","Y","P"), c("S","G","T","U","A","W"), c("C","V","L","I","M","F"))
                ,Hyd_Polar_7<- list(c("K","E","R","S","Q","D"), c("N","T","U","P","G"), c("A","Y","H","W","V","M","F","L","I","C"))
                ,vanderWaals<- list(c("G","A","S","T","U","P","D"), c("N","V","E","Q","I","L","C"), c("M","H","K","F","R","Y","W"))
                ,Polarity_va<- list(c("L","I","F","W","C","M","V","Y"), c("P","A","T","U","G","S"), c("H","Q","R","K","N","E","D"))
                ,Polarizabil<- list(c("G","A","S","D","T","U"), c("C","P","N","V","E","Q","I","L"), c("K","M","H","F","R","Y","W"))
                ,Charge_sets<- list(c("K","R"), c("A","N","C","Q","G","H","I","L","M","F","P","S","T","U","W","Y","V"), c("D","E"))
                ,struc_Helix<- list(c("E","A","L","M","Q","K","R","H"), c("V","I","Y","C","W","F","T","U"), c("G","N","P","S","D"))
                ,Solv_Buried<- list(c("A","L","F","C","G","I","V","W"), c("R","K","Q","E","N","D"), c("M","P","S","T","U","H","Y")))

MMI_val<-NULL
for(i in 1:length(MMI_class)){
  MMI_val<-rbind(MMI_val,c(rep(1, lengths(MMI_class[[i]][1])),
                           rep(2, lengths(MMI_class[[i]][2])),
                           rep(3, lengths(MMI_class[[i]][3]))))
}

MMI.a.p<-NULL
for(x in 1:nrow(pair.p)){
  protein<-strsplit(pair.p$chain[x], "")[[1]]
  composit<-NULL;descript<-NULL;distributor<-NULL
  for(k in 1:nrow(MMI_val)){
    prot<-NULL
    for(i in 1:length(protein)){
      prot[i]<-MMI_val[1,which(protein[i]==unlist(MMI_class[[k]]))]
    }
    mat<-matrix(prot, nrow=(length(prot)+1), ncol=2, byrow = F)
    mat_val<-NULL
    for(ii in 1:(nrow(mat)-2)){
      mat_val[ii]<-paste0(mat[ii,1], mat[ii,2], collapse = "")
    }
    composit[(k-1)*3+1]<-length(which(prot==1))/length(protein)
    composit[(k-1)*3+2]<-length(which(prot==2))/length(protein)
    composit[(k-1)*3+3]<-length(which(prot==3))/length(protein)
    descript[(k-1)*3+1]<-length(which(mat_val=="12"|mat_val=="21"))/length(mat_val)
    descript[(k-1)*3+2]<-length(which(mat_val=="23"|mat_val=="32"))/length(mat_val)
    descript[(k-1)*3+3]<-length(which(mat_val=="31"|mat_val=="13"))/length(mat_val)
    
    if(length(which(prot==1)) == 0){distributor[(k-1)*15+1]<-0;distributor[(k-1)*15+2]<-0;
    distributor[(k-1)*15+3]<-0;distributor[(k-1)*15+4]<-0;distributor[(k-1)*15+5]<-0}
    else if(length(which(prot==1)) <= 2){distributor[(k-1)*15+2]<-0;
    distributor[(k-1)*15+3]<-0;distributor[(k-1)*15+4]<-0;distributor[(k-1)*15+5]<-0}
    else{
      distributor[(k-1)*15+1]<-which(prot==1)[1]/length(prot)
      distributor[(k-1)*15+2]<-which(prot==1)[ceiling(length(which(prot == 1))*0.25)]/length(prot)
      distributor[(k-1)*15+3]<-which(prot==1)[floor(length(which(prot == 1))*0.50)]/length(prot)
      distributor[(k-1)*15+4]<-which(prot==1)[floor(length(which(prot == 1))*0.75)]/length(prot)
      distributor[(k-1)*15+5]<-which(prot==1)[floor(length(which(prot == 1))*1.00)]/length(prot)}
    if(length(which(prot==2)) == 0){distributor[(k-1)*15+6]<-0;distributor[(k-1)*15+7]<-0;
    distributor[(k-1)*15+8]<-0;distributor[(k-1)*15+9]<-0;distributor[(k-1)*15+10]<-0}
    else if(length(which(prot==2)) <= 2){distributor[(k-1)*15+7]<-0;
    distributor[(k-1)*15+8]<-0;distributor[(k-1)*15+9]<-0;distributor[(k-1)*15+10]<-0}
    else{
      distributor[(k-1)*15+6]<-which(prot==2)[1]/length(prot)
      distributor[(k-1)*15+7]<-which(prot==2)[ceiling(length(which(prot == 2))*0.25)]/length(prot)
      distributor[(k-1)*15+8]<-which(prot==2)[floor(length(which(prot == 2))*0.50)]/length(prot)
      distributor[(k-1)*15+9]<-which(prot==2)[floor(length(which(prot == 2))*0.75)]/length(prot)
      distributor[(k-1)*15+10]<-which(prot==2)[floor(length(which(prot == 2))*1.00)]/length(prot)}
    if(length(which(prot==3)) == 0){distributor[(k-1)*15+11]<-0;distributor[(k-1)*15+12]<-0;
    distributor[(k-1)*15+13]<-0;distributor[(k-1)*15+14]<-0;distributor[(k-1)*15+15]<-0}
    else if(length(which(prot==3)) <= 2){distributor[(k-1)*15+12]<-0;
    distributor[(k-1)*15+13]<-0;distributor[(k-1)*15+14]<-0;distributor[(k-1)*15+15]<-0}
    else{
      distributor[(k-1)*15+11]<-which(prot==3)[1]/length(prot)
      distributor[(k-1)*15+12]<-which(prot==3)[ceiling(length(which(prot == 3))*0.25)]/length(prot)
      distributor[(k-1)*15+13]<-which(prot==3)[floor(length(which(prot == 3))*0.50)]/length(prot)
      distributor[(k-1)*15+14]<-which(prot==3)[floor(length(which(prot == 3))*0.75)]/length(prot)
      distributor[(k-1)*15+15]<-which(prot==3)[floor(length(which(prot == 3))*1.00)]/length(prot)}
  }
  MMI.a.p<-rbind(MMI.a.p, c(composit,descript,distributor))
}

MMI.a.n<-NULL
for(x in 1:nrow(pair.n)){
  protein<-strsplit(pair.n$chain[x], "")[[1]]
  composit<-NULL;descript<-NULL;distributor<-NULL
  for(k in 1:nrow(MMI_val)){
    prot<-NULL
    for(i in 1:length(protein)){
      prot[i]<-MMI_val[1,which(protein[i]==unlist(MMI_class[[k]]))]
    }
    mat<-matrix(prot, nrow=(length(prot)+1), ncol=2, byrow = F)
    mat_val<-NULL
    for(ii in 1:(nrow(mat)-2)){
      mat_val[ii]<-paste0(mat[ii,1], mat[ii,2], collapse = "")
    }
    composit[(k-1)*3+1]<-length(which(prot==1))/length(protein)
    composit[(k-1)*3+2]<-length(which(prot==2))/length(protein)
    composit[(k-1)*3+3]<-length(which(prot==3))/length(protein)
    descript[(k-1)*3+1]<-length(which(mat_val=="12"|mat_val=="21"))/length(mat_val)
    descript[(k-1)*3+2]<-length(which(mat_val=="23"|mat_val=="32"))/length(mat_val)
    descript[(k-1)*3+3]<-length(which(mat_val=="31"|mat_val=="13"))/length(mat_val)
    
    if(length(which(prot==1)) == 0){distributor[(k-1)*15+1]<-0;distributor[(k-1)*15+2]<-0;
    distributor[(k-1)*15+3]<-0;distributor[(k-1)*15+4]<-0;distributor[(k-1)*15+5]<-0}
    else if(length(which(prot==1)) <= 2){distributor[(k-1)*15+2]<-0;
    distributor[(k-1)*15+3]<-0;distributor[(k-1)*15+4]<-0;distributor[(k-1)*15+5]<-0}
    else{
      distributor[(k-1)*15+1]<-which(prot==1)[1]/length(prot)
      distributor[(k-1)*15+2]<-which(prot==1)[ceiling(length(which(prot == 1))*0.25)]/length(prot)
      distributor[(k-1)*15+3]<-which(prot==1)[floor(length(which(prot == 1))*0.50)]/length(prot)
      distributor[(k-1)*15+4]<-which(prot==1)[floor(length(which(prot == 1))*0.75)]/length(prot)
      distributor[(k-1)*15+5]<-which(prot==1)[floor(length(which(prot == 1))*1.00)]/length(prot)}
    if(length(which(prot==2)) == 0){distributor[(k-1)*15+6]<-0;distributor[(k-1)*15+7]<-0;
    distributor[(k-1)*15+8]<-0;distributor[(k-1)*15+9]<-0;distributor[(k-1)*15+10]<-0}
    else if(length(which(prot==2)) <= 2){distributor[(k-1)*15+7]<-0;
    distributor[(k-1)*15+8]<-0;distributor[(k-1)*15+9]<-0;distributor[(k-1)*15+10]<-0}
    else{
      distributor[(k-1)*15+6]<-which(prot==2)[1]/length(prot)
      distributor[(k-1)*15+7]<-which(prot==2)[ceiling(length(which(prot == 2))*0.25)]/length(prot)
      distributor[(k-1)*15+8]<-which(prot==2)[floor(length(which(prot == 2))*0.50)]/length(prot)
      distributor[(k-1)*15+9]<-which(prot==2)[floor(length(which(prot == 2))*0.75)]/length(prot)
      distributor[(k-1)*15+10]<-which(prot==2)[floor(length(which(prot == 2))*1.00)]/length(prot)}
    if(length(which(prot==3)) == 0){distributor[(k-1)*15+11]<-0;distributor[(k-1)*15+12]<-0;
    distributor[(k-1)*15+13]<-0;distributor[(k-1)*15+14]<-0;distributor[(k-1)*15+15]<-0}
    else if(length(which(prot==3)) <= 2){distributor[(k-1)*15+12]<-0;
    distributor[(k-1)*15+13]<-0;distributor[(k-1)*15+14]<-0;distributor[(k-1)*15+15]<-0}
    else{
      distributor[(k-1)*15+11]<-which(prot==3)[1]/length(prot)
      distributor[(k-1)*15+12]<-which(prot==3)[ceiling(length(which(prot == 3))*0.25)]/length(prot)
      distributor[(k-1)*15+13]<-which(prot==3)[floor(length(which(prot == 3))*0.50)]/length(prot)
      distributor[(k-1)*15+14]<-which(prot==3)[floor(length(which(prot == 3))*0.75)]/length(prot)
      distributor[(k-1)*15+15]<-which(prot==3)[floor(length(which(prot == 3))*1.00)]/length(prot)}
  }
  MMI.a.n<-rbind(MMI.a.n, c(composit,descript,distributor))
}


MMI.b.p<-NULL
for(x in 1:nrow(pair.p)){
  protein<-strsplit(pair.p$chain.1[x], "")[[1]]
  composit<-NULL;descript<-NULL;distributor<-NULL
  for(k in 1:nrow(MMI_val)){
    prot<-NULL
    for(i in 1:length(protein)){
      prot[i]<-MMI_val[1,which(protein[i]==unlist(MMI_class[[k]]))]
    }
    mat<-matrix(prot, nrow=(length(prot)+1), ncol=2, byrow = F)
    mat_val<-NULL
    for(ii in 1:(nrow(mat)-2)){
      mat_val[ii]<-paste0(mat[ii,1], mat[ii,2], collapse = "")
    }
    composit[(k-1)*3+1]<-length(which(prot==1))/length(protein)
    composit[(k-1)*3+2]<-length(which(prot==2))/length(protein)
    composit[(k-1)*3+3]<-length(which(prot==3))/length(protein)
    descript[(k-1)*3+1]<-length(which(mat_val=="12"|mat_val=="21"))/length(mat_val)
    descript[(k-1)*3+2]<-length(which(mat_val=="23"|mat_val=="32"))/length(mat_val)
    descript[(k-1)*3+3]<-length(which(mat_val=="31"|mat_val=="13"))/length(mat_val)
    
    if(length(which(prot==1)) == 0){distributor[(k-1)*15+1]<-0;distributor[(k-1)*15+2]<-0;
    distributor[(k-1)*15+3]<-0;distributor[(k-1)*15+4]<-0;distributor[(k-1)*15+5]<-0}
    else if(length(which(prot==1)) <= 2){distributor[(k-1)*15+2]<-0;
    distributor[(k-1)*15+3]<-0;distributor[(k-1)*15+4]<-0;distributor[(k-1)*15+5]<-0}
    else{
      distributor[(k-1)*15+1]<-which(prot==1)[1]/length(prot)
      distributor[(k-1)*15+2]<-which(prot==1)[ceiling(length(which(prot == 1))*0.25)]/length(prot)
      distributor[(k-1)*15+3]<-which(prot==1)[floor(length(which(prot == 1))*0.50)]/length(prot)
      distributor[(k-1)*15+4]<-which(prot==1)[floor(length(which(prot == 1))*0.75)]/length(prot)
      distributor[(k-1)*15+5]<-which(prot==1)[floor(length(which(prot == 1))*1.00)]/length(prot)}
    if(length(which(prot==2)) == 0){distributor[(k-1)*15+6]<-0;distributor[(k-1)*15+7]<-0;
    distributor[(k-1)*15+8]<-0;distributor[(k-1)*15+9]<-0;distributor[(k-1)*15+10]<-0}
    else if(length(which(prot==2)) <= 2){distributor[(k-1)*15+7]<-0;
    distributor[(k-1)*15+8]<-0;distributor[(k-1)*15+9]<-0;distributor[(k-1)*15+10]<-0}
    else{
      distributor[(k-1)*15+6]<-which(prot==2)[1]/length(prot)
      distributor[(k-1)*15+7]<-which(prot==2)[ceiling(length(which(prot == 2))*0.25)]/length(prot)
      distributor[(k-1)*15+8]<-which(prot==2)[floor(length(which(prot == 2))*0.50)]/length(prot)
      distributor[(k-1)*15+9]<-which(prot==2)[floor(length(which(prot == 2))*0.75)]/length(prot)
      distributor[(k-1)*15+10]<-which(prot==2)[floor(length(which(prot == 2))*1.00)]/length(prot)}
    if(length(which(prot==3)) == 0){distributor[(k-1)*15+11]<-0;distributor[(k-1)*15+12]<-0;
    distributor[(k-1)*15+13]<-0;distributor[(k-1)*15+14]<-0;distributor[(k-1)*15+15]<-0}
    else if(length(which(prot==3)) <= 2){distributor[(k-1)*15+12]<-0;
    distributor[(k-1)*15+13]<-0;distributor[(k-1)*15+14]<-0;distributor[(k-1)*15+15]<-0}
    else{
      distributor[(k-1)*15+11]<-which(prot==3)[1]/length(prot)
      distributor[(k-1)*15+12]<-which(prot==3)[ceiling(length(which(prot == 3))*0.25)]/length(prot)
      distributor[(k-1)*15+13]<-which(prot==3)[floor(length(which(prot == 3))*0.50)]/length(prot)
      distributor[(k-1)*15+14]<-which(prot==3)[floor(length(which(prot == 3))*0.75)]/length(prot)
      distributor[(k-1)*15+15]<-which(prot==3)[floor(length(which(prot == 3))*1.00)]/length(prot)}
  }
  MMI.b.p<-rbind(MMI.b.p, c(composit,descript,distributor))
}

MMI.b.n<-NULL
for(x in 1:nrow(pair.n)){
  protein<-strsplit(pair.n$chain.1[x], "")[[1]]
  composit<-NULL;descript<-NULL;distributor<-NULL
  for(k in 1:nrow(MMI_val)){
    prot<-NULL
    for(i in 1:length(protein)){
      prot[i]<-MMI_val[1,which(protein[i]==unlist(MMI_class[[k]]))]
    }
    mat<-matrix(prot, nrow=(length(prot)+1), ncol=2, byrow = F)
    mat_val<-NULL
    for(ii in 1:(nrow(mat)-2)){
      mat_val[ii]<-paste0(mat[ii,1], mat[ii,2], collapse = "")
    }
    composit[(k-1)*3+1]<-length(which(prot==1))/length(protein)
    composit[(k-1)*3+2]<-length(which(prot==2))/length(protein)
    composit[(k-1)*3+3]<-length(which(prot==3))/length(protein)
    descript[(k-1)*3+1]<-length(which(mat_val=="12"|mat_val=="21"))/length(mat_val)
    descript[(k-1)*3+2]<-length(which(mat_val=="23"|mat_val=="32"))/length(mat_val)
    descript[(k-1)*3+3]<-length(which(mat_val=="31"|mat_val=="13"))/length(mat_val)
    
    if(length(which(prot==1)) == 0){distributor[(k-1)*15+1]<-0;distributor[(k-1)*15+2]<-0;
    distributor[(k-1)*15+3]<-0;distributor[(k-1)*15+4]<-0;distributor[(k-1)*15+5]<-0}
    else if(length(which(prot==1)) <= 2){distributor[(k-1)*15+2]<-0;
    distributor[(k-1)*15+3]<-0;distributor[(k-1)*15+4]<-0;distributor[(k-1)*15+5]<-0}
    else{
      distributor[(k-1)*15+1]<-which(prot==1)[1]/length(prot)
      distributor[(k-1)*15+2]<-which(prot==1)[ceiling(length(which(prot == 1))*0.25)]/length(prot)
      distributor[(k-1)*15+3]<-which(prot==1)[floor(length(which(prot == 1))*0.50)]/length(prot)
      distributor[(k-1)*15+4]<-which(prot==1)[floor(length(which(prot == 1))*0.75)]/length(prot)
      distributor[(k-1)*15+5]<-which(prot==1)[floor(length(which(prot == 1))*1.00)]/length(prot)}
    if(length(which(prot==2)) == 0){distributor[(k-1)*15+6]<-0;distributor[(k-1)*15+7]<-0;
    distributor[(k-1)*15+8]<-0;distributor[(k-1)*15+9]<-0;distributor[(k-1)*15+10]<-0}
    else if(length(which(prot==2)) <= 2){distributor[(k-1)*15+7]<-0;
    distributor[(k-1)*15+8]<-0;distributor[(k-1)*15+9]<-0;distributor[(k-1)*15+10]<-0}
    else{
      distributor[(k-1)*15+6]<-which(prot==2)[1]/length(prot)
      distributor[(k-1)*15+7]<-which(prot==2)[ceiling(length(which(prot == 2))*0.25)]/length(prot)
      distributor[(k-1)*15+8]<-which(prot==2)[floor(length(which(prot == 2))*0.50)]/length(prot)
      distributor[(k-1)*15+9]<-which(prot==2)[floor(length(which(prot == 2))*0.75)]/length(prot)
      distributor[(k-1)*15+10]<-which(prot==2)[floor(length(which(prot == 2))*1.00)]/length(prot)}
    if(length(which(prot==3)) == 0){distributor[(k-1)*15+11]<-0;distributor[(k-1)*15+12]<-0;
    distributor[(k-1)*15+13]<-0;distributor[(k-1)*15+14]<-0;distributor[(k-1)*15+15]<-0}
    else if(length(which(prot==3)) <= 2){distributor[(k-1)*15+12]<-0;
    distributor[(k-1)*15+13]<-0;distributor[(k-1)*15+14]<-0;distributor[(k-1)*15+15]<-0}
    else{
      distributor[(k-1)*15+11]<-which(prot==3)[1]/length(prot)
      distributor[(k-1)*15+12]<-which(prot==3)[ceiling(length(which(prot == 3))*0.25)]/length(prot)
      distributor[(k-1)*15+13]<-which(prot==3)[floor(length(which(prot == 3))*0.50)]/length(prot)
      distributor[(k-1)*15+14]<-which(prot==3)[floor(length(which(prot == 3))*0.75)]/length(prot)
      distributor[(k-1)*15+15]<-which(prot==3)[floor(length(which(prot == 3))*1.00)]/length(prot)}
  }
  MMI.b.n<-rbind(MMI.b.n, c(composit,descript,distributor))
}
setwd("C:/Protein/Final Revision")
write.csv(rbind(MMI.a.p, MMI.a.n), "MMI_a.csv", row.names = F)
write.csv(rbind(MMI.b.p, MMI.b.n), "MMI_b.csv", row.names = F)


