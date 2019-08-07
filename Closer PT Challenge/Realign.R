j<-1
for(i in 1: nrow(corpus)){
  if(corpus$issue[i] %in% lookup){
    corpus$issue[i]<-"other"
  }
}
j