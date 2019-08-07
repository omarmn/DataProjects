#Initialise
#TM package require for text manipulation and generating Document Term Matrix
require(tm)

#dplyr for manipulating data frames
require(dplyr)

#caret for partitioning data into test and train sets
require(caret)

#e1071 for using SVM and Naivebaye's algorithms used for supervised learning
require(e1071)

#topicmodels contains LDA algorithm for unsupervised learning
require(topicmodels)

#Used to extract data from LDA model
require(tidytext)

#Used to plot topics from LDA
require(ggrepel)

#Set seed manually
set.seed(475)

#-----------------
#Read data
#-----------------
data<-read.csv("Documents/GitHub/Closer PT Challenge/complains_data.csv")
#Extract the aforementioned columns and create a new data frame
corpus<-data.frame(complaint=data$Consumer.complaint.narrative, issue_tv=data$Issue)

#Coerce columns into characers
corpus$complaint<-as.character(corpus$complaint)
corpus$issue_tv<-as.character(corpus$issue_tv)

#-----------------
#Balance data
#-----------------
#Count complainst by issue
countbyissue<-aggregate(corpus$issue_tv, by=list(corpus$issue_tv), length)

#Give data frame meaniful names
colnames(countbyissue)<-c("issue","count")

#Order data in descending order
countbyissue<-countbyissue[order(-countbyissue$count),]

#Create a bar plot
barplot(countbyissue$count, main = "Count of Each Issue", col="blue")


sum(countbyissue$count[1:100])/sum(countbyissue$count)

countbyissue$count[100]

#Create a lookup table for issues are below the 100th issue
lookup<-as.character(countbyissue$issue[101:161])

#Loop through those issues and replace with "other" as issue
for(i in 1: nrow(corpus)){
  if(corpus$issue_tv[i] %in% lookup){
    corpus$issue_tv[i]<-"other"
  }
}

#Sample 151 complaints from each issue
corpus <- corpus %>% group_by(issue_tv) %>% sample_n(151)

#Convertfrom dplyr to data frame
corpus <- ungroup(corpus) %>% as.data.frame()

#Count complainst by issue
countbyissue<-aggregate(corpus$issue_tv, by=list(corpus$issue_tv), length)

#Give data frame meaniful names
colnames(countbyissue)<-c("issue","count")

#Order data in descending order
countbyissue<-countbyissue[order(-countbyissue$count),]

#Create a bar plot
barplot(countbyissue$count, main = "Count of Each Issue(after balancing)", col="blue")

#-----------------
#Clean text: stopwords, tolower, punctuation, XXXX, \n, stemming
#-----------------

#Convert letters to lower case
corpus$complaint<-tolower(corpus$complaint)

#Remove Numbers
corpus$complaint<-removeNumbers(corpus$complaint)

#Remove Punctuation
corpus$complaint<-removePunctuation(corpus$complaint)

#Create a function that will take a pattern and remove it
findremove<- function (pattern, object) { gsub(pattern, "", object)}

#Remove xxxx
corpus$complaint<-findremove("xxxx", corpus$complaint)

#Remove xx
corpus$complaint<-findremove("xx", corpus$complaint)

#Remove \n
corpus$complaint<-findremove("\n", corpus$complaint)

#Remove \t
corpus$complaint<-findremove("\t", corpus$complaint)

#Strip extra white spaces
corpus$complaint<-stripWhitespace(corpus$complaint)

#Convert complaints into a Corpus using tm package
complaintCorpus<-Corpus(VectorSource(corpus$complaint))

#Remove Stopwords
complaintCorpus<-tm_map(complaintCorpus, removeWords, stopwords())

#Stemming words
#corpus$complaint<-stemDocument(corpus$complaint)


#-----------------
#Feature creation: DTM
#-----------------
#Create DTM using TM
dtm<-DocumentTermMatrix(complaintCorpus)

#View the DTM
inspect(dtm)

#Sum each term across all documents (i.e column sum)
cumulativeAllTerms<-colSums(as.matrix(dtm))

#Sort in descending order
cumulativeAllTerms<-cumulativeAllTerms[order(-cumulativeAllTerms)]

#Show top 100 terms
head(cumulativeAllTerms, 100)

#Create my list of stopwords
otherstopwords<-c("told","called","back","can","will","get","said","never","also",
                  "even","just","know","another","like","want","went","please","take",
                  "however","going","see","got","several","able")

#Remove custom stopwords
complaintCorpus<-tm_map(complaintCorpus, removeWords, otherstopwords)

#Create a new DTM
dtm<-DocumentTermMatrix(complaintCorpus)

#Find terms that appear 10 times or more
freqterms<-findFreqTerms(dtm, lowfreq = 10)

#Limit DTM to contain terms that appear >= 10
dtm<-DocumentTermMatrix(complaintCorpus, list(dictionary=freqterms))
inspect(dtm)

#Sum count of each term across all documents
cumulativeAllTerms<-colSums(as.matrix(dtm))

#Sort in descending order and take top 30 terms
Top30<-head(cumulativeAllTerms[order(-cumulativeAllTerms)], 30)

#Convert to data frame
Top30<-data.frame(term=names(Top30), count=Top30)
Top30<-Top30[order(-Top30$count),]

#Plot
barplot(rev(Top30$count), horiz = T, names.arg = Top30$term, las=2, col="blue", main="Most Frequent 30 Terms")


#-----------------
#Create train and test sets
#-----------------
#Convert issue/target variable to factor, in order to conserve levels in case some categories don't appear in one of the set (highly unlikely since data is balanced)
corpus$issue_tv<-as.factor(corpus$issue_tv)

#Create an index with 75% split based on issue value in raw data
inTrain<-createDataPartition(corpus$issue_tv,p=0.75,list=FALSE)

#Subset raw data with index
train<-corpus[inTrain,]

#Subset raw data with NOT index
test<-corpus[-inTrain,]

#Subset cleaned corpus for training & test sets
corpustrain<-complaintCorpus[inTrain]
corpustest<-complaintCorpus[-inTrain]

#Create DTM based on subsetted cleaned corpus
dtmtrain<-DocumentTermMatrix(corpustrain, list(dictionary=freqterms))
dtmtest<-DocumentTermMatrix(corpustest, list(dictionary=freqterms))

#Function to convert non-zero values to 1
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
}

#Convert non-zero values to 1 in train and test DRM
dtmtrain<- dtmtrain %>% apply(MARGIN=2,convert_counts)
dtmtest<- dtmtest %>% apply(MARGIN=2,convert_counts)

#Convert DTM to data frames
dtmtrain<-as.data.frame(dtmtrain)
dtmtest<-as.data.frame(dtmtest)

#Bind target variable to test and train DTMs
dtmtrain<-cbind(issue_tv=train$issue_tv,dtmtrain)
dtmtest<-cbind(issue_tv=test$issue_tv,dtmtest)

#-----------------
#Training Models - Supervised
#-----------------
#Train a model based on Naive Bayes using e1017 package
fit_NB<-naiveBayes(dtmtrain, dtmtrain$issue_tv)

#Predict using Naive Bayes model using the test set
pred_NB<-predict(fit_NB, newdata= dtmtest)

#Create a confusion matrix for that model/prediction
conf_NB<-confusionMatrix(pred_NB,dtmtest$issue_tv)

#Extract accuracy
conf_NB$overall["Accuracy"]

#Train a model based on SVM in the e1017 package
fit_SVM<-svm(issue_tv ~ ., data = dtmtrain, scale=FALSE)

#Predict using Naive Bayes model using the test set
pred_SVM<-predict(fit_SVM, newdata= dtmtest)

#Create a confusion matrix for that model/prediction
conf_SVM<-confusionMatrix(pred_SVM,dtmtest$issue_tv)

#Extract Accuracy
conf_SVM$overall["Accuracy"]

#-----------------
#Training Models - unsupervised
#-----------------
#10 topics/categories
k<-10

#Run LDA algorithm
lda<-LDA(dtm, k=10, method = "GIBBS")

#-----------------
#The below code is borrowed from this site: https://www.datacamp.com/community/tutorials/ML-NLP-lyric-analysis
#-------------------
theme_lyrics <- function(aticks = element_blank(),
                         pgminor = element_blank(),
                         lt = element_blank(),
                         lp = "none")
{
  theme(plot.title = element_text(hjust = 0.5), #center the title
        axis.ticks = aticks, #set axis ticks to on or off
        panel.grid.minor = pgminor, #turn on or off the minor grid lines
        legend.title = lt, #turn on or off the legend title
        legend.position = lp) #turn on or off the legend
}

word_chart <- function(data, input, title) {
  data %>%
    #set y = 1 to just plot one variable and use word as the label
    ggplot(aes(as.factor(row), 1, label = input, fill = factor(topic) )) +
    #you want the words, not the points
    geom_point(color = "transparent") +
    #make sure the labels don't overlap
    geom_label_repel(nudge_x = .2,  
                     direction = "y",
                     box.padding = 0.1,
                     segment.color = "transparent",
                     size = 3) +
    facet_grid(~topic) +
    theme_lyrics() +
    theme(axis.text.y = element_blank(), axis.text.x = element_blank(),
          #axis.title.x = element_text(size = 9),
          panel.grid = element_blank(), panel.background = element_blank(),
          panel.border = element_rect("lightgray", fill = NA),
          strip.text.x = element_text(size = 9)) +
    labs(x = NULL, y = NULL, title = title) +
    #xlab(NULL) + ylab(NULL) +
    #ggtitle(title) +
    coord_flip()
}


num_words <- 10 #number of words to visualize

#create function that accepts the lda model and num word to display
top_terms_per_topic <- function(lda_model, num_words) {
  
  #tidy LDA object to get word, topic, and probability (beta)
  topics_tidy <- tidy(lda_model, matrix = "beta")
  
  
  top_terms <- topics_tidy %>%
    group_by(topic) %>%
    arrange(topic, desc(beta)) %>%
    #get the top num_words PER topic
    slice(seq_len(num_words)) %>%
    arrange(topic, beta) %>%
    #row is required for the word_chart() function
    mutate(row = row_number()) %>%
    ungroup() %>%
    #add the word Topic to the topic labels
    mutate(topic = paste("Topic", topic, sep = " "))
  #create a title to pass to word_chart
  title <- paste("LDA Top Terms for", k, "Topics")
  #call the word_chart function you built in prep work
  word_chart(top_terms, top_terms$term, title)
}
#call the function you just built!

top_terms_per_topic(lda, num_words)