
# Generate a g-factor from the behavioral data
# Code adopted from Dubois et al. 2018: http://dx.doi.org/10.1098/rstb.2017.0284
# Original code for bi-factor analysis: https://github.com/adolphslab/HCP_MRI-behavior
# Generate a fluid intelligence factor (latent score)
# Generate a crystallized intelligence factor (sum score)

library(tidyverse)
library(psych)
library(lavaan)
library(Hmisc)
library(corrplot)
library(semPlot)
library(colorRamps)

###Helpers functions
#Compute Comparative Fit Index for a factor analysis 
CFI <-function(x){
  return((1-((x$STATISTIC-x$dof))/(x$null.chisq-x$null.dof)))
}
#Compute Comparative Fit Index for a bifactor analysis 
CFI_biv <-function(x){
  return((1-((x$stats$STATISTIC-x$stats$dof))/(x$stats$null.chisq-x$stats$null.dof)))
}
#Compute implied matrix for a factor analysis
impliedMatrix<-function(x){
  if (dim(x$loadings)[2]==1) {
    imp <- x$loadings %*% t(x$loadings)
  } 
  else {imp <- x$loadings %*% x$Phi %*% t(x$loadings)}
  diag(imp)<- diag(imp) + x$uniquenesses
  return(imp)
}
#Compute implied matrix for a bifactor analysis
impliedMatrix_biv<-function(x){
  Gloadings     <- x$schmid$sl[,1]
  Floadings     <- x$schmid$sl[,2:(ncol(x$schmid$sl)-3)]
  uniquenesses  <- x$schmid$sl[,ncol(x$schmid$sl)-1]
  imp           <- Gloadings %*% t(Gloadings) + Floadings %*% t(Floadings)
  diag(imp)     <- diag(imp) + uniquenesses
  return(imp)
}

#Matrix to store fit measures of factor analyses
fitInds <- matrix(nrow = 2, ncol = 4)
rownames(fitInds) <- c('g_model','gf_model')
colnames(fitInds) <- c('CFI','RMSEA','SRMR','BIC')






#Read cognitive scores
data <- read.csv(file = 'CogScores_12_1186subjects_index.csv') #Read cognitive scores
cogdf <- select(data, c(2:13)) 
subjects <- select(data, c(1))





###Factor of general intelligence

fm     <- "minres"       #Use maximum likelihood estimator
rotate <- "oblimin"   #Use oblimin factor rotation

#Parallel analysis to compute numbers of factors for bi-factor model
fa.parallel(cogdf,n.obs=NULL,fm="minres",fa="both",nfactors=1, 
            main="Parallel Analysis Scree Plots",
            n.iter=20,error.bars=FALSE,se.bars=FALSE,SMC=FALSE,ylabel=NULL,show.legend=TRUE,
            sim=TRUE,quant=.95,cor="cor",use="pairwise",plot=TRUE,correct=.5)


#Bi-factor model
model = 1
g_model      <- omega(cogdf,nfactors=4,fm=fm,key=NULL,flip=FALSE,
                 digits=3,title="Omega",sl=TRUE,labels=NULL, plot=FALSE,
                 n.obs=NA,rotate=rotate,Phi = NULL,option="equal",covar=FALSE)


#Compute fit of g-model
obs       <-  cov(cogdf) #Observed covariance matrices
lobs      <-  obs[!lower.tri(obs)]
imp     <-  impliedMatrix_biv(g_model)
limp    <-  imp[!lower.tri(imp)]
fitInds[model,1] <-  CFI_biv(g_model)
fitInds[model,2] <-  g_model$schmid$RMSEA[1]
fitInds[model,3] <-  sqrt(mean((limp - lobs)^2))
fitInds[model,4] <-  g_model$stats$BIC


#Visualize g-model
#print(g_model)
diagram(g_model,digits=3,cut=0.2)



###Factor of fluid intelligence scores

fluid_scores = cogdf %>% select('PicSeq_Unadj','CardSort_Unadj','Flanker_Unadj','PMAT24_A_CR','ProcSpeed_Unadj','VSPLOT_TC', 'ListSort_Unadj')

#One factor model
model = 2
gf_model   <- fa(fluid_scores,nfactors=1)


#Compute fit of gf-model
obs       <-  cov(fluid_scores) #Observed covariance matrices
lobs      <-  obs[!lower.tri(obs)]
imp    <-  impliedMatrix(gf_model)
limp   <-  imp[!lower.tri(imp)]
fitInds[model,1] <-  CFI(gf_model)
fitInds[model,2] <-  gf_model$RMSEA[1]
fitInds[model,3] <-  sqrt(mean((limp - lobs)^2))
fitInds[model,4] <-  gf_model$BIC

#Visualize fits of g-model and gf-model 
print(fitInds,digits=3)


### Crystallized intelligence as sum score
sum_gc = (cogdf$PicVocab_Unadj + cogdf$ReadEng_Unadj)/2



#Intelligence scores
g <- factor.scores(cogdf,g_model$schmid$sl[,1:5])$scores[,1]
gf <- as.numeric(gf_model$score)
gc <- sum_gc


#Save scores
df_intelligence <- data.frame(subjects, g, gf, gc)
write.csv(df_intelligence, 'intelligence_factors_1186Subjects.csv', row.names = FALSE) 




