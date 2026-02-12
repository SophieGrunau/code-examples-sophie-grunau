# S3_PCA_SVM_method_code.R
# Original Author: Gary O'Sullivan
# Author of adapted code: Sophie Grunau
# Dependencies:
#   - R packages: ggplot2, grid, e1071, RColorBrewer, readr, gridExtra, scales, ggnewscale
#   - Data files: OSullivan_plot_3c_data.txt, new_samples.txt
# Note: Paths for data and new_data must be adapted to your system.

# Purpose: Performs sampling of rock data and trains an SVM model on the dataset..
# This script reproduces key figures (3b and c) from O'Sullivan et al. (2020) .
# Originally adapted to help a phd colleague - George (in Geology) understand the code and
# adapt figures for his own samples.
# comments with "#SOPHIE:" show added additional preprocessing and visualization steps

# ------------ ADAPTED SCRIPT ------------
#In order for this code to work the imported file must be .txt format; have 5 columns arranged in this order -
#Grain, Lithology, LREE, Sr/Y, Age; must contain no spaces within cells; must contain no missing values
#Data available @ https://doi.pangaea.de/10.1594/PANGAEA.906570 published in OCT2019

### FOR GEORGE:
# You need to change line 31 to the directory where you folder is at.
# If you use windows you might have to change "/" to "\" for files
# I named your two sample groups "Group 1" (23WC07ap) and "Group 2" (23WC08ap) in line 39,
# if you change these, it will change in all the plots
# I named the Lithology categories "Groups" (same as in paper) but you can change it in line 40
# I have also added extra comments to make the code more readable

#necessary libraries
library(ggplot2)
library(grid)
library(e1071)
library(RColorBrewer)
library(readr)
library(gridExtra)
library(scales)
library(ggnewscale)

# ------------ READ IN DATA - NEEDS TO BE CHANGED TO YOUR FILES ------------
#change the directory to the location of your .txt file
setwd("~/Documents/Work/Bewerbungen/code-examples-sophie-grunau/reproducing_paper_figure_R/") 
data <- read.table("Data/OSullivan_plot_3c_data.txt", header=TRUE, sep="", dec=".") 
Bedrockdata<-data[1:1106,2:4]
#SOPHIE: REPLACED Testdata WITH new_samples
#endofdata<-nrow(data) #how many rows are there in the data
#Testdata<-data[1107:endofdata, ] #the first test datapoint to the last, using the 'endofdata' integer

new_data <- read.table("Data/new_samples.txt", header=TRUE, sep="", dec=".") #filename, use tab deliminated .txt file
sample_group_name <- c("Group 1", "Group 2")
lithology_clas_name <- 'Groups'
new_data$Sample_Group <- ifelse(grepl("23WC07ap", new_data$Grain), sample_group_name[1], ifelse(grepl("23WC08ap", new_data$Grain), sample_group_name[2], NA))

#SOPHIE: Creates directory for outputs
dir.create("Plots", showWarnings = FALSE)

#The ONLY numbers you need to specify in this code are the following, wherein you need
#to define the age range you would like the unknowns to be displayed over on SVM plot
#these values are defaulted to find the lowest and highest values in your .txt 'age' column
#replace the arguments with integers if you want to set absolute ranges
# e.g. AgerangeMinimum<-min(data$Age[which(data$Age>0)]) to AgerangeMinimum<-1000 for a
# 1 Ga minimum

# SOPHIE: CHANGED DATA TO NEW DATA
AgerangeMinimum<-min(new_data$Age[which(new_data$Age>0)])
AgerangeMaximum<-max(new_data$Age[which(new_data$Age>0)])
  Agemiddle = AgerangeMinimum+((AgerangeMaximum-AgerangeMinimum)/2)


# ------------ PLOT 3b: Sr/Y vsΣLREE(defined here as La-Nd), which produces an almost identical separation of the data ------------
# SOPHIE: REMOVED WARNINGS AND CHANGED LABELS TO VARIABLE, 
  
#Bedrockdataplot<-ggplot(Bedrockdata, aes(x=Bedrockdata$LREE, y=Bedrockdata$Sr.Y)) +
#  geom_point(aes(), size = 1.5) +
#  labs (x = "LREE ppm", y = "Sr/Y",colour = "Groups") +
#  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#        panel.background = element_blank(), axis.line = element_line(colour = "dark grey")) +
#  scale_x_log10() + scale_y_log10()
#Bedrockdataplot <- Bedrockdataplot + geom_point(aes(colour=factor(Lithology)),size=1)
#Bedrockdataplot<- Bedrockdataplot + scale_colour_manual(values=c("#FFF100", "#FFC45A", "#E60000","#00B72D","#5050FF","#D272FF"))
#Allplot<- Bedrockdataplot + geom_point(aes(x=Testdata$LREE, y=Testdata$Sr.Y),pch=17, size=1, data = Testdata)
#Allplot + geom_density_2d(aes(x=Testdata$LREE, y=Testdata$Sr.Y, fill = "UM"), bins = 6, data = Testdata) #delete or skip this line if density plot is overfit
#ALLplot #First plot, knowns in colour, unknowns plotted as triangles and by density, density plot will fail if too few unknows are plot - in which case delete line 35
#We have now created the first biplot, code from here on enacts SVM

Bedrockplot<-ggplot(Bedrockdata, aes(x=LREE, y=Sr.Y)) +
  geom_point(aes(), size = 1.5) +
  #grid
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "dark grey")) +
  #x- and y-label and ticks
  labs (x = "LREE ppm", y = "Sr/Y",colour = lithology_clas_name) +
  scale_x_log10(labels = label_number()) + scale_y_log10(labels = label_number()) + #SOPHIE: changed to display as whole numbers to match paper plot
  #point size and colour
  geom_point(aes(colour=factor(Lithology)),size=1) +
  scale_colour_manual(values=c("#FFF100", "#FFC45A", "#E60000","#00B72D","#5050FF","#D272FF"))
#add test data to plot
Bedrockplot_newData <- Bedrockplot +
  geom_point(aes(x = LREE, y = Sr.Y, shape = "Samples"), size = 1, data = new_data) +
  scale_shape_manual(values = c("Samples" = 17), guide = guide_legend(title = NULL))
# add density plot
Bedrockplot_newData_density <- Bedrockplot_newData + 
  geom_density_2d(aes(x=LREE, y=Sr.Y), bins = 6, data = new_data) #delete or skip this line if density plot is overfit
ggsave("Plots/Plot3b_new_data_density.pdf", plot = Bedrockplot_newData_density, width = 6, height = 7.5) 

# ------------ PREPARE DATA FOR PLOT 3c: Define test data ------------
SVM_Train_dataset <- Bedrockdata[, 3:2] #coordinates of the grains
SVM_Train_dataset$Lithology <- Bedrockdata$Lithology #add the lithology
SVM_Train_dataset$Lithology <- factor(SVM_Train_dataset$Lithology) # SOPHIE: ADD LINE FOR NEXT LINE TO NOT RESULT IN ERROR
SVM_Train_dataset$Lithology<- droplevels(SVM_Train_dataset$Lithology, exclude = if(anyNA(levels(SVM_Train_dataset$Lithology))) NULL else NA) #housekeeping

# SOPHIE: CHANGED TO FOR LOOP AS IT IS LESS FRAGILE IN CASE ROWS ADDED/ REMOVED
#The following makes separate lists of the different bedrock groups
#ALKonly <-SVM_Train_dataset_all[1:134, ]
#HMonly <-SVM_Train_dataset_all[135:294, ]
#IMonly <-SVM_Train_dataset_all[295:671, ]
#LMonly <-SVM_Train_dataset_all[672:907, ]
#Sonly <-SVM_Train_dataset_all[908:994, ]
#UMonly <-SVM_Train_dataset_all[995:1106, ]

#set.seed(64) #seed is set so that the transformation is always the same and thus plots are comparable, the numbers were selected randomly between 1-100
#sampleALK <- sample.int(n = nrow(ALKonly), size = 70, replace = F) #the selection is lottery-style (non-replacive)
#TrainALK <- ALKonly[sampleALK, ] #set that will be used to construct the ALK group in SVM
#TestALK  <- ALKonly[-sampleALK, ] #set that will be used to test SVM
#set.seed(34) #process now repeats for each group
#sampleHM <- sample.int(n = nrow(HMonly), size = 70, replace = F)
#TrainHM <- HMonly[sampleHM, ]
#TestHM  <- HMonly[-sampleHM, ]
#set.seed(2)
#sampleIM <- sample.int(n = nrow(IMonly), size = 70, replace = F)
#TrainIM <- IMonly[sampleIM, ]
#TestIM  <- IMonly[-sampleIM, ]
#set.seed(136)
#sampleLM <- sample.int(n = nrow(LMonly), size = 70, replace = F)
#TrainLM <- LMonly[sampleLM, ]
#TestLM  <- LMonly[-sampleLM, ]
#set.seed(185)
#sampleS <- sample.int(n = nrow(Sonly), size = 70, replace = F)
#TrainS <- Sonly[sampleS, ]
#TestS  <- Sonly[-sampleS, ]
#set.seed(113)
#sampleUM <- sample.int(n = nrow(UMonly), size = 70, replace = F)
#TrainUM <- UMonly[sampleUM, ]
#TestUM  <- UMonly[-sampleUM, ]
#TrainTotalSet <- do.call("rbind", list(TrainALK,TrainHM,TrainIM,TrainLM,TrainS,TrainUM)) #all the data for SVM model training
#TestTotalSet <- do.call("rbind", list(TestALK,TestHM,TestIM,TestLM,TestS,TestUM)) #all the data for SVM model testing


###### Sample 70 data points for each group as training data (rest is test data) ######
#The following section samples 70 grains from each group as SVM should be carried
#out upon groups of equal size to avoid bias. The grains not used for the SVM calculation will
#then be used to carry out a verification of the success rate of the classification.
#Properly, this is a train/test split of the train dataset, but to avoid confusion with the
#actual test dataset (i.e. the unknowns) this split will be prefixed with "internal":
#meaning that these are train and test sets internal to the overall train dataset.

# Step 1: Define data
SVM_splits <- split(SVM_Train_dataset, SVM_Train_dataset$Lithology)
levels_lithology <- levels(SVM_Train_dataset$Lithology)
seeds <- c(64, 34, 2, 136, 185, 113)

# Step 2: Prepare empty lists to store results
train_list <- list()
test_list  <- list()

# Step 3: Loop
for(i in 1:length(levels_lithology)) {
  lith_level<- levels_lithology[i]
  grp <- SVM_splits[[lith_level]] 
  
  set.seed(seeds[i])
  samp <- sample.int(n = nrow(grp), size = 70, replace = FALSE)
  train_list[[lith_level]] <- grp[samp, ]
  test_list[[lith_level]]  <- grp[-samp, ]
}

# Step 4: Combine
TrainTotalSet  <- do.call(rbind, train_list) #all the data for SVM model training
TestTotalSet  <- do.call(rbind, test_list) #all the data for SVM model testing


###### Apply SMV Model that predicts the group depending on LREE and Sr.Y - check predictability ######
# Take base-10 logarithm of LREE and SR.Y
TrainTotalSetLOG <-log10(TrainTotalSet[,1:2])
TrainTotalSetLOG$Lithology<-TrainTotalSet$Lithology
TestTotalSetLOG <-log10(TestTotalSet[,1:2])
TestTotalSetLOG$Lithology<-TestTotalSet$Lithology

# Fit a Support Vector Machine (SVM) using module e1071 to the training data
SVMmodel <- svm(TrainTotalSetLOG$Lithology~., data = TrainTotalSetLOG, kernel = "radial", cost = 0.5, gamma = 2)
plot(SVMmodel, grid = 50, TrainTotalSetLOG, svSymbol = '', dataSymbol = '', col = brewer.pal(7, "Set3"))

#How good was the prediction
PredictionFit_test <- predict(SVMmodel, TestTotalSetLOG)
misclass <- table(predict = PredictionFit_test, truth = TestTotalSetLOG$Lithology) #misclassification table
# SOPHIE: SAVED TABLE
#Write explanation line and then append the table
writeLines("Rows = predicted class, Columns = true class", "Plots/Plot3c_SVM_model_misclassification_table.txt")
write.table(as.data.frame.matrix(misclass), "Plots/Plot3c_SVM_model_misclassification_table.txt", 
            sep = "\t", quote = FALSE, row.names = TRUE, col.names = NA, append = TRUE)

#The following creates an artificial set of data, as a grid of points in PC space, the SVM classification is then applied to this. This grid is then used to create plots using ggplot
SVMmodelGrid<-expand.grid(seq(min(TrainTotalSetLOG[, 1]), max(TrainTotalSetLOG[, 1]),length.out=500),                                                                                                         
                          seq(min(TrainTotalSetLOG[, 2]), max(TrainTotalSetLOG[, 2]),length.out=500)) 
names(SVMmodelGrid) <- names(TrainTotalSetLOG)[1:2] #axes are named as the PCs
PredictGrid <- predict(SVMmodel, SVMmodelGrid) #SVM is applied to the grid of artificial data
SVMgridAesthetics <- data.frame(SVMmodelGrid, PredictGrid) #transforms the grid to data frame


# SOPHIE: CHANGED x- AND y-ticks TO FIT PAPER PLOT
#Define a labeling function that forces plain numbers from a log scale
log_labels <- function(x) {
  format(10^x, scientific = FALSE, trim = TRUE)
}
#Plot 3c
SVM_plot<- ggplot(SVMgridAesthetics, aes(x = LREE, y = Sr.Y, fill = PredictGrid)) + #SOPHIE: Changed name of plot
  geom_tile() +
  #grid
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "dark grey")) +
  #x- and y-label
  labs (x = "LREE", y = "Sr/Y") +
  # Legend
  #scale_fill_manual(values=c("#ededff","#e9f2f2","#d1d1d1", "#b3b3b3", "#999999","#63635c", "#828282")) #grid has now been used to create a ggplot object, plot SVM_plot if you want to see it
  scale_fill_manual(name = lithology_clas_name, values=c("#FDFACF","#F9E5C3","#F1C3C3", "#D3ECDA", "#D3D3F1","#EDDCF5")) +
  # SOPHIE: Change tick labels
  scale_x_continuous(labels = log_labels) +
  scale_y_continuous(labels = log_labels)
# save
ggsave("Plots/Plot3c_SVM_model.pdf", plot = SVM_plot, width = 6, height = 7.5)
 

# ------------ Categories/ Test your data ------------
#Now we're bringing in the UNKNOWNS
UnknownsCoordinates <- log10(new_data[, 4:3]) #SOPHIE: Changed to log to make it work
PredictionFit_new_data <- predict(SVMmodel, UnknownsCoordinates) #uses SVM to predict grain class, you now have guesses for each unknown grain
PredictionFit_new_data <- as.data.frame(PredictionFit_new_data) # (above as a data frame)
PredictionFit_new_data$Grain <- new_data$Grain #SOPHIE: Add grain label
PredictionFit_new_data <- PredictionFit_new_data[, c("Grain", names(PredictionFit_new_data)[names(PredictionFit_new_data) != "Grain"])]
PredictionFit_new_data$Sample_Group <- new_data$Sample_Group #SOPHIE: Add sample_group
PredictionFit_new_data$Age<- new_data$Age #adds age information about the unknowns if included in file
write_excel_csv(PredictionFit_new_data, "Plots/Plot3c_SVM_model_new_data_table.txt") #Export a csv with predictions for each individual grain. You need to choose where the file will be saved on your computer, and its name.

###### 3c Plot your data on SMV plot ######
new_dataFinalLOG<-log10(new_data[,3:4])
new_dataFinalLOG$Age<-new_data$Age
new_dataFinalLOG$Sample_Group<-new_data$Sample_Group
new_dataFinalLOG$Age<-as.numeric(new_data$Age)
mycol <- rgb(0, 0, 0, max = 255, alpha = 50, names = "50%" )

# SOPHIE: REMOVED WARNINGS    
#SVMfinal<-SVMggplot + geom_tile(aes(), size = 1) +
#  geom_point(aes(x=TestdataFinalLOG$LREE, y=TestdataFinalLOG$Sr.Y,colour = TestdataFinalLOG$Age, fill = "ALK"), size=1.5, shape = 19, data = TestdataFinalLOG) +
#  scale_colour_gradient2(low = "yellow", midpoint = Agemiddle, mid = "orange", high = "darkred", limits=c(AgerangeMinimum,AgerangeMaximum), na.value = mycol ) + 
#  labs (colour = "age", fill = "Groups")

#Plot 3c - new data sample groups
SVM_plot_newdata_groups <- SVM_plot + geom_tile() +  # SVM grid/background
  # SOPHIE: Add new data points -> separate into groups
  geom_point(aes(x = LREE, y = Sr.Y, colour = Sample_Group), size = 1.5, shape = 19, data = new_dataFinalLOG, inherit.aes = FALSE) +
  # SOPHIE: Set the colours
  scale_colour_manual(values = setNames(c("darkred", "blue"), sample_group_name)) +
  labs(colour = "Sample group", fill = lithology_clas_name) 
# save
ggsave("Plots/Plot3c_SVM_model_new_data_by_group.pdf", plot = SVM_plot_newdata_groups, width = 6, height = 7.5)


#SOPHIE: Plot 3c - new data sample groups & ages
new_dataFinalLOG_group1 <- new_dataFinalLOG[new_dataFinalLOG$Sample_Group == sample_group_name[1], ]
new_dataFinalLOG_group2 <- new_dataFinalLOG[new_dataFinalLOG$Sample_Group == sample_group_name[2], ]

SVM_plot_newdata_groups_ages <- SVM_plot + 
  geom_tile() +   # grid/background (inherits its own data & aes)
  
  # group 2: colour mapped to Age (solid points)
  geom_point(aes(x = LREE, y = Sr.Y, colour = Age), size = 1.5, shape = 19, data = new_dataFinalLOG_group2, inherit.aes = FALSE) +
  scale_colour_gradient2(name = "Sample age (Group 2)",low = "#a9f7fd", mid = "#559e83", high = "darkblue", limits=c(AgerangeMinimum,AgerangeMaximum), na.value = mycol ) +
  
  # allow a second colour scale
  new_scale_colour() +
  
  # group 1: fill mapped to Age, use shape 21 (fill + border)
  geom_point(aes(x = LREE, y = Sr.Y, colour = Age), size = 1.5, shape = 19, data = new_dataFinalLOG_group1, inherit.aes = FALSE) +
  scale_colour_gradient2(name = "Sample age (Group 1)",low = "#ffc100", midpoint = Agemiddle, mid = "#ff4d00", high = "darkred", limits=c(AgerangeMinimum,AgerangeMaximum), na.value = mycol ) + 
  
  # keep the SVM fill legend name
  labs(fill = lithology_clas_name) +
  # Control legend order
  guides(fill = guide_legend(order = 1),
         colour = guide_colourbar(order = 2))
# save
ggsave("Plots/Plot3c_SVM_model_new_data_by_group_age.pdf", plot = SVM_plot_newdata_groups_ages, width = 6, height = 7.5)


#copy and paste the following to a line if you like, it sticks both main plots together:
#grid.arrange(Allplot,SVMfinal)
