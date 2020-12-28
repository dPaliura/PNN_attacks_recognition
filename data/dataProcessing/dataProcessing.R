# Set path to local repo instead of [directory]
dir <- "[directory]/data"
setwd(dir)


subsetting.rate <- 0.25
min.observations <- 1000
normals.amount <- 3000

# Read data
fields <- read.csv("dataSet/FieldNames.csv", head=FALSE)

train.set <- read.csv("dataSet/KDDTrain+.csv", head=FALSE)[,-43]
test.set <- read.csv("dataSet/KDDTest+.csv", head=FALSE)[,-43]

# Filter data
train.set <- na.exclude(train.set)
test.set <- na.exclude(test.set)

# Paste all data into one set
data <- rbind(train.set, test.set)

# Add column names as given in FieldNames.csv file
names(data) <- c(as.character(fields[,1]), "class")

# Take a look at data
summary(data)

# And save indeces of sets in pasted data
train.size <- nrow(train.set)
train.indcs <- 1:train.size
test.indcs <- 1:nrow(test.set) + train.size


# Remove non-actual variables
rm(train.size, train.set, test.set)


# Normalize continuous data
for (i in 1:(ncol(data)-1)){
    if (fields[i,2] == "continuous"){
        rng <- range(data[,i])
        if (rng[1]!=0 | rng[2]!=1){
            if (rng[1] != rng[2]){
                data[,i] <- (data[,i] - rng[1])/(rng[2] - rng[1])
            }
        }
    }
}

# Encoding symbolic data
for (i in 1:nrow(fields)){
    field.indx <- which(names(data)==fields[i,1])
    classes <- unique(data[,field.indx])
    data[,field.indx] = sapply(data[,field.indx], function(x) which(x==classes))
}

# Take a look at final data format and check if it contains any NA
summary(data)
any(is.na(data))

# Save all unique classes into variable due to end of work with data variable
classes <- unique(data$class)

# Split data into subsets
train.set <- data[train.indcs,]
test.set <- data[sample(test.indcs),]
rm(data, train.indcs, test.indcs, 
   field.indx, rng, i)

# Save only test set at the moment
write.csv(test.set, "../input/KDDTest_procsd.csv")
rm(test.set)

# Reduce observations or each class maximum to max.class.observations
train.set.reduced <- NULL
for (cls in classes){
    observations <- train.set[train.set$class==cls,]
    n <- nrow(observations)
    if (cls == "normal"){
        indcs <- sample(1:n, normals.amount)
    }
    else{
        if (n < min.observations){
            indcs <- sample(1:n)
        }
        else{
            indcs <- sample(1:n, max(min.observations, floor(n*subsetting.rate)))
        }
    }
    
    train.set.reduced <- rbind(train.set.reduced, observations[indcs,])
}

any(is.na(train.set.reduced))

train.set.reduced <- na.exclude(train.set.reduced)

# Save train set into file
write.csv(train.set.reduced, "../input/KDDTrain_procsd_redcd.csv")

rm(train.set, fields, observations, train.set.reduced, 
   classes, cls, indcs, min.observations, n, subsetting.rate)

