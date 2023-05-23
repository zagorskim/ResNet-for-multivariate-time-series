require("wrapr")
require("RWeka")
require("httr")
setwd("D:/PW/Warsztaty\ z\ Technik\ Uczenia\ Maszynowego/FastTrack4")
set.seed(1234)

read_data <- function(chosenDataset, fileFormat) {
  datasets <- c(
    "AbnormalHeartbeat"                                            ,
    "ACSF1",
    "Adiac",
    "AllGestureWiimoteX",
    "AllGestureWiimoteY",
    "AllGestureWiimoteZ",
    "ArrowHead",
    "ArticularyWordRecognition",
    "AsphaltObstacles",
    "AsphaltObstaclesCoordinates",
    "AsphaltPavementType",
    "AsphaltPavementTypeCoordinates",
    "AsphaltRegularity",
    "AsphaltRegularityCoordinates",
    "AtrialFibrillation",
    "BasicMotions",
    "Beef",
    "BeetleFly",
    "BinaryHeartbeat",
    "BirdChicken",
    "Blink",
    "BME",
    "Car",
    "CatsDogs",
    "CBF",
    "CharacterTrajectories",
    "Chinatown",
    "ChlorineConcentration",
    "CinCECGTorso",
    "Coffee",
    "Colposcopy",
    "Computers",
    "CounterMovementJump",
    "Cricket",
    "CricketX",
    "CricketY",
    "CricketZ",
    "Crop",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxTW",
    "DodgerLoopDay",
    "DodgerLoopGame",
    "DodgerLoopWeekend",
    "DuckDuckGeese",
    "DucksAndGeese",
    "Earthquakes",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "EigenWorms",
    "ElectricDeviceDetection",
    "ElectricDevices",
    "EMOPain",
    "EOGHorizontalSignal",
    "EOGVerticalSignal",
    "Epilepsy",
    "ERing",
    "EthanolConcentration",
    "EthanolLevel",
    "EyesOpenShut",
    "FaceAll",
    "FaceDetection",
    "FaceFour",
    "FacesUCR",
    "FiftyWords",
    "FingerMovements",
    "Fish",
    "FordA",
    "FordB",
    "FreezerRegularTrain",
    "FreezerSmallTrain",
    "FruitFlies",
    "Fungi",
    "GestureMidAirD1",
    "GestureMidAirD2",
    "GestureMidAirD3",
    "GesturePebbleZ1",
    "GesturePebbleZ2",
    "GunPoint",
    "GunPointAgeSpan",
    "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung",
    "Ham",
    "HandMovementDirection",
    "HandOutlines",
    "Handwriting",
    "Haptics",
    "Heartbeat",
    "Herring",
    "HouseTwenty",
    "InlineSkate",
    "InsectEPGRegularTrain",
    "InsectEPGSmallTrain",
    "InsectSound",
    "InsectWingbeat",
    "ItalyPowerDemand",
    "JapaneseVowels",
    "KeplerLightCurves",
    "LargeKitchenAppliances",
    "Libras",
    "Lightning2",
    "Lightning7",
    "LSST",
    "Mallat",
    "Meat",
    "MedicalImages",
    "MelbournePedestrian",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxTW",
    "MindReading",
    "MixedShapes",
    "MixedShapesSmallTrain",
    "MosquitoSound",
    "MoteStrain",
    "MotionSenseHAR",
    "MotorImagery",
    "NATOPS",
    "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2",
    "OliveOil",
    "OSULeaf",
    "PEMS-SF",
    "PenDigits",
    "PhalangesOutlinesCorrect",
    "Phoneme",
    "PhonemeSpectra",
    "PickupGestureWiimoteZ",
    "PigAirwayPressure",
    "PigArtPressure",
    "PigCVP",
    "PLAID",
    "Plane",
    "PowerCons",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxTW",
    "RacketSports",
    "RefrigerationDevices",
    "RightWhaleCalls",
    "Rock",
    "ScreenType",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "SemgHandGenderCh2",
    "SemgHandMovementCh2",
    "SemgHandSubjectCh2",
    "ShakeGestureWiimoteZ",
    "ShapeletSim",
    "ShapesAll",
    "SharePriceIncrease",
    "SmallKitchenAppliances",
    "SmoothSubspace",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    "SpokenArabicDigits",
    "StandWalkJump",
    "StarLightCurves",
    "Strawberry",
    "SwedishLeaf",
    "Symbols",
    "SyntheticControl",
    "Tiselac",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "Trace",
    "TwoLeadECG",
    "TwoPatterns",
    "UMD",
    "UrbanSound",
    "UWaveGestureLibrary",
    "UWaveGestureLibraryAll",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "Wafer",
    "Wine",
    "WordSynonyms",
    "Worms",
    "WormsTwoClass",
    "Yoga")
  
  # Choosing input data
  dataURL <- "https://timeseriesclassification.com/Downloads/"
  
  download.file(paste(dataURL, chosenDataset, ".zip", sep = ""), "file.zip")
  unzip("./file.zip", exdir = "./data")

  file.remove("file.zip")

  if(fileFormat == "arff") {
    connection <- file(description = paste("./data/", chosenDataset, "_TRAIN.arff", sep = ""))
    dfTrain <- read.arff(connection)
    connection <- file(description = paste("./data/", chosenDataset, "_TEST.arff", sep = ""))
    dfTest <- read.arff(connection)
    correct = TRUE
  } else if(fileFormat == "txt") 
  {
    connection <- file(description = paste("./data/", chosenDataset, "_TRAIN.txt", sep = ""))
    dfTrain <- read.table(connection)
    connection <- file(description = paste("./data/", chosenDataset, "_TEST.txt", sep = ""))
    dfTest <- read.table(connection)
    correct = TRUE
  } else {
    cat("Wrong format")
  }

  file.remove(paste("./data/", list.files("./data"), sep = ""))
  ret <- list(dfTrain, dfTest, fileFormat == "txt")
  names(ret) <- c("dfTrain", "dfTest", "isTxt")
  return (ret)
}

cut_series <- function(list, isTxt) {
  first = 0
  last = 0
  if(isTxt) {
    first = 1
  } else
    last = 1
  for(index1 in 1:length(list[[1]][[1]])) {
    for(index2 in (sample(floor(length(list[[1]]) * 0.1) : floor(length(list[[1]]) * 0.4), 1) + first):(length(list[[1]]) - last)) {
      list[[1]][[index2]][index1] = -1 
    }
  }
  for(index1 in 1:length(list[[1]][[2]])) {
    for(index2 in (sample(floor(length(list[[2]]) * 0.41) : floor(length(list[[2]]) * 0.7), 1) + first):(length(list[[2]]) - last)) {
      list[[2]][[index2]][index1] = -1 
    }
  }
  for(index1 in 1:length(list[[1]][[3]])) {
    for(index2 in (sample(floor(length(list[[3]]) * 0.71) : floor(length(list[[3]])), 1) + first):(length(list[[3]]) - last)) {
      list[[3]][[index2]][index1] = -1 
    }
  }
  return(list)
}

cut_preprocessing <- function (df, isTxt) {
  # Cutting procedure
  target = 1
  if(!isTxt)
    target = length(df)

  classes <- split(df, df[target])
  splittingFactor <- lapply(classes, function(x) sample(factor(sort(1:length(x[[1]]))%%3)))
  splitList <- list()
  for(index in 1:length(classes)) {
    splitList[[index]] <- split(classes[[index]], splittingFactor[[index]])
  }
  cut <- lapply(splitList, function(x) cut_series(x, isTxt))
  for(index in 1:length(classes)) {
    classes[[index]] <- unsplit(cut[[index]], splittingFactor[[index]])
  }
  df <- unsplit(classes, df[target])
  
  return(df)
}

# with many classes sometimes triggers error and sometimes not?????!!!
# for(i in 1:10){
#   cut_series(splitList[[1]], TRUE)
# }

cut_store <- function(dfTrain, dfTest, isTxt, chosenDir = "", chosenSaveFormat = "txt") {
  
  # Cutting
  train <- cut_preprocessing(dfTrain, isTxt)
  test <- cut_preprocessing(dfTest, isTxt)
  
  if(chosenDir == "")
    chosenDir <- paste("./cut_data/", "data", sep = "") 

    if(chosenSaveFormat == "arff") {
      write.arff(train, file=paste(chosenDir, "_cut_TRAIN.arff", sep = ""))
      write.arff(test, file=paste(chosenDir, "_cut_TEST.arff", sep = ""))
      correct = TRUE
    } else if(chosenSaveFormat == "txt") {
      write.table(train, file=paste(chosenDir, "_cut_TRAIN.txt", sep = ""))
      write.table(test, file=paste(chosenDir, "_cut_TEST.txt", sep = ""))
      correct = TRUE
    } else {
      cat("Wrong format")
    }
  
  return(list(train, test))
}

# TBD add wrapper to get_data with all dialog options already inserted
get_data <- function(download, dataset = "BeetleFly", format = "txt", train_path = "", test_path = "") {
  if (download) {
    to[dfTrain <- dfTrain, dfTest <- dfTest, isTxt <- isTxt] <- read_data(dataset, format)
    temp <- cut_store(dfTrain, dfTest, isTxt)
    train <- temp[[1]]
    test <- temp[[2]]
  } else {
    connection <- file(description = train_path)
    if (substr(train_path, nchar(train_path) - 3, nchar(train_path)) == "arff") {
      train <- as.list(read.arff(connection))
    } else if (substr(train_path, nchar(train_path) - 2, nchar(train_path)) == "txt")
    {
      train <- as.list(read.table(connection))
    }
    connection <- file(description = test_path)
    if (substr(test_path, nchar(test_path) - 3, nchar(test_path)) == "arff") {
      test <- as.list(read.arff(connection))
    } else if (substr(test_path, nchar(test_path) - 2, nchar(test_path)) == "txt")
    {
      test <- as.list(read.table(connection))
    }
  }
  # testClear <- lapply(test, function(x)
  #   na.omit(x))
  # trainClear <- lapply(train, function(x)
  #   na.omit(x))
  return(list(train, test))
}

