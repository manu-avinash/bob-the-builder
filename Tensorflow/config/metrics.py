class Metrics():
    def __init__(self):
        pass
#Accuracy metrics
    def Accuracy(self):
        pass
    def BinaryAccuracy(self):
        pass
    def CategoricalAccuracy(self):
        pass
    def SparseCategoricalAccuracy(self):
        pass
    def TopKCategoricalAccuracy(self):
        pass
    def SparseTopKCategoricalAccuracy(self):
        pass
    #Probabilistic metrics
    def BinaryCrossentropy(self):
        pass
    def CategoricalCrossentropy(self):
        pass
    def SparseCategoricalCrossentropy(self):
        pass
    def KLDivergence(self):
        pass
    def Poisson(self):
        pass
    #Regression metrics
    def MeanSquaredError(self):
        pass
    def RootMeanSquaredError(self):
        pass
    def MeanAbsoluteError(self):
        pass
    def MeanAbsolutePercentageError(self):
        pass
    def MeanSquaredLogarithmicError():
        pass
    def CosineSimilarity(self):
        pass
    def LogCoshError(self):
        pass
    def R2Score(self):
        pass
    #Classification metrics based on True/False positives & negatives    
    def AUC(self):
        pass
    def Precision(self):
        pass
    def Recall(self):
        pass
    def TruePositives(self):
        pass
    def TrueNegatives(self):
        pass
    def FalsePositives(self):
        pass
    def FalseNegatives(self):
        pass
    def PrecisionAtRecall(self):
        pass
    def RecallAtPrecision(self):
        pass
    def SensitivityAtSpecificity(self):
        pass
    def SpecificityAtSensitivity(self):
        pass
    def F1Score(self):
        pass
    def FBetaScore(self):
        pass
    def PearsonCorrelation(self):
        pass
    def ConcordanceCorrelation(self):
        pass
    #Image segmentation metrics
    def IoU(self):
        pass
    def BinaryIoU(self):
        pass
    def OneHotIoU(self):
        pass
    def OneHotMeanIoU(self):
        pass
    def MeanIoU(self):
        pass
    #Hinge metrics for "maximum-margin" classification
    def Hinge(self):
        pass
    def SquaredHinge(self):
        pass
    def CategoricalHinge(self):
        pass
    #Metric wrappers and reduction metrics
    def MeanMetricWrapper(self):
        pass
    def Mean(self):
        pass
    def Sum(self):
        pass