import pandas as pd
import numpy as np
import copy as cp 
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
class MachineLearningClassifier:

    def __init__(self,Classifier):
        self.ClassifierContainer = Classifier

    def Train(self,X,Y):
        self.ClassifierContainer =  self.ClassifierContainer.fit(X,Y)

    def Test(self,X,Y):
        return accuracy_score(Y,self.ClassifierContainer.predict(X))

 

class featureSelection:
    def __init__(self, File, Classifier):
        self.File       = File
        self.Classifier = Classifier

    def ParseAndGenerateData(self):
        try:
            self.data = pd.read_csv(self.File).sample(frac=1).reset_index(drop=True)
            print("The File was read correctly")
            print (self.data.head())
            self.Y = self.data.iloc[:,0]
            print ("Keys Parse")
            self.X = self.data.iloc[:,1:]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size = 0.3)
            print("Data Parsing Finish")
        except:
            print("The file does not exist, Use .ChangeFileName to change the file")

    def ChangeFileName(self,File):
        self.File = File
          

    def BuildDataSetForSetTesting(self,Features):
        data = self.X_test.iloc[:,Features]
        return data       
    def TrainClassifier(self,Features):
        subSetDataTraining = self.X_train.iloc[:,Features]
        self.Classifier.Train(subSetDataTraining,self.y_train)

    def TestAccuracyWithSetOfFeatures(self,Features):
        subSetDataTesting = self.X_test.iloc[:,Features]
        return self.Classifier.Test(subSetDataTesting,self.y_test)

    def FeatureSelectionAlgorithm(self):         
        PossibleFeatures  =  np.arange(np.size(self.data,1)-1)
        Result            = FeaturesArray     = []
        MaxAccuracy = -1
        for numFeatures in range(len(PossibleFeatures)):
            index = bestAccuracy  = -1
            for  i,EachFeatureLeft in enumerate(PossibleFeatures):
                FeaturesArray.append(EachFeatureLeft)
                self.TrainClassifier(FeaturesArray)
                accuracy =  self.TestAccuracyWithSetOfFeatures(FeaturesArray)
                if bestAccuracy < accuracy:
                    index  = i
                    bestAccuracy = accuracy
                print("Tested Features: ",FeaturesArray, " with accuary %f",accuracy)
                FeaturesArray=FeaturesArray[:-1] 
            FeaturesArray.append(PossibleFeatures[index])
            PossibleFeatures=np.delete(PossibleFeatures,index)
            print ("The best features in this round are ", FeaturesArray ," with accuary of ", bestAccuracy)
            if MaxAccuracy < bestAccuracy:
                result = cp.deepcopy(FeaturesArray)
                MaxAccuracy = bestAccuracy
        print ("The best features are ", result ," with accuary of ", MaxAccuracy)







if __name__ == "__main__":
    # K nearest neighbor
    print("K nearest neighbor")
    nn    = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', p=1)
    knn   = MachineLearningClassifier(nn)
    FsKNN = featureSelection("data.csv", knn)
    FsKNN.ParseAndGenerateData()
    FsKNN.FeatureSelectionAlgorithm()


    # # naive bayes 
    print("Naive Bayes Algorithm")
    naive   = GaussianNB()
    naive   = MachineLearningClassifier(naive)
    FsNaive = featureSelection("data.csv", naive)
    FsNaive.ParseAndGenerateData()
    FsNaive.FeatureSelectionAlgorithm()



    # SVM
    print("Vector Machines")
    VectorMachines = SVC(gamma='auto')    # 
    VectorMachines = MachineLearningClassifier(VectorMachines)
    FsSVM          = featureSelection("data.csv", VectorMachines)
    FsSVM.ParseAndGenerateData()
    FsSVM.FeatureSelectionAlgorithm()