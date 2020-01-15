import numpy as np
import scipy
from time import time
import numpy.linalg as lin
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

Regression=0
Classifier=1

class ELM:
    def __init__(self, elm_type, NumberofHiddenNeurons, ActivationFunction, kkkk, C):
        self.elm_type = elm_type
        self.ActivationFunction = ActivationFunction
        self.kkkk = kkkk
        self.NumberofHiddenNeurons = NumberofHiddenNeurons
        self.C = C

    def train(self, train_data, test_data):
        T = train_data[:, 0].T
        T = T.reshape(1, T.shape[0])
        P = train_data[:, 1:].T

        TVT = test_data[:, 0]
        TVT = TVT.reshape(1, TVT.shape[0])
        TVP = test_data[:, 1:].T

        NumberofTrainingData = P.shape[1]
        NumberofTestingData = TVP.shape[1]
        NumberofInputNeurons = P.shape[0]
        
        if self.elm_type != Regression:
            ls = np.concatenate((T, TVT), axis=1)
            sorted_target = np.sort(ls)
            label = []
            label.append(sorted_target[0][0])
            j = 1
            for i in range(NumberofTrainingData+NumberofTestingData):
                if sorted_target[0][i] != label[j-1]:
                    j += 1
                    label.append(sorted_target[0][i])

            number_class = j
            NumberofOutputNeurons = number_class

            temp_T = np.zeros([NumberofOutputNeurons, NumberofTrainingData])
            for i in range(NumberofTrainingData):
                for j in range(number_class):
                    if label[j] == T[0][i]:
                        break

                temp_T[j][i] = 1
            T = temp_T*2 - 1

            temp_TV_T = np.zeros([NumberofOutputNeurons, NumberofTestingData])
            for i in range(NumberofTestingData):
                for j in range(number_class):
                    if label[j] == TVT[0][i]:
                        break

                temp_TV_T[j][i] = 1
            TVT = temp_TV_T*2 - 1
            
        D_YYM = []
        D_Input = []
        D_beta = []
        D_beta1 = []
        TY = []
        FY = []
        BiasofHiddenNeurons1 = []
        Y = np.zeros((T.shape[0], T.shape[1]))
        
        E1 = T
        BB = []
        D_YYM = []
        for i in range(self.kkkk):
            start_train_time = time()
            Y2 = E1
            scaler = MinMaxScaler(feature_range=(0.01,0.99))
            scaler.fit(Y2.T)
            
            Y22 = scaler.transform(Y2.T)
            Y2 = Y22
            Y4 = (-np.log(1/Y2 - 1)).T
            
            m = np.eye(P.shape[0]) / self.C + P.dot(P.T)
            n = P.dot(Y4.T)
            YYM = lin.solve(m,n)
            
            YJX = P.T.dot(YYM)
            BB2 = np.sum(YJX - Y4.T, axis=0)
            BB.append(BB2[0] / Y4.shape[1])
            end_train_time = time()
            
            TrainingTime = end_train_time - start_train_time
            
            GXZ111 = P.T.dot(YYM) - BB[i]
            
            GXZ2 = 1/(1+ np.exp(-GXZ111.T))
            FYY = scaler.inverse_transform(GXZ2.T).T
            
            FT1 = FYY

            #TrainingAccuracy = np.sqrt(mean_squared_error(FT1, E1))
            E1 = E1 - FT1
            
            Y = Y + FYY
            D_YYM.append(YYM)
            
            if self.elm_type == Classifier:
                MissClassificationRate_Training = 0
                for j in range(T.shape[1]):
                    x, label_index_expected = T[:, j].max(0), T[:, j].argmax(0)
                    x, label_index_actual = Y[:, j].max(0), Y[:, j].argmax(0)

                    if label_index_actual != label_index_expected:
                        MissClassificationRate_Training += 1

                TrainingAccuracy = 1 - MissClassificationRate_Training / T.shape[1]
        
        start_test_time = time()
        TY2 = np.zeros((TVT.shape[0], TVT.shape[1]))
        for i in range(self.kkkk):
            GXZ1 = D_YYM[i].T.dot(TVP) - BB[i]
            GXZ2 = 1/(1+ np.exp(-GXZ1.T))
            FYY = scaler.inverse_transform(GXZ2).T
            TY2 = TY2 + FYY
            #TestingAccuracy = np.sqrt(mean_squared_error(TY2, TVT))
            
        end_test_time = time()      
        test_time = end_test_time - start_test_time

        le = []
        la = []
        if self.elm_type == Classifier:
            MissClassificationRate_Testing = 0
            for j in range(TVT.shape[1]):
                x, label_index_expected = TVT[:, j].max(0), TVT[:, j].argmax(0)
                x, label_index_actual = TY2[:, j].max(0), TY2[:, j].argmax(0)
                le.append(label_index_expected)
                la.append(label_index_actual)
                if label_index_actual != label_index_expected:
                    MissClassificationRate_Testing += 1

            TestingAccuracy = 1 - MissClassificationRate_Testing / TVT.shape[1]
                
        return TrainingAccuracy, TestingAccuracy, np.array(le), np.array(la)