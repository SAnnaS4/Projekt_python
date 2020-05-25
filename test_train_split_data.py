# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 20:09:10 2018

"""
import numpy as np
import numpy.random as nr
import pandas as pd

class train_test(object):
    def __init__(self, spectra):
        self.spectra = spectra

    def splitfirst(self):
        labels = np.array(self.spectra['label'])
        temp_features = self.spectra.values[:, 2:]

        nr.seed(42)
        import sklearn.model_selection as ms
        indices = range(temp_features.shape[0])
        indices = ms.train_test_split(indices, test_size=0.2)
        x_train = temp_features[indices[0], :]
        y_train = np.ravel(labels[indices[0]])
        x_test = temp_features[indices[1], :]
        y_test = np.ravel(labels[indices[1]])

        return x_train, y_train, x_test, y_test

    def split(self):
        labels = np.array(self.spectra['label'])
        temp_features = self.spectra.values[:, 2:]

        nr.seed(42)
        import sklearn.model_selection as ms
        indices = range(temp_features.shape[0])
        indices = ms.train_test_split(indices, test_size=0.1)
        x_train = temp_features[indices[0], :]
        y_train = np.ravel(labels[indices[0]])
        x_test = temp_features[indices[1], :]
        y_test = np.ravel(labels[indices[1]])

        return x_train, y_train, x_test, y_test

    def splitPatient(self):
        labels = np.array(self.spectra['label'])
        temp_features = self.spectra.values[1:, 2:]

        nr.seed(42)
        import sklearn.model_selection as ms
        indices = range(temp_features.shape[0])
        indices = ms.train_test_split(indices, test_size=0.1)
        x_train = temp_features[indices[0], :]
        y_train = np.ravel(labels[indices[0]])
        x_test = temp_features[indices[1], :]
        y_test = np.ravel(labels[indices[1]])

        return x_train, y_train, x_test, y_test

    def div_set_underbalanced(self):
        labels = np.array(self.spectra['label'])
        temp_label_1 = labels[labels == 2]
        temp_features_1 = self.spectra.values[labels == 2, 1:]
        temp_label_2 = labels[labels == 3]
        temp_features_2 = self.spectra.values[labels == 3, 1:]
        temp_label_3 = labels[labels == 4]
        temp_features_3 = self.spectra.values[labels == 4, 1:]
        temp_label_4 = labels[labels == 5]
        temp_features_4 = self.spectra.values[labels == 5, 1:]

        print("healthy ", temp_features_1.shape[0])
        print("cancer          - ", temp_features_2.shape[0])

        ind_1 = nr.choice(temp_features_1.shape[0], temp_features_3.shape[0], replace=True)
        ind_2 = nr.choice(temp_features_2.shape[0], temp_features_3.shape[0], replace=True)
        ind_4 = nr.choice(temp_features_4.shape[0], temp_features_3.shape[0], replace=True)

        print('Undersampling')
        print("EAC - ", temp_features_1[ind_1, :].shape[0])
        print("Plattenepithel - ", temp_features_2[ind_2, :].shape[0])
        print("Stroma - ", temp_features_3.shape[0])
        print("Blank - ", temp_features_4[ind_4].shape[0])

        temp_features = np.concatenate((temp_features_1[ind_1, :],temp_features_2[ind_2, :], temp_features_3, temp_features_4[ind_4, :]), axis=0)
        temp_labels = np.concatenate((temp_label_1[ind_1,],temp_label_2[ind_2,] ,temp_label_3, temp_label_4[ind_4,]), axis=0)

        nr.seed(42)
        import sklearn.model_selection as ms
        indices = range(temp_features.shape[0])
        indices = ms.train_test_split(indices, test_size=0.1)
        x_train = temp_features[indices[0], :]
        y_train = np.ravel(temp_labels[indices[0]])
        x_test = temp_features[indices[1], :]
        y_test = np.ravel(temp_labels[indices[1]])

        return x_train, y_train, x_test, y_test

    def div_set_balanced_3classes(self):
        labels = np.array(self.spectra['label'])
        labels=np.int_(labels)
        temp_label_1 = labels[labels == 1]
        temp_features_1 = self.spectra.values[labels == 1, 1:]
        temp_label_2 = labels[labels == 2]
        temp_features_2 = self.spectra.values[labels == 2, 1:]
        temp_label_3 = labels[labels == 3]
        temp_features_3 = self.spectra.values[labels == 3, 1:]
        temp_label_4 = labels[labels == 4]
        temp_features_4 = self.spectra.values[labels == 4, 1:]
        print('Undersampling')
        print("EAC ", temp_features_1.shape[0])
        print("Stroma          - ", temp_features_2.shape[0])
        print("Plattenepithel          - ", temp_features_3.shape[0])
        print("Blank          - ", temp_features_4.shape[0])

        ind_3 = nr.choice(temp_features_3.shape[0], temp_features_1.shape[0], replace=True)
        ind_2 = nr.choice(temp_features_2.shape[0], temp_features_1.shape[0], replace=True)
        ind_4 = nr.choice(temp_features_4.shape[0], temp_features_1.shape[0], replace=True)

        temp_features = np.concatenate((temp_features_1,temp_features_2[ind_2,:], temp_features_3[ind_3,:], temp_features_4[ind_4,:]), axis=0)
        temp_features = np.float_(temp_features)
        temp_labels = np.concatenate((temp_label_1, temp_label_2[ind_2,],temp_label_3[ind_3,],temp_label_4[ind_4,]), axis=0)
        print('After Undersampling')
        print("EAC ", temp_features_1.shape[0])
        print("Stroma          - ", temp_features_2[ind_2,:].shape[0])
        print("Plattenepithel          - ", temp_features_3[ind_3,:].shape[0])
        print("Blank          - ", temp_features_4[ind_4, :].shape[0])

        nr.seed(42)
        import sklearn.model_selection as ms
        indices = range(temp_features.shape[0])
        indices = ms.train_test_split(indices, test_size=0.1)
        x_train = temp_features[indices[0], :]
        y_train = np.ravel(temp_labels[indices[0]])
        x_test = temp_features[indices[1], :]
        y_test = np.ravel(temp_labels[indices[1]])

        return x_train, y_train, x_test, y_test

    def div_set_balanced_real3classes(self):
        labels = np.array(self.spectra['label'])
        labels=np.int_(labels)
        temp_label_1 = labels[labels == 1]
        temp_features_1 = self.spectra.values[labels == 1, 1:]
        temp_label_2 = labels[labels == 2]
        temp_features_2 = self.spectra.values[labels == 2, 1:]
        temp_label_3 = labels[labels == 3]
        temp_features_3 = self.spectra.values[labels == 3, 1:]

        print('Undersampling')
        print("EAC und Stroma ", temp_features_1.shape[0])
        print("Plattenepithel          - ", temp_features_2.shape[0])
        print("Blank          - ", temp_features_3.shape[0])


        ind_3 = nr.choice(temp_features_3.shape[0], temp_features_1.shape[0], replace=True)
        ind_2 = nr.choice(temp_features_2.shape[0], temp_features_1.shape[0], replace=True)
        #ind_4 = nr.choice(temp_features_4.shape[0], temp_features_1.shape[0], replace=True)

        temp_features = np.concatenate((temp_features_1,temp_features_2[ind_2,:], temp_features_3[ind_3,:]), axis=0)
        temp_features = np.float_(temp_features)
        temp_labels = np.concatenate((temp_label_1, temp_label_2[ind_2,],temp_label_3[ind_3,]), axis=0)
        print('After Undersampling')
        print("EAC und Stroma ", temp_features_1.shape[0])
        print("Plattenepithel          - ", temp_features_2[ind_2,:].shape[0])
        print("          Blank- ", temp_features_3[ind_3,:].shape[0])


        nr.seed(42)
        import sklearn.model_selection as ms
        indices = range(temp_features.shape[0])
        indices = ms.train_test_split(indices, test_size=0.1)
        x_train = temp_features[indices[0], :]
        y_train = np.ravel(temp_labels[indices[0]])
        x_test = temp_features[indices[1], :]
        y_test = np.ravel(temp_labels[indices[1]])

        return x_train, y_train, x_test, y_test

    def div_set_underbalanced_2_class(self):
        labels = np.array(self.spectra['label'])
        temp_label_1 = labels[labels == 1]
        temp_features_1 = self.spectra.values[labels == 1, 1:]
        temp_label_2 = labels[labels == 0]
        temp_features_2 = self.spectra.values[labels == 0, 1:]


        print("All other ", temp_features_1.shape[0])
        print("EAC         - ", temp_features_2.shape[0])

        ind_1 = nr.choice(temp_features_1.shape[0], temp_features_2.shape[0], replace=True)

        print('Undersampling')
        print("All other - ", temp_features_1[ind_1, :].shape[0])
        print("EAC - ", temp_features_2.shape[0])

        temp_features = np.concatenate((temp_features_1[ind_1, :],temp_features_2), axis=0)
        temp_labels = np.concatenate((temp_label_1[ind_1,],temp_label_2), axis=0)

        nr.seed(42)
        import sklearn.model_selection as ms
        indices = range(temp_features.shape[0])
        indices = ms.train_test_split(indices, test_size=0.1)
        x_train = temp_features[indices[0], :]
        y_train = np.ravel(temp_labels[indices[0]])
        x_test = temp_features[indices[1], :]
        y_test = np.ravel(temp_labels[indices[1]])

        return x_train, y_train, x_test, y_test


    def div_SMOTE_set(self):
        print('SMOTE Oversampling')
        labels = np.array(self.spectra['label'])
        features = np.asmatrix(self.spectra.drop(['label'], axis=1))

        from imblearn.over_sampling import SMOTE, ADASYN


        temp_features_1 = self.spectra.values[labels == 1, 1:]

        temp_features_2 = self.spectra.values[labels == 2, 1:]
        print('Before Sampling')

        print("healthy ", temp_features_1.shape[0])
        print("cancer          - ", temp_features_2.shape[0])

        temp_features, temp_labels = SMOTE().fit_resample(features, labels)
        print('After Sampling')
        print("labels   - ", temp_labels.shape[0])
        print("healthy   - ", temp_labels[temp_labels == 1].shape[0])
        print("cancer ", temp_labels[temp_labels == 2].shape[0])
        nr.seed(42)
        import sklearn.model_selection as ms
        indices = range(temp_features.shape[0])
        indices = ms.train_test_split(indices, test_size=0.1)
        x_train = temp_features[indices[0], :]
        y_train = np.ravel(temp_labels[indices[0]])
        x_test = temp_features[indices[1], :]
        y_test = np.ravel(temp_labels[indices[1]])


        return x_train, y_train, x_test, y_test


    def div_feat_labels(self):
        labels = np.array(self.spectra['label'])
        temp_label_0 = labels[labels==0]
        temp_features_0 = self.spectra.values[labels==0,1:]
        temp_label_1 = labels[labels==1]
        temp_features_1 = self.spectra.values[labels==1,1:]
        temp_label_2 = labels[labels==2]
        temp_features_2 = self.spectra.values[labels==2,1:]
        temp_label_3 = labels[labels==3]
        temp_features_3 = self.spectra.values[labels==3,1:]
        temp_label_4 = labels[labels==4]
        temp_features_4 = self.spectra.values[labels==4,1:]
        temp_label_5 = labels[labels==5]
        temp_features_5 = self.spectra.values[labels==5,1:]
        
        ind_0 = nr.choice(temp_features_0.shape[0], temp_features_4.shape[0], replace=True)
        ind_1 = nr.choice(temp_features_1.shape[0], temp_features_4.shape[0], replace=True)
        ind_2 = nr.choice(temp_features_2.shape[0], temp_features_4.shape[0], replace=True)
        ind_3 = nr.choice(temp_features_3.shape[0], temp_features_4.shape[0], replace=True)
        ind_5 = nr.choice(temp_features_5.shape[0], temp_features_4.shape[0], replace=True)
        
        print("bg clinical   - ",temp_features_0[ind_0,:].shape[0])
        print("bg instrument - ",temp_features_1[ind_1,:].shape[0])
        print("skin          - ",temp_features_2[ind_2,:].shape[0])
        print("thyroid       - ",temp_features_3[ind_3,:].shape[0])
        print("parathyroid   - ",temp_features_4.shape[0])
        print("muscle        - ",temp_features_5[ind_5,:].shape[0])
        
        temp_features = np.concatenate((temp_features_0[ind_0,:],temp_features_1[ind_1,:],temp_features_2[ind_2,:],temp_features_3[ind_3,:],temp_features_4,temp_features_5[ind_5,:]),axis=0)
        temp_labels = np.concatenate((temp_label_0[ind_0,],temp_label_1[ind_1,],temp_label_2[ind_2,],temp_label_3[ind_3,],temp_label_4,temp_label_5[ind_5,]),axis=0)
        
        data_df = pd.DataFrame(temp_features)
        data_df.insert(loc=0,value = (temp_labels.reshape(-1,1)), column='label')
        return data_df
    
    def div_feat_labels_cv(self):
        labels = np.array(self.spectra['label'])
        temp_label_0 = labels[labels==0]
        temp_features_0 = self.spectra.values[labels==0,1:]
        temp_label_1 = labels[labels==1]
        temp_features_1 = self.spectra.values[labels==1,1:]
        temp_label_2 = labels[labels==2]
        temp_features_2 = self.spectra.values[labels==2,1:]
        temp_label_3 = labels[labels==3]
        temp_features_3 = self.spectra.values[labels==3,1:]
        temp_label_4 = labels[labels==4]
        temp_features_4 = self.spectra.values[labels==4,1:]
        temp_label_5 = labels[labels==5]
        temp_features_5 = self.spectra.values[labels==5,1:]
        
        ind_0 = nr.choice(temp_features_0.shape[0], temp_features_4.shape[0], replace=True)
        ind_1 = nr.choice(temp_features_1.shape[0], temp_features_4.shape[0], replace=True)
        ind_2 = nr.choice(temp_features_2.shape[0], temp_features_4.shape[0], replace=True)
        ind_3 = nr.choice(temp_features_3.shape[0], temp_features_4.shape[0], replace=True)
        ind_5 = nr.choice(temp_features_5.shape[0], temp_features_4.shape[0], replace=True)
        
        print("bg clinical   - ",temp_features_0[ind_0,:].shape[0])
        print("bg instrument - ",temp_features_1[ind_1,:].shape[0])
        print("skin          - ",temp_features_2[ind_2,:].shape[0])
        print("thyroid       - ",temp_features_3[ind_3,:].shape[0])
        print("parathyroid   - ",temp_features_4.shape[0])
        print("muscle        - ",temp_features_5[ind_5,:].shape[0])
        
        temp_features = np.concatenate((temp_features_0[ind_0,:],temp_features_1[ind_1,:],temp_features_2[ind_2,:],temp_features_3[ind_3,:],temp_features_4,temp_features_5[ind_5,:]),axis=0)
        temp_labels = np.concatenate((temp_label_0[ind_0,],temp_label_1[ind_1,],temp_label_2[ind_2,],temp_label_3[ind_3,],temp_label_4,temp_label_5[ind_5,]),axis=0)
        
        return temp_features, temp_labels
    
    def div_set(self):
        features = self.spectra.values[:,1:]
        features=np.float_(features)
        labels = np.array(self.spectra['label'])
        labels = np.int_(labels)
        
        nr.seed(42)
        import sklearn.model_selection as ms
        indices = range(features.shape[0])
        indices = ms.train_test_split(indices, test_size=0.1)
        x_train = features[indices[0],:]
        y_train = np.ravel(labels[indices[0]])
        x_test = features[indices[1],:]
        y_test = np.ravel(labels[indices[1]])
        print("dataset split into train and test")
        return x_train, y_train, x_test, y_test