# -*- coding: utf-8 -*-
"""
Created on

@author: Marianne
"""
from sklearn.externals import joblib
import numpy.random as nr
import time
from sklearn import svm
import sklearn.metrics as sklm
from sklearn.model_selection import cross_validate
from pandas_ml import ConfusionMatrix
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from hypercube_data import Cube_Read
from hypercube_data import cube
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score,f1_score
from scipy.signal import savgol_filter
from sklearn.metrics import classification_report
import csv
import os
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.lines import Line2D
from sklearn.model_selection import StratifiedShuffleSplit,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

class blind_test_algorithm(object):
    def __init__(self, train_data, test_data, Validation_x_train, Validation_y_train, Validation_x_test, Validation_y_test,test_path,PatNum,classifier):
        ### gaussian ####
        # self.Lastnm = 122

        ###gaussian pca ###
        self.Lastnm = 65



        self.Firstnm = 0
        self.WaveArea = self.Lastnm - self.Firstnm
        print(
            'Wavelength area from :' + str(self.Firstnm * 5 + 500) + 'nm   till :' + str((self.Lastnm-1) * 5 + 500) + 'nm')
        self.features_train = np.float_(train_data.values[:, self.Firstnm+1:self.Lastnm+1])
        self.labels_train = np.int_(train_data.values[:, 0])
        self.features_test = np.float_(test_data.values[:, self.Firstnm+1:self.Lastnm+1])
        self.labels_test = np.int_(test_data.values[:, 0])
        self.x_train = Validation_x_train[:, self.Firstnm:self.Lastnm]
        self.x_test = Validation_x_test[:, self.Firstnm:self.Lastnm]
        self.y_train = Validation_y_train
        self.y_test = Validation_y_test
        ################




        '''
        # first order
        self.features_train = np.diff(np.float_(train_data.values[:, self.Firstnm + 1:self.Lastnm + 1]))

        self.features_test = np.diff(np.float_(test_data.values[:, self.Firstnm + 1:self.Lastnm + 1]))

        self.x_train = np.diff(Validation_x_train[:, self.Firstnm:self.Lastnm])
        self.x_test = np.diff(Validation_x_test[:, self.Firstnm:self.Lastnm])
        #

        # PCA
        self.pca=PCA(n_components=4)
        self.pca.fit((np.float_(train_data.values[:, self.Firstnm + 1:self.Lastnm + 1])))
        self.features_train =   self.pca.transform((np.float_(train_data.values[:, self.Firstnm + 1:self.Lastnm + 1])))
        self.pca.fit((np.float_(test_data.values[:, self.Firstnm + 1:self.Lastnm + 1])))
        self.features_test = self.pca.transform((np.float_(test_data.values[:, self.Firstnm + 1:self.Lastnm + 1])))
        self.pca.fit((Validation_x_train[:, self.Firstnm:self.Lastnm]))
        self.x_train = self.pca.transform((Validation_x_train[:, self.Firstnm:self.Lastnm]))
        self.pca.fit((Validation_x_test[:, self.Firstnm:self.Lastnm]))
        self.x_test = self.pca.transform((Validation_x_test[:, self.Firstnm:self.Lastnm]))
        '''

        self.test_path = test_path
        self.path=test_path
        self.x = PatNum
        self.classifier=classifier
    def check_patientnumber(self):
        f = 'Offenbach.csv'#'TNM 2014-2015_Barrett-CA, Offenbach (1).csv'
        line = []
        # with open(f, "rt") as infile:
        #    read = csv.reader(infile)
        #    l=0
        #    for row in read :
        #        line.append(row)
        patientnumber = 0
        linenew = []
        with open(f, newline='') as f:
            reader = csv.reader(f, delimiter=';')
            row1 = next(reader)
            for i, line in enumerate(reader):
                linenew.append(line)
        #print(linenew[1][0])
        for path in self.path:
            file_list = os.listdir(path)
            for inpath in file_list:
                newpath = self.path[0] + '/' + inpath
                file_listnew = os.listdir(newpath)
                FirstLoop = 1
                for file in file_listnew:
                    # run over csv files only
                    if file.endswith(".csv"):
                        # if path=='Magen-CA/2018_10_17_20_30_54/' or path=='Magen-CA/2018_10_17_20_38_29/'or path=='Magen-CA/2018_10_18_16_54_40/'or path=='Magen-CA/2018_10_18_17_05_41/'or path=='Magen-CA/2018_10_18_17_41_23/'or path== 'Magen-CA/2018_10_18_17_44_22/' or path == 'Magen-CA/2018_10_18_17_50_48/' or path=='Magen-CA/2018_10_18_18_03_56/' or path=='Magen-CA/2018_10_18_18_07_12/' or path=='Magen-CA/2018_10_18_18_11_04/' or path=='Magen-CA/2018_10_18_18_15_50/' or path=='Magen-CA/2018_10_18_18_24_55/' or path=='Magen-CA/2018_10_18_18_30_43/' :
                        # if path in file_path_gc:
                        for line in linenew:

                            if line[5] == inpath or line[6] == inpath:
                                # skipping the file with the mean values
                                # skipping the file with the mean values

                                if FirstLoop == 1:
                                    if line[5] == inpath or line[6] == inpath:
                                        if FirstLoop == 1:
                                            if line[6] == inpath:
                                                patientnumber = patientnumber
                                            else:
                                                patientnumber = patientnumber + 1

                                if patientnumber-1== (self.x):
                                            self.test_path=newpath


    def training(self,modeltraining, param_grid):
        cv=StratifiedShuffleSplit(n_splits=10,test_size=0.4, random_state=42)

        scores=['recall','f1']
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        import sklearn
        backend = "threading"
        sklearn.utils.parallel_backend(backend, n_jobs=-1)
        for score in scores:
            print('Tuning hyper-parameters for %s'%score)
            model=GridSearchCV(estimator=modeltraining,param_grid=param_grid,cv=cv,scoring='%s_macro' % score,n_jobs=-1)
            model.fit(self.x_train,self.y_train)
            print('Best parameters set found on development set:')
            print(model.best_params_)
            print('Best CV score set found on development set:')
            print(model.best_score_)
            print('Grid scores on development set:')
            means=model.cv_results_['mean_test_score']
            stds=model.cv_results_['std_test_score']
            for mean, std,params in zip(means,stds,model.cv_results_['params']):
                print('%0.3f (+/-%0.3f) for %r'
                %(mean,std * 2,params))

            #two classes


        #acc, f1score, spec, mcc, sens = blind_test_algorithm.classification(self, model)
        #blind_test_algorithm.visualization(self, model)

        return model



    def classification(self,model):
        model.fit(self.x_train, self.y_train)

        start_time0 = time.time()
        predicted = model.predict(self.features_test)
        runtime0 = time.time()
        print("--- classification %s seconds ---" % (runtime0 - start_time0))
        equal = np.array_equal(self.labels_test, predicted)
        #print(self.labels_test)
        #print(predicted)
        if equal == True:
            print('all equal')
            # conf_mat=1
            #tn, fp, fn, tp = confusion_matrix(self.labels_test, predicted).ravel()
            #specificity = tn / (tn + fp)
            #print(confusion_matrix(self.labels_test, predicted))
            #print(classification_report(self.labels_test, predicted))
            #cm = ConfusionMatrix(self.labels_test, predicted)
            #cm.print_stats()
            sens=np.zeros((1,1))
            spec=np.zeros((1,1))
            acc=np.zeros((1,1))
            mcc=np.zeros((1,1))
            f1score=np.zeros((1,1))
            #cm = ConfusionMatrix(self.labels_test, predicted)
            #cm.print_stats()
            #acc = sklm.accuracy_score(self.labels_test, predicted)
            #tn, fp, fn, tp = confusion_matrix(self.labels_test, predicted).ravel()
            #spec = tn / (tn + fp)
            #sens = tp / (tp + fn)
            #mcc = ((tp * tn) - (fp + fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            #prec = tp / (tp + fp)
            #f1score = 2 * (prec * sens) / (prec + sens)
        else:
            conf_mat = ConfusionMatrix(self.labels_test, predicted)

            #tn, fp, fn, tp = confusion_matrix(self.labels_test, predicted).ravel()
            #specificity = tn / (tn + fp)
            print(confusion_matrix(self.labels_test, predicted))
            print(classification_report(self.labels_test, predicted))

            cm = ConfusionMatrix(self.labels_test, predicted)
            cm.print_stats()
            acc = sklm.accuracy_score(self.labels_test, predicted)
            #f1 = sklm.precision_recall_fscore_support(self.labels_test, predicted)
            #multilabel

            mcm = sklm.multilabel_confusion_matrix(self.labels_test, predicted)
            tn = mcm[:, 0, 0]
            tp = mcm[:, 1, 1]
            fn = mcm[:, 1, 0]
            fp = mcm[:, 0, 1]

            #binary label

            #tn, fp, fn, tp = confusion_matrix(self.labels_test, predicted).ravel()
            spec= tn / (tn + fp)
            sens=tp/(tp+fn)
            mcc=((tp*tn)-(fp+fn))/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
            prec=tp/(tp+fp)
            f1score=2*(prec*sens)/(prec+sens)



        '''
        acc = sklm.accuracy_score(self.labels_test, predicted)
        f1 = sklm.f1_score(self.labels_test, predicted)
        rec = sklm.recall_score(self.labels_test, predicted)
        print("TN :", tn)
        print("TP :", tp)
        print("FN :", fn)
        print("FP :", fp)
        print("testing acuracy for class 1  :  ", acc)
        print("testing F1-Score for class 1 :   ", f1)
        print("testing Specificity for class 1 : ", specificity)
        print("testing Recall for class 1:       ", rec)
        spec_2 = tp / (tp + fn)
        print("testing Specificity for class 2", spec_2)
        print("testing MCC", sklm.matthews_corrcoef(self.labels_test, predicted))
        false_positive_rate, true_positive_rate, thresholds = sklm.roc_curve(self.labels_test, predicted, pos_label=1)
        roc_auc = sklm.auc(false_positive_rate, true_positive_rate)

        plt.title('Receiver Operating Characteristic')
        plt.plot(false_positive_rate, true_positive_rate, 'b',
                 label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 1.2])
        plt.ylim([-0.1, 1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(self.classifier+'1'+str(self.x))

        plt.title('Receiver Operating Characteristic')
        plt.plot(false_positive_rate, true_positive_rate, 'b',
                 label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 1.2])
        plt.ylim([-0.1, 1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(self.classifier+'2'+str(self.x))
        #conf_mat.print_stats()
        '''
        #tn, fp, fn, tp, acc, f1, rec, specificity, spec_2=1,1,1,1,1,1,1,1,1
        f1score = np.array2string(f1score, formatter={'float_kind': lambda x: " % 2f" % x})
        sens = np.array2string(sens, formatter={'float_kind': lambda x: " % 2f" % x})
        spec = np.array2string(spec, formatter={'float_kind': lambda x: " % 2f" % x})
        mcc = np.array2string(mcc, formatter={'float_kind': lambda x: " % 2f" % x})
        acc = np.array2string(acc, formatter={'float_kind': lambda x: " % 2f" % x})
        return acc,f1score,spec,mcc,sens  # conf_mat#conf_mat.stats_class
    def visualization(self,model):
        filnr=0
        file_list = os.listdir(self.test_path)
        # for inpath in file_list:
        #     newpath=self.path[0]+'/'+inpath
        #     file_listnew = os.listdir(newpath)
        for file in file_list:
            if file.endswith(".dat"):
                with open(os.path.join(self.test_path, file), newline='')  as filex:
                    filename=filex.name
                    start_time = time.time()
                    ###################################
                    #SNV
                    #learn_data, y = Cube_Read(
                    #    filename, wavearea=100, Firstnm=0, Lastnm=100).cube_snv_matrix()

                    #########################################
                    # reflectance
                    learn_data, y = Cube_Read(
                            filename, wavearea=100, Firstnm=0, Lastnm=100).cube_matrix_learn()

                    ##############################


                    learn_data=learn_data[:,0:61]
                    #####################################
                    #Dif
                    #learn_data=np.diff(learn_data)
                    ###################################
                    #PCA
                    #learn_data=self.pca.fit(learn_data)
                    #learn_data = self.pca.transform(learn_data)
                    ##########################################

                    # #Gaussian#
                    # ####################################################
                    # i_wave_length = 122
                    # data_2d = learn_data.reshape((640,int(y/640),61))
                    # spectrum_data_1D = learn_data[:,:]#spectrum_data.reshape((640 * pixel, 61))
                    # #
                    # from skimage.filters import gaussian
                    # sigmas = [1]
                    # n_features = 62
                    # n_features = data_2d.shape[2]
                    # new_data = np.zeros([640 * int(y/640), len(sigmas) * n_features])
                    # #
                    # for s_i, s in enumerate(sigmas):
                    #     for c_i in range(n_features):
                    #         new_data[..., s_i * n_features + c_i] = gaussian(data_2d[..., c_i], sigma=s).reshape(-1)
                    #
                    # new_data = np.column_stack((spectrum_data_1D, new_data))
                    # learn_data=new_data
                    ################################################################
                    ### Gaussian PCA##############
                    i_wave_length = 65
                    data_2d = learn_data.reshape((640, int(y / 640), 61))
                    spectrum_data_1D = learn_data[:, :]  # spectrum_data.reshape((640 * pixel, 61))
                    #
                    from skimage.filters import gaussian
                    sigmas = [1]
                    n_features = 62
                    n_features = data_2d.shape[2]
                    new_data = np.zeros([640 * int(y / 640), len(sigmas) * n_features])
                    #
                    for s_i, s in enumerate(sigmas):
                        for c_i in range(n_features):
                            new_data[..., s_i * n_features + c_i] = gaussian(data_2d[..., c_i], sigma=s).reshape(-1)

                    pca = PCA(n_components=4)
                    pca.fit((np.float_(
                        new_data[:, :])))
                    new_data_transform = pca.transform((np.float_(
                        new_data[:, :])))

                    new_data = np.column_stack((spectrum_data_1D, new_data_transform))


                    learn_data = new_data

                    #################################################


                    pixely=y

                    #spectrum_data=learn_data[:]

                    #########################
                    #Golay Filter
                    #learn_data = savgol_filter(learn_data, 9, 2, mode='nearest', axis=1)
                    ####################################
                    #learn_data = learn_data[:,0 :61]


                    prediction_time = time.time()
                    prediction = model.predict(learn_data)
                    print("--- prediction %s seconds ---" % (prediction_time - start_time))
                    pixely=int(pixely/640)

                    #spectrum_data, pixely = Cube_Read(filex).cube_matrix()
                    abc = prediction.reshape((640, pixely))
                    #spectrum_data = spectrum_data.reshape((640, pixely,100))

                    #spectrum_datanew, ynew=Cube_Read(
                    #    filename, wavearea=100, Firstnm=0, Lastnm=100).cube_matrix()
                    #spectrum_datanew=spectrum_datanew.reshape((640, ynew, 100))





                    colors = np.zeros((640, pixely, 4))
                    for i in range(pixely):
                        for j in range(640):
                            if abc[j, i] == 1:
                                colors[j, i, :] = [0, 255, 0, 0.6]  # green EAC
                            if abc[j, i] == 2:
                                colors[j, i, :] = [0, 0, 255, 0.6]  # blue plattenepithel
                            if abc[j, i] == 3:
                                colors[j, i, :] = [255, 255, 0,0.6]  # yellow stroma
                            if abc[j, i] == 4:
                                colors[j, i, :] = [0, 0, 0, 1]  # black background
                            if abc[j, i] == 0:
                                colors[j, i, :] = [0, 0, 0, 0.6]  # black background

                    plot = cube(
                       filename,wavearea=100, Firstnm=0, Lastnm=100).cube_plot()

                    colors = np.rot90(colors)
                    plt.imshow(colors)
                    filnr=filnr+1
                    plt.savefig(str(self.x) + 'Results' + file+str(filnr)+self.classifier+'allClasses_Reflec_PCA_GaussMLP.jpg',dpi=1200) ##
                    np.save(str(self.x) + 'Results' + file+str(filnr)+self.classifier+'allClasses_Reflect_PCA_GaussMLP',colors)


    def RandomFor(self):
        from sklearn.ensemble.forest import RandomForestClassifier
        nr.seed(14)
        model = RandomForestClassifier(criterion='entropy', n_estimators=80)
        print(' Random Forrest Tuning ')
        acc, f1score, spec, mcc, sens = blind_test_algorithm.classification(self, model)
        #blind_test_algorithm.check_patientnumber(self)
        #blind_test_algorithm.visualization(self, model)
        return acc, f1score, spec, mcc, sens

    def LogisticReg (self):
        print(
            '_----------------------------------------------------------------------------------------------_')
        print('Logistic REg')

        model = LogisticRegression(max_iter=1000000, C=1000,  solver='lbfgs')
        modeltraining=LogisticRegression()
        param_grid={'C':[0.001,0.01,1,10,100,1000]}
        #filename = 'MLP' + self.classifier + self.test_path[0] + '.pkl'
        #model = blind_test_algorithm.training(self, modeltraining=modeltraining,param_grid=param_grid)
        acc, f1score, spec, mcc, sens = blind_test_algorithm.classification(self, model)
        blind_test_algorithm.check_patientnumber(self)
        blind_test_algorithm.visualization(self, model)
        print(
            '_----------------------------------------------------------------------------------------------_')
        #f1=np.array2string(f1, formatter={'float_kind':lambda x: "%.2f" % x})
        #spec=np.array2string(spec, formatter={'float_kind':lambda x: "%.2f" % x})

        return acc, f1score, spec, mcc, sens

    def LinearSVC(self):
        print('SVM')
        from sklearn.ensemble import BaggingClassifier
        svmmodel = svm.LinearSVC(max_iter=1000000, C=1)
        import sklearn
        backend = "threading"
        sklearn.utils.parallel_backend(backend, n_jobs=-1)
        model = BaggingClassifier(base_estimator=svmmodel, max_samples=0.1, max_features=0.1,n_jobs=-1)
        acc, f1score, spec, mcc, sens = blind_test_algorithm.classification(self, model)
        #filename = 'MLP'+self.classifier+self.test_path[0]+'.pkl'
        #joblib.dump(model, filename)
        blind_test_algorithm.check_patientnumber(self)
        blind_test_algorithm.visualization(self, model)
        '''
        model.fit(self.x_train, self.y_train)
        start_time0 = time.time()
        predicted = model.predict(self.features_test)
        runtime0 = time.time()
        print("--- classification %s seconds ---" % (runtime0-start_time0))
        equal = np.array_equal(self.labels_test, predicted)
        print(self.labels_test)
        print(predicted)
        if equal == True:
            print('all equal' )
            #conf_mat=1
            tn, fp, fn, tp = confusion_matrix(self.labels_test, predicted).ravel()
            specificity = tn / (tn + fp)
            print(confusion_matrix(self.labels_test, predicted))
            print(classification_report(self.labels_test, predicted))

        else:
            #conf_mat = ConfusionMatrix(self.labels_test, predicted)
            tn, fp, fn, tp = confusion_matrix(self.labels_test, predicted).ravel()
            specificity = tn / (tn + fp)
            print (confusion_matrix(self.labels_test, predicted))
            print(classification_report(self.labels_test, predicted))
        #
        '''

        #f1 = np.array2string(f1, formatter={'float_kind': lambda x: "%.2f" % x})
        #spec = np.array2string(spec, formatter={'float_kind': lambda x: "%.2f" % x})

        return acc,f1score,spec,mcc,sens

    def kNN(self):
        print(' KNN ')
        from sklearn.ensemble import BaggingClassifier
        svmmodel = svm.LinearSVC(max_iter=1000000, C=1)
        import sklearn
        backend = "threading"
        sklearn.utils.parallel_backend(backend, n_jobs=-1)
        model = neighbors.KNeighborsClassifier(metric='manhattan', algorithm='kd_tree', n_jobs=-1)
        acc, f1score, spec, mcc, sens = blind_test_algorithm.classification(self, model)
        blind_test_algorithm.check_patientnumber(self)
        blind_test_algorithm.visualization(self, model)
        return acc, f1score, spec, mcc, sens
    
    def neural_network(self):
        from sklearn.neural_network import MLPClassifier
        print('Test MLPClassifier')
        model = MLPClassifier(hidden_layer_sizes=(32, 16), activation='tanh',solver='adam', random_state=1, max_iter=10000)

        #test
        #model = MLPClassifier(hidden_layer_sizes=(32, 16), activation='tanh', solver='adam', random_state=1,
        #                      max_iter=1)
        acc, f1score, spec, mcc, sens = blind_test_algorithm.classification(self, model)
        blind_test_algorithm.visualization(self, model)
        return acc, f1score, spec, mcc, sens  # conf_mat#conf_mat.stats_class

    def SVM_Training(self):

        param_grid = {'C': [1, 100, 1000], 'kernel': ['linear','poly']}
        model=svm.SVC(max_iter=10000)
        test = blind_test_algorithm.training(self, modeltraining=model,
                                                                     param_grid=param_grid)

        print(neighbors.KNeighborsClassifier().get_params().keys())
        model=neighbors.KNeighborsClassifier()
        param_grid={'metric': ['manhattan','l2','minkowski'], 'algorithm': ['ball_tree','kd_tree'],'n_neighbors':[5,10,20]}
        test = blind_test_algorithm.training(self, modeltraining=model,
                                             param_grid=param_grid)
        #print(neighbors.KNeighborsClassifier().get_params().keys())
        model=RandomForestClassifier()
        param_grid={'criterion': ['gini','entropy'], 'n_estimators': [80,40,100]}
        test = blind_test_algorithm.training(self, modeltraining=model,
                                             param_grid=param_grid)
        model=LogisticRegression(max_iter=1000)
        param_grid={'solver': ['gini','entropy'], 'C': [1,10,100,1000],'solver':['lbfgs','saga']}
        test = blind_test_algorithm.training(self, modeltraining=model,
                                             param_grid=param_grid)


        from sklearn.neural_network import MLPClassifier
        model=MLPClassifier(hidden_layer_sizes=(32, 16),  random_state=1, max_iter=10000)
        param_grid={'activation': ['tanh'], 'solver':['lbfgs','sgd']}
        test = blind_test_algorithm.training(self, modeltraining=model,
                                             param_grid=param_grid)

        test = blind_test_algorithm.training(self, modeltraining=model,
                                                                      param_grid=param_grid)




class test_algorithm(object):
    def __init__(self, train_data, test_data, Validation_x_train, Validation_y_train, Validation_x_test,
                 Validation_y_test, test_path, PatNum, classifier):
        self.Lastnm = 60
        self.Firstnm = 0
        self.WaveArea = self.Lastnm - self.Firstnm
        print(
            'Wavelength area from :' + str(self.Firstnm * 5 + 500) + 'nm   till :' + str(self.Lastnm * 5 + 500) + 'nm')
        self.features_train = train_data.values[:, self.Firstnm:self.Lastnm]
        self.labels_train = train_data.values[:, 0]
        self.features_test = test_data.values[:, self.Firstnm:self.Lastnm]
        self.labels_test = test_data.values[:, 0]
        self.x_train = Validation_x_train[:, self.Firstnm:self.Lastnm]
        self.x_test = Validation_x_test[:, self.Firstnm:self.Lastnm]
        self.y_train = Validation_y_train
        self.y_test = Validation_y_test
        self.test_path = test_path
        self.path = test_path
        self.x = PatNum
        self.classifier = classifier


    def RandomFor(self):
        from sklearn.ensemble.forest import RandomForestClassifier
        nr.seed(14)
        model = RandomForestClassifier(criterion='entropy', n_estimators=80)
        print(' Random Forrest Tuning ')
        tn, fp, fn, tp, acc, f1, rec, specificity, spec_2, model = test_algorithm.classification(self, model)
        blind_test_algorithm.visualization(self,model)
        return tn, fp, fn, tp, acc, f1, rec, specificity, spec_2

    def LogisticReg(self):
        print(
            '_----------------------------------------------------------------------------------------------_')
        print('Logistic REg')

        model = LogisticRegression(max_iter=1000000, C=1000, solver='lbfgs')
        modeltraining = LogisticRegression()
        param_grid = {'C': [0.001, 0.01, 1, 10, 100, 1000]}
        # filename = 'MLP' + self.classifier + self.test_path[0] + '.pkl'
        # model = blind_test_algorithm.training(self, modeltraining=modeltraining,param_grid=param_grid)
        acc, f1score, spec, mcc, sens = blind_test_algorithm.classification(self, model)

        #test_algorithm.visualization(self, model)
        print(
            '_----------------------------------------------------------------------------------------------_')
        #f1 = np.array2string(f1, formatter={'float_kind': lambda x: "%.2f" % x})
        #spec = np.array2string(spec, formatter={'float_kind': lambda x: "%.2f" % x})
        return  acc,f1score,spec,mcc,sens

    def LinearSVC(self):
        print('SVM')
        from sklearn.ensemble import BaggingClassifier
        svmmodel = svm.LinearSVC(max_iter=1000000, C=1)
        model = BaggingClassifier(base_estimator=svmmodel, max_samples=0.5, max_features=0.5)
        #tn, fp, fn, tp, acc, f1, rec, specificity, spec_2 = test_algorithm.classification(self, model)
        # filename = 'MLP'+self.classifier+self.test_path[0]+'.pkl'
        # joblib.dump(model, filename)

        acc, f1score, spec, mcc, sens = blind_test_algorithm.classification(self, model)

        # test_algorithm.visualization(self, model)
        print(
            '_----------------------------------------------------------------------------------------------_')
        #f1 = np.array2string(f1, formatter={'float_kind': lambda x: "%.2f" % x})
        #spec = np.array2string(spec, formatter={'float_kind': lambda x: "%.2f" % x})
        return acc,f1score,spec,mcc,sens

    def kNN(self):
        print(' KNN ')
        model = neighbors.KNeighborsClassifier(metric='manhattan', algorithm='kd_tree', n_jobs=-1)
        tn, fp, fn, tp, acc, f1, rec, specificity, spec_2 = blind_test_algorithm.classification(self, model)
        # blind_test_algorithm.visualization(self, model)
        return tn, fp, fn, tp, acc, f1, rec, specificity, spec_2  # conf_mat#conf_mat.stats_class

    def neural_network(self):
        from sklearn.neural_network import MLPClassifier
        print('Test MLPClassifier')
        model = MLPClassifier(hidden_layer_sizes=(64,32, 16), activation='tanh', random_state=1, max_iter=10000, solver='adam')
        tn, fp, fn, tp, acc, f1, rec, specificity, spec_2, model = blind_test_algorithm.classification(self, model)
        blind_test_algorithm.visualization(self, model)
        return tn, fp, fn, tp, acc, f1, rec, specificity, spec_2  # conf_mat#conf_mat.stats_class

