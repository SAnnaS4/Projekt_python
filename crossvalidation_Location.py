import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))

from test_train_split_data import train_test
from machine_learning_alg import blind_test_algorithm
import numpy as np
from LoadCSVDataLocation import LoadData
import pandas as pd
import os
import csv
import random

'''

SNV_data= pd.read_csv('data_EAC_platte_stroma_blank.csv')

patientnumber=96

labels = SNV_data.values[:, 0]

data_values = SNV_data.values[:, 2:]

############################################
from sklearn.preprocessing import StandardScaler

data = np.transpose(data_values)
scaler = StandardScaler()
scaler.fit(data)
SNV_Scale = scaler.transform(data)
SNV_Scale = np.transpose(SNV_Scale)
SNV_data.values[:,2:]=SNV_Scale

patients = np.array(SNV_data['patients'])
for i in range(1, 29):
    value = SNV_data.values[patients == i, 2:]
    value = -1 * (np.log10(abs(value)))
    SNV_data.values[patients == i, 2:] = value


SNV_data.replace({'labels': 2}, 0)
SNV_data.replace({'labels': 3}, 1)
SNV_data.replace({'labels': 4}, 1)
SNV_data.replace({'labels': 5}, 1)
EAC=SNV_data
'''

# f = 'zur Klassifizierung TNM 2014-2017_Barrett-CA, Offenbach.csv'  # 'TNM 2014-2015_Barrett-CA, Offenbach (1).csv'
# line = []
# linenew = []
# with open(f, newline='') as f:
#     reader = csv.reader(f, delimiter=';')
#     row1 = next(reader)
#     for i, line in enumerate(reader):
#         linenew.append(line)
#
pathall = ['C:/Users/Anna/Desktop/Masterarbeit/all']
# pathPatient = []
# patientN = 0
# for path in pathall:
#     file_list = os.listdir(path)
#     for inpath in file_list:
#         newpath = pathall[0] + '/' + inpath
#         file_listnew = os.listdir(newpath)
#         #FirstLoop = 1
#         for file in file_listnew:
#             # run over csv files only
#             if file.endswith(".mk2"):
#
#                 for line in linenew:
#
#                     if line[6] == inpath or line[7] == inpath:
#
#                         if line[7] == inpath:
#                             # pathPatient.append(newpath)
#                             testx = 1
#                             patientN = patientN - 1
#                             print(patientN)
#                             print("Inpath:", newpath)
#                             patientN = patientN + 1
#                         else:
#                             pathPatient.append(newpath)
#                             print(patientN)
#                             print("Inpath:", newpath)
#                             patientN = patientN + 1
#                         #FirstLoop = 2

print('##############################################################################################################')
print('EAC- Source Offenbach')
#Daten aus Ordner laden
EAC, count0, count1, count2, count3, count4, count5, patientnumber = LoadData(pathall, groupname0=0, groupname1=0,
                                                                              groupname2=1, groupname3=1, groupname4=2,
                                                                              groupname5=3).Reflectance_alg(style=None)
testname= 'MLP_Reflectance_Spatial_all_classes_630nm_Quotient_530nm_EACStromaOneClass'
# mean_std_plot(EAC).plot_graph(name='Aufnahmeort',number=[count0,count1,count2,count3],type='Absorbance')
number = [count0, count1, count2, count3, count4, count5]
print('All Spectra:' + str(count1 + count2 + count3 + count4 + count0 + count5))
# print('2:' + str(count2))
print('test size of 10 %:' + str(patientnumber * 0.1))
print('##############################################################################################################')
#Testdaten = 10% der Daten
testsize = round(patientnumber * 0.1)
print('Patient Number: ' + str(patientnumber))
#4x Patientennummerarray als Nullen
acc1, acc2, acc3, acc4 = np.zeros(patientnumber), np.zeros(patientnumber), np.zeros(patientnumber), np.zeros(
    patientnumber)

#nochmal aber mehr
spec2, f12, f11, spec1, sens1, mcc2, mcc1, sens2, spec4, f14, f13, spec3, sens3, mcc4, mcc3, sens4 = np.zeros(
    patientnumber), np.zeros(patientnumber), np.zeros(patientnumber), np.zeros(patientnumber), np.zeros(
    patientnumber), np.zeros(patientnumber), np.zeros(patientnumber), np.zeros(patientnumber), np.zeros(
    patientnumber), np.zeros(patientnumber), np.zeros(patientnumber), np.zeros(patientnumber), np.zeros(
    patientnumber), np.zeros(patientnumber), np.zeros(patientnumber), np.zeros(patientnumber)
acc1, sens1, spec1, mcc1, f11 = ['Accuracy'], ['Sens'], ['Spec'], ['MCC'], ['F1']

########################################
 # Training
# ########################################
#
i = 1
patients = np.array(EAC['patients'])
patients = np.int_(patients)
datastest = EAC.values[patients >= i, 3:]
labelstest = EAC.values[patients >= i, 0]
Patienttestpath = EAC.values[patients >= i, 2]
#
EAC_test = pd.DataFrame(datastest)
EAC_test.insert(loc=0, column='label', value=labelstest)
# #
# EAC_test.columns = ['label', '500nm', '505nm']
# # '''
# # ### gaussian ###
# # EAC_test.columns = ['label', '500nm', '505nm', '510nm', '515nm', '520nm',
# #'525nm', '530nm', '535nm', '540nm',
# #                            '545nm', '550nm', '555nm', '560nm', '565nm', '570nm', '575nm',
# #                            '580nm', '585nm', '590nm',
# #                            '595nm',
# #                            '600nm', '605nm', '610nm', '615nm', '620nm', '625nm', '630nm',
# #                            '635nm', '640nm', '645nm',
# #                            '650nm', '655nm', '660nm', '665nm', '670nm', '675nm', '680nm',
# #                            '685nm', '690nm', '695nm',
# #                            '700nm', '705nm', '710nm', '715nm', '720nm', '725nm', '730nm',
# #                            '735nm', '740nm', '745nm',
# #                            '750nm', '755nm', '760nm', '765nm', '770nm', '775nm', '780nm',
# #                            '785nm', '790nm', '795nm',
# #                            '800nm','500nm', '505nm', '510nm', '515nm', '520nm',
# #                            '525nm', '530nm', '535nm', '540nm',
# #                            '545nm', '550nm', '555nm', '560nm', '565nm', '570nm', '575nm',
# #                            '580nm', '585nm', '590nm',
# #                            '595nm',
# #                            '600nm', '605nm', '610nm', '615nm', '620nm', '625nm', '630nm',
# #                            '635nm', '640nm', '645nm',
# #                            '650nm', '655nm', '660nm', '665nm', '670nm', '675nm', '680nm',
# #                            '685nm', '690nm', '695nm',
# #                            '700nm', '705nm', '710nm', '715nm', '720nm', '725nm', '730nm',
# #                            '735nm', '740nm', '745nm',
# #                            '750nm', '755nm', '760nm', '765nm', '770nm', '775nm', '780nm',
# #                            '785nm', '790nm', '795nm',
# #                            '800nm']
# # '''
# #
# ### gaussian pca###
#
#
EAC_test.columns = ['label', '500nm', '505nm', '510nm', '515nm', '520nm',
                    '525nm', '530nm', '535nm', '540nm',
                    '545nm', '550nm', '555nm', '560nm', '565nm', '570nm', '575nm',
                    '580nm', '585nm', '590nm',
                    '595nm',
                    '600nm', '605nm', '610nm', '615nm', '620nm', '625nm', '630nm',
                    '635nm', '640nm', '645nm',
                    '650nm', '655nm', '660nm', '665nm', '670nm', '675nm', '680nm',
                    '685nm', '690nm', '695nm',
                    '700nm', '705nm', '710nm', '715nm', '720nm', '725nm', '730nm',
                    '735nm', '740nm', '745nm',
                    '750nm', '755nm', '760nm', '765nm', '770nm', '775nm', '780nm',
                    '785nm', '790nm', '795nm',
                    '800nm', 'c1', 'c2', 'c3', 'c4']
print('Patienttestset')
labels = np.array(EAC_test['label'])
labels = np.int_(labels)
# # temp_features_1 = EAC_test.values[labels == 0, 1:]
temp_features_1 = EAC_test.values[labels == 1, 1:]
temp_features_2 = EAC_test.values[labels == 2, 1:]
temp_features_3 = EAC_test.values[labels == 3, 1:]
temp_features_4 = EAC_test.values[labels == 4, 1:]
#
# Undersampling
Validation_x_train, Validation_y_train, Validation_x_test, Validation_y_test = train_test(
      EAC_test).div_set_underbalanced_2_class()
# # Oversampling
# Validation_x_train, Validation_y_train, Validation_x_test, Validation_y_test = train_test(
#     EAC_test).div_set_balanced_3classes()
#
# acc1[i - 1], f11, spec1, mcc1, sens1 = blind_test_algorithm(EAC_test, EAC_test, Validation_x_train,
#                                                             Validation_y_train,
#                                                             Validation_x_test,
#                                                             Validation_y_test, pathall, i,
#                                                             classifier='Training').SVM_Training()

acc1[i - 1], f11, spec1, mcc1, sens1 = blind_test_algorithm(EAC_test, EAC_test, Validation_x_train,
                                                            Validation_y_train,
                                                            Validation_x_test,
                                                            Validation_y_test, pathall, i,
                                                            classifier='Training').modelnameTraining(Processindex='UnderBalanced MLP')

# #################################################
# End
##################################################

for i in range(2, 80):
#for i in range(79, 80):
    # SG_data = savgol_filter(data.values[:, 1:], 9, 2, mode='nearest', axis=1)
    # SG_data1 = pd.DataFrame(SG_data)
    print(
        '##############################################################################################################')
    print('Patientnumber:  ' + str(i))
    patients = np.array(EAC['patients'])
    patients = np.int_(patients)
    datastest = EAC.values[patients == i, 3:]

    labelstest = EAC.values[patients == i, 0]

    Patienttestpath = EAC.values[patients == i, 2]
    print("Labelstest " + labelstest)
    if Patienttestpath.size != 0:
        print(Patienttestpath[0])
        # r=1
        patienttest = [i]
        # ind_i = 1
        # while ind_i == 1:
        #
        #     randomNumber = random.sample(range(i + 1, patientnumber), testsize)
        #     print(randomNumber)
        #     randomNumber = np.asarray(randomNumber)
        #     ind_i = np.where(randomNumber == i + 1)
        #     print(ind_i)
        #     print(ind_i[0].size)
        #     if ind_i[0].size == 1:
        #         ind_i = 2
        #     else:
        #         ind_i = 1
        #
        # for i_rand in range (0,testsize-1):
        #     datastest = np.vstack((EAC.values[patients == randomNumber[i_rand], 2:], datastest))
        #     labelstest = np.append(EAC.values[patients == randomNumber[i_rand], 0], labelstest)

        # for i_testsize in range(testsize - 1):
        #
        #
        #     if (patientnumber-i)>testsize:
        #         #randomNumber=random.sample(range(0,patientnumber),testsize)
        #
        #         randomNumber= random.randint(i,patientnumber)
        #         datastest = np.vstack((EAC.values[patients == randomNumber, 2:], datastest))
        #         labelstest = np.append(EAC.values[patients == randomNumber, 0], labelstest)
        #     else:
        #         randomNumber = random.randint(0, i)
        #         datastest = np.vstack((EAC.values[patients == randomNumber, 2:], datastest))
        #         labelstest = np.append(EAC.values[patients == randomNumber, 0], labelstest)
        #     patienttest.append(randomNumber)
        #     i_testsize=i_testsize+1

        print('Patienttest: ')
        print(*patienttest)
        EAC_test = pd.DataFrame(datastest)
        EAC_test.insert(loc=0, column='label', value=labelstest)

        # EAC_test.columns = ['label', '500nm', '505nm']
        ##### gaussian ####
        # EAC_test.columns = ['label', '500nm', '505nm', '510nm', '515nm', '520nm', '525nm', '530nm', '535nm',
        #                     '540nm',
        #                     '545nm', '550nm', '555nm', '560nm', '565nm', '570nm', '575nm', '580nm', '585nm', '590nm',
        #                     '595nm',
        #                     '600nm', '605nm', '610nm', '615nm', '620nm', '625nm', '630nm', '635nm', '640nm', '645nm',
        #                     '650nm', '655nm', '660nm', '665nm', '670nm', '675nm', '680nm', '685nm', '690nm', '695nm',
        #                     '700nm', '705nm', '710nm', '715nm', '720nm', '725nm', '730nm', '735nm', '740nm', '745nm',
        #                     '750nm', '755nm', '760nm', '765nm', '770nm', '775nm', '780nm', '785nm', '790nm', '795nm',
        #                     '800nm', '500nm', '505nm', '510nm', '515nm', '520nm', '525nm', '530nm', '535nm',
        #                     '540nm',
        #                     '545nm', '550nm', '555nm', '560nm', '565nm', '570nm', '575nm', '580nm', '585nm', '590nm',
        #                     '595nm',
        #                     '600nm', '605nm', '610nm', '615nm', '620nm', '625nm', '630nm', '635nm', '640nm', '645nm',
        #                     '650nm', '655nm', '660nm', '665nm', '670nm', '675nm', '680nm', '685nm', '690nm', '695nm',
        #                     '700nm', '705nm', '710nm', '715nm', '720nm', '725nm', '730nm', '735nm', '740nm', '745nm',
        #                     '750nm', '755nm', '760nm', '765nm', '770nm', '775nm', '780nm', '785nm', '790nm', '795nm',
        #                     '800nm']  # , '805nm', '810nm', '815nm', '820nm', '825nm', '830nm', '835nm', '840nm', '845nm',
        # # '850nm', '855nm', '860nm', '865nm', '870nm', '875nm', '880nm', '885nm', '890nm', '895nm',
        # # '900nm', '905nm', '910nm', '915nm', '920nm', '925nm', '930nm', '935nm', '940nm', '945nm',
        # # '950nm', '955nm', '960nm', '965nm', '970nm', '975nm', '980nm', '985nm', '990nm', '995nm', ]
        ### gaussian pca ###

        EAC_test.columns = ['label', '500nm', '505nm', '510nm', '515nm', '520nm',
                            '525nm', '530nm', '535nm', '540nm',
                            '545nm', '550nm', '555nm', '560nm', '565nm', '570nm', '575nm',
                            '580nm', '585nm', '590nm',
                            '595nm',
                            '600nm', '605nm', '610nm', '615nm', '620nm', '625nm', '630nm',
                            '635nm', '640nm', '645nm',
                            '650nm', '655nm', '660nm', '665nm', '670nm', '675nm', '680nm',
                            '685nm', '690nm', '695nm',
                            '700nm', '705nm', '710nm', '715nm', '720nm', '725nm', '730nm',
                            '735nm', '740nm', '745nm',
                            '750nm', '755nm', '760nm', '765nm', '770nm', '775nm', '780nm',
                            '785nm', '790nm', '795nm',
                            '800nm', 'c1', 'c2', 'c3', 'c4']
    print('Patienttestset')

    labels = np.array(EAC_test['label'])
    labels = np.int_(labels)
    # temp_features_1 = EAC_test.values[labels == 0, 1:]
    temp_features_1 = EAC_test.values[labels == 1, 1:]
    temp_features_2 = EAC_test.values[labels == 2, 1:]
    temp_features_3 = EAC_test.values[labels == 3, 1:]
    temp_features_4 = EAC_test.values[labels == 4, 1:]
    # temp_features_6 = EAC_test.values[labels == 5, 1:]

    # print("Dysplasie - ", temp_features_1.shape[0])
    # print("Metaplasie - ", temp_features_2.shape[0])
    print("EAC - ", temp_features_1.shape[0])
    print("Plattenepithel - ", temp_features_2.shape[0])
    print("Stroma - ", temp_features_3.shape[0])
    print("Blank - ", temp_features_4.shape[0])
    sum_all_test = temp_features_2.shape[0] + temp_features_1.shape[0] + temp_features_3.shape[0] + \
                   temp_features_4.shape[0]

    if sum_all_test != 0 and i !=78:
        # datastrain_first = EAC.values[:, 2:]
        # labelstrain_first = EAC.values[:, 0]
        datastrain = EAC.values[patients != i, 3:]
        labelstrain = EAC.values[patients != i, 0]

        EAC_train = pd.DataFrame(datastrain)
        EAC_train.insert(loc=0, column='label', value=labelstrain)

        # EAC_train.columns = ['label', '500nm', '505nm']
        # #########################################################
        # ### gaussian ###
        # EAC_train.columns = ['label', '500nm', '505nm', '510nm', '515nm', '520nm', '525nm', '530nm', '535nm',
        #                      '540nm',
        #                      '545nm', '550nm', '555nm', '560nm', '565nm', '570nm', '575nm', '580nm', '585nm', '590nm',
        #                      '595nm',
        #                      '600nm', '605nm', '610nm', '615nm', '620nm', '625nm', '630nm', '635nm', '640nm', '645nm',
        #                      '650nm', '655nm', '660nm', '665nm', '670nm', '675nm', '680nm', '685nm', '690nm', '695nm',
        #                      '700nm', '705nm', '710nm', '715nm', '720nm', '725nm', '730nm', '735nm', '740nm', '745nm',
        #                      '750nm', '755nm', '760nm', '765nm', '770nm', '775nm', '780nm', '785nm', '790nm', '795nm',
        #                      '800nm', '500nm', '505nm', '510nm', '515nm', '520nm', '525nm', '530nm', '535nm',
        #                      '540nm',
        #                      '545nm', '550nm', '555nm', '560nm', '565nm', '570nm', '575nm', '580nm', '585nm', '590nm',
        #                      '595nm',
        #                      '600nm', '605nm', '610nm', '615nm', '620nm', '625nm', '630nm', '635nm', '640nm', '645nm',
        #                      '650nm', '655nm', '660nm', '665nm', '670nm', '675nm', '680nm', '685nm', '690nm', '695nm',
        #                      '700nm', '705nm', '710nm', '715nm', '720nm', '725nm', '730nm', '735nm', '740nm', '745nm',
        #                      '750nm', '755nm', '760nm', '765nm', '770nm', '775nm', '780nm', '785nm', '790nm', '795nm',
        #                      '800nm']  # , '805nm', '810nm', '815nm', '820nm', '825nm', '830nm', '835nm', '840nm', '845nm',
        # # '850nm', '855nm', '860nm', '865nm', '870nm', '875nm', '880nm', '885nm', '890nm', '895nm',
        # # '900nm', '905nm', '910nm', '915nm', '920nm', '925nm', '930nm', '935nm', '940nm', '945nm',
        # # '950nm', '955nm', '960nm', '965nm', '970nm', '975nm', '980nm', '985nm', '990nm', '995nm', ]

        #############################################################################
        ### gaussian pca ###
        EAC_train.columns = ['label', '500nm', '505nm', '510nm', '515nm', '520nm',
                             '525nm', '530nm', '535nm', '540nm',
                             '545nm', '550nm', '555nm', '560nm', '565nm', '570nm', '575nm',
                             '580nm', '585nm', '590nm',
                             '595nm',
                             '600nm', '605nm', '610nm', '615nm', '620nm', '625nm', '630nm',
                             '635nm', '640nm', '645nm',
                             '650nm', '655nm', '660nm', '665nm', '670nm', '675nm', '680nm',
                             '685nm', '690nm', '695nm',
                             '700nm', '705nm', '710nm', '715nm', '720nm', '725nm', '730nm',
                             '735nm', '740nm', '745nm',
                             '750nm', '755nm', '760nm', '765nm', '770nm', '775nm', '780nm',
                             '785nm', '790nm', '795nm',
                             '800nm', 'c1', 'c2', 'c3', 'c4']

        ####################################################################
        print('Patienttrainset')
        labels = np.array(EAC_train['label'])
        labels = np.int_(labels)
        # temp_features_1 = EAC_train.values[labels == 0, 1:]
        temp_features_2 = EAC_train.values[labels == 1, 1:]
        temp_features_3 = EAC_train.values[labels == 2, 1:]
        temp_features_4 = EAC_train.values[labels == 3, 1:]
        temp_features_5 = EAC_train.values[labels == 4, 1:]
        # temp_features_6 = EAC_train.values[labels == 5, 1:]

        # print("Dysplasie - ", temp_features_1.shape[0])
        # print("Metaplasie - ", temp_features_2.shape[0])
        print("EAC - ", temp_features_2.shape[0])
        print("Plattenepithel - ", temp_features_3.shape[0])
        print("Stroma - ", temp_features_4.shape[0])
        print("Blank - ", temp_features_5.shape[0])
        Validation_x_train, Validation_y_train, Validation_x_test, Validation_y_test = train_test(
            EAC_train).div_set_balanced_real3classes()

        # blind_test_algorithm(train_data, test_data, x_train, y_train, x_test,
        #                                    y_test, test_path, PatNum).RandomFor()
        # blind_test_algorithm(train_data, test_data, x_train, y_train, x_test,
        #                     y_test, test_path, PatNum, classifier='LogisticRegression').LogisticReg()
        # blind_test_algorithm(train_data, test_data, x_train, y_train, x_test,
        #                    y_test, test_path, PatNum, classifier='SVM').LinearSVC()

        # acc1[i - 1], f11[i - 1],spec1[i - 1],mcc1[i - 1],sens1[i - 1]= blind_test_algorithm(EAC_train, EAC_test, Validation_x_train, Validation_y_train,
        #                                                  Validation_x_test,
        #                                                Validation_y_test, pathall, i,
        #                                                      classifier='Logistic Regression').LogisticReg()

        # acc3[i - 1], f13[i - 1], spec3[i - 1], mcc3[i - 1], sens3[i - 1] = blind_test_algorithm(EAC_train, EAC_test, Validation_x_train,
        #                                                            Validation_y_train,
        #                                                            Validation_x_test,
        #                                                            Validation_y_test, pathall, i,
        #                                                            classifier='Random Forrest').RandomFor()

        # acc4[i - 1], f14, spec4, mcc4, sens4 = blind_test_algorithm(EAC_train, EAC_test, Validation_x_train,
        #                                                            Validation_y_train,
        #                                                            Validation_x_test,
        #                                                            Validation_y_test, pathall, i,
        #                                                            classifier='knn').kNN()

        # acc2[i - 1], f12,spec2,mcc2,sens2 = blind_test_algorithm(EAC_train, EAC_test, Validation_x_train, Validation_y_train,
        #                                                  Validation_x_test,
        #                                                Validation_y_test, pathall, i,
        #                                                  classifier='Linear SVM').LinearSVC()
        # acc, f1,spec,mcc,sens= blind_test_algorithm(EAC_train, EAC_test, Validation_x_train, Validation_y_train,
        #                                              Validation_x_test,
        #                                            Validation_y_test, Patienttestpath[0], i,
        #                                                  classifier='MlP_2classes_32_16_DIFF').neural_network()
        acc, f1, spec, mcc, sens = blind_test_algorithm(EAC_train, EAC_test, Validation_x_train, Validation_y_train,
                                                        Validation_x_test,
                                                        Validation_y_test, Patienttestpath[0], i,
                                                        classifier=testname).neural_network()
        f11.append((f1))
        # f12.append((f12))
        sens1.append((sens))
        spec1.append((spec))
        mcc1.append((mcc))
        acc1.append((acc))
        # sens2.append((sens2))
        # spec2.append((spec2))
        # mcc2.append((mcc2))
        # f13.append((f13))
        # f14.append((f14))
        # sens3.append((sens3))
        # spec3.append((spec3))
        # mcc3.append((mcc3))
        # sens4.append((sens4))
        # spec4.append((spec4))
        # mcc4.append((mcc4))

        # tn1[ i - 1], fp1[i - 1], fn1[i - 1], tp1[i - 1], acc1[i - 1], f11[i - 1], rec1[
        #     i - 1], \
        # spec1[
        #     i - 1], spec_21[i - 1] = blind_test_algorithm(EAC_train, EAC_test, Validation_x_train, Validation_y_train, Validation_x_test,
        #                                                           Validation_y_test, 'test', i,
        #                                                           classifier='Logistic').LogisticReg()
        # tn2[i - 1], fp2[i - 1], fn2[i - 1], tp2[i - 1], acc2[i - 1], f12[i - 1], rec2[
        #    i - 1], \
        #  spec2[
        # i - 1], spec_22[i - 1] = blind_test_algorithm(EAC_train, EAC_test, Validation_x_train, Validation_y_train, Validation_x_test,
        #                                                           Validation_y_test, 'test', i,classifier='KNN').kNN()
    # tn3[i - 1], fp3[i - 1], fn3[i - 1], tp3[i - 1], acc3[i - 1], f13[i - 1], rec3[
    #    i - 1], \
    # spec3[
    #   i - 1], spec_23[i - 1] = blind_test_algorithm(train_data, test_data, x_train, y_train, x_test,
    #                                                         y_test, test_path, PatNum,
    #                                                        classifier='Random_3').RandomFor()
    # tn4[i - 1], fp4[i - 1], fn4[i - 1], tp4[i - 1], acc4[i - 1], f14[i - 1], rec4[
    #   i - 1], \
    # spec4[
    #   i - 1], spec_24[i - 1] = blind_test_algorithm(train_data, test_data, x_train, y_train, x_test,
    #                                                         y_test, test_path, PatNum,
    #                                                         classifier='ANN_3').neural_network()

    # blind_test_algorithm(train_data, test_data, x_train, y_train, x_test,
    #                                     y_test, test_path, PatNum).kNN()
    # blind_test_algorithm(train_data, test_data, x_train, y_train, x_test,
    #                                     y_test, test_path, PatNum).neural_network()
    # print('Load train data')
    # train_data = LoadData(train_path).Reflectance_alg(style='Savitzky_Golay_smoothing')
    # print('Load test data')
    # test_data = LoadData(test_path).Reflectance_alg(style='Savitzky_Golay_smoothing')#SNV_alg(style='Savitzky_Golay_smoothing',numbers='All')
    #
    # x_train, y_train, x_test, y_test = train_test(train_data).div_SMOTE_set()
    #
    # blind_test_algorithm(train_data, test_data, x_train, y_train, x_test,
    #                              y_test, test_path, PatNum).LinearSVC()
    # blind_test_algorithm(train_data, test_data, x_train, y_train, x_test,
    #                                     y_test, test_path, PatNum).RandomFor()
    # blind_test_algorithm(train_data, test_data, x_train, y_train, x_test,
    #                                      y_test, test_path, PatNum).LogisticReg()
    # blind_test_algorithm(train_data, test_data, x_train, y_train, x_test,
    #                                      y_test, test_path, PatNum).kNN()
    # blind_test_algorithm(train_data, test_data, x_train, y_train, x_test,
    #                                      y_test, test_path, PatNum).neural_network()


print('###############################################################################################')
print('Average Performance of each Classifier')
print('MLP')
print(testname)

print('Accuracy')
print(acc1)
print('SENS')
print(sens1)
print('Spec')
print(spec1)
print('MCC')
print(mcc1)
print('F1')
print(f11)

np.savetxt('ACC_LR.csv', acc1, delimiter=',')
np.savetxt('F1_LR.csv', f11, delimiter=',')
np.savetxt('MCC_LR.csv', mcc1, delimiter=',')
np.savetxt('SPEC_LR.csv', spec1, delimiter=',')
np.savetxt('SENS_LR.csv', sens1, delimiter=',')

np.savetxt('ACC_RF.csv', acc3, delimiter=',')
np.savetxt('F1_RF.csv', f13, delimiter=',')
np.savetxt('MCC_RF.csv', mcc3, delimiter=',')
np.savetxt('SPEC_RF.csv', spec3, delimiter=',')
np.savetxt('SENS_RF.csv', sens3, delimiter=',')

np.savetxt('ACC_SVM.csv', acc2, delimiter=',')
np.savetxt('F1_SVM.csv', f12, delimiter=',')
np.savetxt('MCC_SVM.csv', mcc2, delimiter=',')
np.savetxt('SPEC_SVM.csv', spec2, delimiter=',')
np.savetxt('SENS_SVM.csv', sens2, delimiter=',')

np.savetxt('ACC_KNN.csv', acc4, delimiter=',')
np.savetxt('F1_KNN.csv', f14, delimiter=',')
np.savetxt('MCC_KNN.csv', mcc4, delimiter=',')
np.savetxt('SPEC_KNN.csv', spec4, delimiter=',')
np.savetxt('SENS_KNN.csv', sens4, delimiter=',')

print(acc1)
print('f1')
print(f11)
print('rec')
print(sens1)
print('spec')
print(spec1)
print('mcc')
print(mcc1)

'''
print('Average Performance of each Classifier')
print('SVM')

print(acc2)
print('f1')
print(f12)
print('rec')
print(sens2)
print('spec')
print(spec2)
print('mcc' )
print(mcc2)

print('Average Performance of each Classifier')
print('KNN')

print(acc4)
print('f1')
print(f14)
print('rec')
print(sens4)
print('spec')
print(spec4)
print('mcc' )
print(mcc4)
'''

print('Average Performance of each Classifier')
print('RF')
print('acc: ')
print(acc3)
print('f1')
print(f13)
print('rec')
print(sens3)
print('spec')
print(spec3)
print('mcc')
print(mcc3)

#
#
#
#
print('SVM')
print('acc' + str((np.average(acc1))))
print('f1' + str((np.average(f11))))
#print('rec1' + str((np.average(rec1))))
print('spec1' + str((np.average(spec1))))
#print('spec2' + str((np.average(spec_21))))
#print('tn' + str((np.average(tn1))))
#print('fn' + str((np.average(fn1))))
#print('tp' + str((np.average(tp1))))
#print('fp' + str((np.average(fp1))))

print('k-NN')
print('acc' + str((np.average(acc2))))
print('f1' + str((np.average(f12))))
#print('rec1' + str((np.average(rec2))))
print('spec1' + str((np.average(spec2))))
#print('spec2' + str((np.average(spec_22))))
#print('tn' + str((np.average(tn2))))
#print('fn' + str((np.average(fn2))))
#print('tp' + str((np.average(tp2))))
#print('fp' + str((np.average(fp2))))

print('Random Forrest')
print('acc' + str((np.average(acc3))))
print('f1' + str((np.average(f13))))
#print('rec1' + str((np.average(rec3))))
print('spec1' + str((np.average(spec3))))
#print('spec2' + str((np.average(spec_23))))
#print('tn' + str((np.average(tn3))))
#print('fn' + str((np.average(fn3))))
#print('tp' + str((np.average(tp3))))
#print('fp' + str((np.average(fp3))))
print('MLP')
print('acc' + str((np.average(acc4))))
print('f1' + str((np.average(f14))))
#print('rec1' + str((np.average(rec4))))
print('spec1' + str((np.average(spec4))))
#print('spec2' + str((np.average(spec_24))))
#print('tn' + str((np.average(tn4))))
#print('fn' + str((np.average(fn4))))
#print('tp' + str((np.average(tp4))))
#print('fp' + str((np.average(fp4))))
