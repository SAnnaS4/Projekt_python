import pandas.tests.groupby.test_value_counts
import sklearn
import os
print('The scikit-learn version is {}.'.format(sklearn.__version__))

from test_train_split_data import train_test
from machine_learning_alg import blind_test_algorithm
import numpy as np
#from LoadCSVDataLocation import LoadData
import pandas as pd
from pixelSections import LoadData

pathall = ['C:/Users/Anna/Desktop/Masterarbeit/data']
print('##############################################################################################################')
print('EAC- Source Offenbach')

#Daten aus Ordner laden
# EAC, count0, count1, count2, count3, count4, count5, patientnumber = LoadData(pathall, groupname0=0, groupname1=0,
#                                                                               groupname2=1, groupname3=2, groupname4=3,
#                                                                               groupname5=4).Reflectance_alg(style=None)

path = 'C:/Users/Anna/Desktop/Masterarbeit/npz/eac' + str(1)
print(path)
x, y = LoadData(pathall, groupname0=0, groupname1=0, groupname2=1, groupname3=2, groupname4=3, groupname5=4).load_section_data()
np.savez_compressed('C:/Users/Anna/Desktop/Masterarbeit/npz/eac', x=x, y=y)

loaded = np.load('C:/Users/Anna/Desktop/Masterarbeit/npz/eac.npz')
print(np.array_equal(x, loaded['x']))
print(np.array_equal(y, loaded['y']))

# Todo: Hier ändern!!
#EAC.to_pickle('C:/Users/Anna/Desktop/Masterarbeit/data/a.pkl')# where to save it, usually as a .pkl

print("ready")


# EAC = pd.read_pickle('C:/Users/Anna/Desktop/Masterarbeit/pkl/a.pkl')
# patientnumber = 94
# count0, count1, count2, count3, count4, count5 = 0,0,0,0,0,0
# testname= 'MLP_Reflectance_Spatial_all_classes_630nm_Quotient_530nm_EACStromaOneClass'
# # mean_std_plot(EAC).plot_graph(name='Aufnahmeort',number=[count0,count1,count2,count3],type='Absorbance')
#
# print('test size of 10 %:' + str(patientnumber * 0.1))
# print('##############################################################################################################')
# #Testdaten = 10% der Daten
# testsize = round(patientnumber * 0.1)
# print('Patient Number: ' + str(patientnumber))
# #4x Patientennummerarray als Nullen
# # acc1, acc2, acc3, acc4 = np.zeros(patientnumber), np.zeros(patientnumber), np.zeros(patientnumber), np.zeros(
# #     patientnumber)
# # spec2, f12, f11, spec1, sens1, mcc2, mcc1, sens2, spec4, f14, f13, spec3, sens3, mcc4, mcc3, sens4 = np.zeros(
# #     patientnumber), np.zeros(patientnumber), np.zeros(patientnumber), np.zeros(patientnumber), np.zeros(
# #     patientnumber), np.zeros(patientnumber), np.zeros(patientnumber), np.zeros(patientnumber), np.zeros(
# #     patientnumber), np.zeros(patientnumber), np.zeros(patientnumber), np.zeros(patientnumber), np.zeros(
# #     patientnumber), np.zeros(patientnumber), np.zeros(patientnumber), np.zeros(patientnumber)
# #acc1, sens1, spec1, mcc1, f11 = ['Accuracy'], ['Sens'], ['Spec'], ['MCC'], ['F1']
#
# ########################################
#  # Training
# # ########################################
# #
# i = 40
# patients = np.array(EAC['patients'])
# patients = np.int_(patients)
# datastest = EAC.values[patients >= i, 3:]
# labelstest = EAC.values[patients >= i, 0]
# Patienttestpath = EAC.values[patients >= i, 2]
# #
# EAC_test = pd.DataFrame(datastest)
# EAC_test.insert(loc=0, column='label', value=labelstest)
#
# EAC_test.columns = ['label', '500nm', '505nm', '510nm', '515nm', '520nm',
#                     '525nm', '530nm', '535nm', '540nm',
#                     '545nm', '550nm', '555nm', '560nm', '565nm', '570nm', '575nm',
#                     '580nm', '585nm', '590nm',
#                     '595nm',
#                     '600nm', '605nm', '610nm', '615nm', '620nm', '625nm', '630nm',
#                     '635nm', '640nm', '645nm',
#                     '650nm', '655nm', '660nm', '665nm', '670nm', '675nm', '680nm',
#                     '685nm', '690nm', '695nm',
#                     '700nm', '705nm', '710nm', '715nm', '720nm', '725nm', '730nm',
#                     '735nm', '740nm', '745nm',
#                     '750nm', '755nm', '760nm', '765nm', '770nm', '775nm', '780nm',
#                     '785nm', '790nm', '795nm',
#                     '800nm', 'c1', 'c2', 'c3', 'c4']
# print('Patienttestset')
# labels = np.array(EAC_test['label'])
# labels = np.int_(labels)
# temp_features_1 = EAC_test.values[labels == 1, 1:]
# temp_features_2 = EAC_test.values[labels == 2, 1:]
# temp_features_3 = EAC_test.values[labels == 3, 1:]
# temp_features_4 = EAC_test.values[labels == 4, 1:]
# #
#
# Validation_x_train, Validation_y_train, Validation_x_test, Validation_y_test = train_test(
#      EAC_test).div_set_balanced_3classes()
#
# acc1, f11, spec1, mcc1, sens1 = blind_test_algorithm(EAC_test, EAC_test, Validation_x_train,
#                                                             Validation_y_train,
#                                                             Validation_x_test,
#                                                             Validation_y_test, pathall, i,
#                                                             classifier='Training').RandomFor()
#
# print('Fertig')
#Processindex='UnderBalanced MLP'

# #################################################
# End
##################################################

# for i in range(2, 80):
# #for i in range(79, 80):
#     # SG_data = savgol_filter(data.values[:, 1:], 9, 2, mode='nearest', axis=1)
#     # SG_data1 = pd.DataFrame(SG_data)
#     print(
#         '##############################################################################################################')
#     print('Patientnumber:  ' + str(i))
#     patients = np.array(EAC['patients'])
#     patients = np.int_(patients)
#     datastest = EAC.values[patients == i, 3:]
#
#     labelstest = EAC.values[patients == i, 0]
#
#     Patienttestpath = EAC.values[patients == i, 2]
#     print("Labelstest " + labelstest)
#     if Patienttestpath.size != 0:
#         print(Patienttestpath[0])
#         patienttest = [i]
#
#         print('Patienttest: ')
#         print(*patienttest)
#         EAC_test = pd.DataFrame(datastest)
#         EAC_test.insert(loc=0, column='label', value=labelstest)
#
#         EAC_test.columns = ['label', '500nm', '505nm', '510nm', '515nm', '520nm',
#                             '525nm', '530nm', '535nm', '540nm',
#                             '545nm', '550nm', '555nm', '560nm', '565nm', '570nm', '575nm',
#                             '580nm', '585nm', '590nm',
#                             '595nm',
#                             '600nm', '605nm', '610nm', '615nm', '620nm', '625nm', '630nm',
#                             '635nm', '640nm', '645nm',
#                             '650nm', '655nm', '660nm', '665nm', '670nm', '675nm', '680nm',
#                             '685nm', '690nm', '695nm',
#                             '700nm', '705nm', '710nm', '715nm', '720nm', '725nm', '730nm',
#                             '735nm', '740nm', '745nm',
#                             '750nm', '755nm', '760nm', '765nm', '770nm', '775nm', '780nm',
#                             '785nm', '790nm', '795nm',
#                             '800nm', 'c1', 'c2', 'c3', 'c4']
#     print('Patienttestset')
#
#     labels = np.array(EAC_test['label'])
#     labels = np.int_(labels)
#     # temp_features_1 = EAC_test.values[labels == 0, 1:]
#     temp_features_1 = EAC_test.values[labels == 1, 1:]
#     temp_features_2 = EAC_test.values[labels == 2, 1:]
#     temp_features_3 = EAC_test.values[labels == 3, 1:]
#     temp_features_4 = EAC_test.values[labels == 4, 1:]
#     # temp_features_6 = EAC_test.values[labels == 5, 1:]
#
#     # print("Dysplasie - ", temp_features_1.shape[0])
#     # print("Metaplasie - ", temp_features_2.shape[0])
#     print("EAC - ", temp_features_1.shape[0])
#     print("Plattenepithel - ", temp_features_2.shape[0])
#     print("Stroma - ", temp_features_3.shape[0])
#     print("Blank - ", temp_features_4.shape[0])
#     sum_all_test = temp_features_2.shape[0] + temp_features_1.shape[0] + temp_features_3.shape[0] + \
#                    temp_features_4.shape[0]
#
#     if sum_all_test != 0 and i !=78:
#         datastrain = EAC.values[patients != i, 3:]
#         labelstrain = EAC.values[patients != i, 0]
#
#         EAC_train = pd.DataFrame(datastrain)
#         EAC_train.insert(loc=0, column='label', value=labelstrain)
#
#         EAC_train.columns = ['label', '500nm', '505nm', '510nm', '515nm', '520nm',
#                              '525nm', '530nm', '535nm', '540nm',
#                              '545nm', '550nm', '555nm', '560nm', '565nm', '570nm', '575nm',
#                              '580nm', '585nm', '590nm',
#                              '595nm',
#                              '600nm', '605nm', '610nm', '615nm', '620nm', '625nm', '630nm',
#                              '635nm', '640nm', '645nm',
#                              '650nm', '655nm', '660nm', '665nm', '670nm', '675nm', '680nm',
#                              '685nm', '690nm', '695nm',
#                              '700nm', '705nm', '710nm', '715nm', '720nm', '725nm', '730nm',
#                              '735nm', '740nm', '745nm',
#                              '750nm', '755nm', '760nm', '765nm', '770nm', '775nm', '780nm',
#                              '785nm', '790nm', '795nm',
#                              '800nm', 'c1', 'c2', 'c3', 'c4']
#
#         ####################################################################
#         print('Patienttrainset')
#         labels = np.array(EAC_train['label'])
#         labels = np.int_(labels)
#         # temp_features_1 = EAC_train.values[labels == 0, 1:]
#         temp_features_2 = EAC_train.values[labels == 1, 1:]
#         temp_features_3 = EAC_train.values[labels == 2, 1:]
#         temp_features_4 = EAC_train.values[labels == 3, 1:]
#         temp_features_5 = EAC_train.values[labels == 4, 1:]
#         # temp_features_6 = EAC_train.values[labels == 5, 1:]
#
#         # print("Dysplasie - ", temp_features_1.shape[0])
#         # print("Metaplasie - ", temp_features_2.shape[0])
#         print("EAC - ", temp_features_2.shape[0])
#         print("Plattenepithel - ", temp_features_3.shape[0])
#         print("Stroma - ", temp_features_4.shape[0])
#         print("Blank - ", temp_features_5.shape[0])
#         Validation_x_train, Validation_y_train, Validation_x_test, Validation_y_test = train_test(
#             EAC_train).div_set_balanced_real3classes()
#
#         acc, f1, spec, mcc, sens = blind_test_algorithm(EAC_train, EAC_test, Validation_x_train, Validation_y_train,
#                                                         Validation_x_test,
#                                                         Validation_y_test, Patienttestpath[0], i,
#                                                         classifier=testname).neural_network()
#         f11.append((f1))
#         # f12.append((f12))
#         sens1.append((sens))
#         spec1.append((spec))
#         mcc1.append((mcc))
#         acc1.append((acc))

# print('###############################################################################################')
# print('Average Performance of each Classifier')
# print('MLP')
# print(testname)
#
# print('Accuracy')
# print(acc1)
# print('SENS')
# print(sens1)
# print('Spec')
# print(spec1)
# print('MCC')
# print(mcc1)
# print('F1')
# print(f11)
#
# np.savetxt('ACC_LR.csv', acc1, delimiter=',')
# np.savetxt('F1_LR.csv', f11, delimiter=',')
# np.savetxt('MCC_LR.csv', mcc1, delimiter=',')
# np.savetxt('SPEC_LR.csv', spec1, delimiter=',')
# np.savetxt('SENS_LR.csv', sens1, delimiter=',')
#
# np.savetxt('ACC_RF.csv', acc3, delimiter=',')
# np.savetxt('F1_RF.csv', f13, delimiter=',')
# np.savetxt('MCC_RF.csv', mcc3, delimiter=',')
# np.savetxt('SPEC_RF.csv', spec3, delimiter=',')
# np.savetxt('SENS_RF.csv', sens3, delimiter=',')
#
# np.savetxt('ACC_SVM.csv', acc2, delimiter=',')
# np.savetxt('F1_SVM.csv', f12, delimiter=',')
# np.savetxt('MCC_SVM.csv', mcc2, delimiter=',')
# np.savetxt('SPEC_SVM.csv', spec2, delimiter=',')
# np.savetxt('SENS_SVM.csv', sens2, delimiter=',')
#
# np.savetxt('ACC_KNN.csv', acc4, delimiter=',')
# np.savetxt('F1_KNN.csv', f14, delimiter=',')
# np.savetxt('MCC_KNN.csv', mcc4, delimiter=',')
# np.savetxt('SPEC_KNN.csv', spec4, delimiter=',')
# np.savetxt('SENS_KNN.csv', sens4, delimiter=',')
#
# print(acc1)
# print('f1')
# print(f11)
# print('rec')
# print(sens1)
# print('spec')
# print(spec1)
# print('mcc')
# print(mcc1)
#
# print('Average Performance of each Classifier')
# print('RF')
# print('acc: ')
# print(acc3)
# print('f1')
# print(f13)
# print('rec')
# print(sens3)
# print('spec')
# print(spec3)
# print('mcc')
# print(mcc3)