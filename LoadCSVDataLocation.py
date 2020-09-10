import shutil

import numpy as np
import csv
import os
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.lines import Line2D
from LoadMarker import LoadMarker
from hypercube_data import Cube_Read
from sklearn.decomposition import PCA
from skimage.filters import gaussian


class LoadData:
    def __init__(self, path, groupname0, groupname1, groupname2, groupname3, groupname4, groupname5):
        self.path = path
        self.groupname0 = groupname0
        self.groupname1 = groupname1
        self.groupname2 = groupname2
        self.groupname3 = groupname3
        self.groupname4 = groupname4
        self.groupname5 = groupname5

    def all_data(self):
        patientnumber = 0
        f = 'zur Klassifizierung TNM 2014-2017_Barrett-CA, Offenbach.csv'  # 'TNM 2014-2015_Barrett-CA, Offenbach (1).csv'
        line = []
        linenew = []
        with open(f, newline='') as f:
            reader = csv.reader(f, delimiter=';')
            row1 = next(reader)
            for i, line in enumerate(reader):
                linenew.append(line)
        print(linenew[1][0])
        # auslesen der Dateien
        for path in self.path:
            import os
            file_list = os.listdir(path)
            for inpath in file_list:
                newpath = self.path[0] + '/' + inpath
                file_listnew = os.listdir(newpath)
                for file in file_listnew:
                    if file.endswith(".mk1") or file.endswith(".mk2"):
                        for line in linenew:
                            all_values = []
                            if line[6] == inpath or line[7] == inpath:
                                if line[7] == inpath:
                                    print("Patientnumber", patientnumber)
                                    print("Inpath:", inpath)
                                else:
                                    patientnumber = patientnumber + 1
                                    print("Patientnumber", patientnumber)
                                    print("Inpath:", inpath)
                                if (patientnumber != 78):
                                    with open(os.path.join(newpath, file), newline='') as fileM:
                                        for fileDat in file_listnew:
                                            if fileDat.endswith(".dat"):
                                                with open(os.path.join(newpath, fileDat), newline='') as fileDatx:
                                                    spectrum_data, pixel = Cube_Read(fileDatx, wavearea=100,
                                                                                     Firstnm=0,
                                                                                     Lastnm=100).cube_matrix()

                                                    spectrum_data = spectrum_data[:, :, 0:61]
                                                    ###########Gaussian Filter PCA ##################
                                                    i_wave_length = 65
                                                    data_2d = spectrum_data[:, :, :]
                                                    spectrum_data_1D = spectrum_data.reshape((640 * pixel, 61))
                                                    #
                                                    sigmas = [1]
                                                    n_features = 62
                                                    n_features = data_2d.shape[2]
                                                    new_data = np.zeros([640 * pixel, len(sigmas) * n_features])
                                                    #
                                                    for s_i, s in enumerate(sigmas):
                                                        for c_i in range(n_features):
                                                            new_data[..., s_i * n_features + c_i] = gaussian(
                                                                data_2d[..., c_i], sigma=s).reshape(-1)
                                                    pca = PCA(n_components=4)
                                                    pca.fit((np.float_(
                                                        new_data[:, :])))
                                                    new_data_transform = pca.transform((np.float_(
                                                        new_data[:, :])))
                                                    new_data = np.column_stack((spectrum_data_1D, new_data_transform))
                                                    new_data2D = new_data.reshape((640, pixel, 65))
                                                    spectrum_data = new_data2D[:, :, :]
                                        for x in range(np.shape(spectrum_data)[0]):
                                            for y in range(np.shape(spectrum_data)[1]):
                                                all_values.append(patientnumber)
                                                all_values.append(newpath)
                                                for iwave in range(np.shape(spectrum_data)[2]):
                                                    all_values.append(spectrum_data[x - 1, y - 1, iwave].tolist())
                                length_array = 67
                                values = np.asarray(all_values)
                                values = values.reshape((int(len(values) / length_array), length_array))

                                zeros_lines = np.where(values[:, 3] == 0)
                                sorted_values = np.delete(values, zeros_lines, 0)
                                data = sorted_values[np.argsort(sorted_values[:, 0])]
                                data_df = pd.DataFrame(data[:, :])

                                data_df.columns = ['patients', 'patientpath', '500nm', '505nm', '510nm', '515nm',
                                                       '520nm',
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
                                data_df.to_pickle('C:/Users/Anna/Desktop/Masterarbeit/pkl/patient/' + str(patientnumber) + '.pkl')
                                print(patientnumber, "loaded")

        print("dataset loaded")

    #Laed Daten mit Radius
    def ordered_data(self):
        all_values = []
        group0, group1, group2, group3, group4, group5 = 0, 0, 0, 0, 0, 0
        patientnumber = 0

        f = 'zur Klassifizierung TNM 2014-2017_Barrett-CA, Offenbach.csv'  # 'TNM 2014-2015_Barrett-CA, Offenbach (1).csv'
        line = []

        linenew = []
        with open(f, newline='') as f:
            reader = csv.reader(f, delimiter=';')
            row1 = next(reader)
            for i, line in enumerate(reader):
                linenew.append(line)
        print(linenew[1][0])
        #auslesen der Dateien
        for path in self.path:
            import os
            file_list = os.listdir(path)
            for inpath in file_list:
                newpath = self.path[0] + '/' + inpath
                file_listnew = os.listdir(newpath)
                for file in file_listnew:
                    if file.endswith(".mk1") or file.endswith(".mk2"):
                        for line in linenew:
                            if line[6] == inpath or line[7] == inpath:
                                # skipping the file with the mean values
                                if line[7] == inpath:
                                    print("Patientnumber", patientnumber)
                                    print("Inpath:", inpath)
                                else:
                                    patientnumber = patientnumber + 1
                                    print("Patientnumber", patientnumber)
                                    print("Inpath:", inpath)
                                if patientnumber != 78:
                                    # import os.path
                                    # file_path = 'C:/Users/Anna/Desktop/Masterarbeit/all' + '/' + inpath
                                    # save_path = 'C:/Users/Anna/Desktop/Masterarbeit/data' + '/' + inpath
                                    # shutil.copytree(file_path, save_path)
                                    if 'group' in locals():
                                        del group

                                    with open(os.path.join(newpath, file), newline='') as fileM:
                                        filenameMarker = fileM.name

                                        MarkerName, Leftx, Topx, Radiusx, index_Marker = LoadMarker(
                                            file_address=filenameMarker).load()
                                        # print(MarkerName)
                                        for fileDat in file_listnew:
                                            if fileDat.endswith(".dat"):
                                                with open(os.path.join(newpath, fileDat), newline='') as fileDatx:
                                                    spectrum_data, pixel = Cube_Read(fileDatx, wavearea=100,
                                                                                     Firstnm=0,
                                                                                     Lastnm=100).cube_matrix()

                                                    spectrum_data = spectrum_data[:, :, 0:61]
                                                    ###########Gaussian Filter PCA ##################
                                                    i_wave_length = 65
                                                    data_2d = spectrum_data[:, :, :]
                                                    spectrum_data_1D = spectrum_data.reshape((640 * pixel, 61))
                                                    #
                                                    sigmas = [1]
                                                    n_features = 62
                                                    n_features = data_2d.shape[2]
                                                    new_data = np.zeros([640 * pixel, len(sigmas) * n_features])
                                                    #
                                                    for s_i, s in enumerate(sigmas):
                                                        for c_i in range(n_features):
                                                            new_data[..., s_i * n_features + c_i] = gaussian(
                                                                data_2d[..., c_i], sigma=s).reshape(-1)

                                                    pca = PCA(n_components=4)
                                                    pca.fit((np.float_(
                                                        new_data[:, :])))
                                                    new_data_transform = pca.transform((np.float_(
                                                        new_data[:, :])))

                                                    new_data = np.column_stack((spectrum_data_1D, new_data_transform))

                                                    new_data2D = new_data.reshape((640, pixel, 65))

                                                    spectrum_data = new_data2D[:, :, :]

                                                    ######################################################################

                                        for n in range(len(MarkerName)):
                                            if MarkerName[n] == 'EAC':
                                                group = self.groupname2
                                                Radius = Radiusx[n]
                                                Left = Leftx[n]
                                                Top = Topx[n]

                                                x_max = Left + Radius
                                                x_min = Left - Radius
                                                y_max = Top + Radius
                                                y_min = Top - Radius

                                                x_ = np.arange(Left - Radius - 1, Left + Radius + 1, dtype=int)
                                                y_ = np.arange(Top - Radius - 1, Top + Radius + 1, dtype=int)
                                                x, y = np.where(
                                                    (x_[:, np.newaxis] - Left) ** 2 + (y_ - Top) ** 2 <= Radius ** 2)
                                                # x, y = np.where((np.hypot((x_-x0)[:,np.newaxis], y_-y0)<= radius)) # alternative implementation
                                                for x, y in zip(x_[x], y_[y]):
                                                    all_values.append(group)
                                                    all_values.append(patientnumber)
                                                    all_values.append(newpath)
                                                    for iwave in range(0, i_wave_length):
                                                        all_values.append(spectrum_data[x - 1, y - 1, iwave].tolist())

                                            if MarkerName[n] == 'Stroma':
                                                group = self.groupname3
                                                Radius = Radiusx[n]
                                                Left = Leftx[n]
                                                Top = Topx[n]

                                                x_max = Left + Radius
                                                x_min = Left - Radius
                                                y_max = Top + Radius
                                                y_min = Top - Radius

                                                x_ = np.arange(Left - Radius - 1, Left + Radius + 1, dtype=int)
                                                y_ = np.arange(Top - Radius - 1, Top + Radius + 1, dtype=int)
                                                x, y = np.where(
                                                    (x_[:, np.newaxis] - Left) ** 2 + (y_ - Top) ** 2 <= Radius ** 2)
                                                # x, y = np.where((np.hypot((x_-x0)[:,np.newaxis], y_-y0)<= radius)) # alternative implementation
                                                for x, y in zip(x_[x], y_[y]):
                                                    all_values.append(group)
                                                    all_values.append(patientnumber)
                                                    all_values.append(newpath)
                                                    for iwave in range(0, i_wave_length):
                                                        all_values.append(spectrum_data[x - 1, y - 1, iwave].tolist())

                                            if MarkerName[n] == 'Plattenepithel':
                                                group = self.groupname4
                                                Radius = Radiusx[n]
                                                Left = Leftx[n]
                                                Top = Topx[n]

                                                x_max = Left + Radius
                                                x_min = Left - Radius
                                                y_max = Top + Radius
                                                y_min = Top - Radius

                                                x_ = np.arange(Left - Radius - 1, Left + Radius + 1, dtype=int)
                                                y_ = np.arange(Top - Radius - 1, Top + Radius + 1, dtype=int)
                                                x, y = np.where(
                                                    (x_[:, np.newaxis] - Left) ** 2 + (y_ - Top) ** 2 <= Radius ** 2)
                                                # x, y = np.where((np.hypot((x_-x0)[:,np.newaxis], y_-y0)<= radius)) # alternative implementation
                                                print("Plattenepithel:  ", patientnumber)
                                                for x, y in zip(x_[x], y_[y]):
                                                    all_values.append(group)
                                                    all_values.append(patientnumber)

                                                    all_values.append(newpath)
                                                    for iwave in range(0, i_wave_length):
                                                        all_values.append(spectrum_data[x - 1, y - 1, iwave].tolist())

                                        if MarkerName[n] == 'Blank' or MarkerName[n] == 'blank':
                                            group = self.groupname5
                                            Radius = Radiusx[n]
                                            Left = Leftx[n]
                                            Top = Topx[n]

                                            x_max = Left + Radius
                                            x_min = Left - Radius
                                            y_max = Top + Radius
                                            y_min = Top - Radius

                                            if y_max > 480:
                                                Radius = 480 - Top
                                            if x_max > 640:
                                                Radius = 640 - Left

                                            x_ = np.arange(Left - Radius - 1, Left + Radius + 1, dtype=int)
                                            y_ = np.arange(Top - Radius - 1, Top + Radius + 1, dtype=int)
                                            x, y = np.where(
                                                (x_[:, np.newaxis] - Left) ** 2 + (y_ - Top) ** 2 <= Radius ** 2)
                                            # x, y = np.where((np.hypot((x_-x0)[:,np.newaxis], y_-y0)<= radius)) # alternative implementation

                                            for x, y in zip(x_[x], y_[y]):

                                                all_values.append(group)
                                                all_values.append(patientnumber)
                                                all_values.append(newpath)
                                                for iwave in range(0, i_wave_length):
                                                    all_values.append(spectrum_data[x - 1, y - 1, iwave].tolist())

                                # if line[7] == inpath:
                                #    patientnumber=patientnumber+1

        length_array = 68

        values = np.asarray(all_values)
        values = values.reshape((int(len(values) / length_array), length_array))

        # remove any rows with all zeros
        zeros_lines = np.where(values[:, 3] == 0)
        sorted_values = np.delete(values, zeros_lines, 0)

        # sort the values for easier manipulation
        data = sorted_values[np.argsort(sorted_values[:, 0])]
        # save the result in panda dataframe for later classification
        data_df = pd.DataFrame(data[:, :])
        # tissue_types = {0: "Gauze", 1: "Instrument", 2: "Skin", 3: "Thyroid", 4: "Parathyroid", 5: "Muscle"}
        # data_df.columns=['label', 'patients', 'patientpath','500nm', '505nm' ]

        data_df.columns = ['label', 'patients', 'patientpath', '500nm', '505nm', '510nm', '515nm', '520nm',
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

        # data_df['label'] = [tissue_types[x] for x in data_df['label']]
        # to save the data as csv file
        #data_df.to_csv('data_EAC_platte_stroma_blank.csv', index=False)
        # self.plot_spectra(data_df, title = 'Original data')
        # data_df = data_df.astype({"label": np.int_, "patients": np.int_})
        print("dataset loaded")
        labels = np.array(data_df['label'])
        labels = np.int_(labels)
        # temp_features_1 = data_df.values[labels == 0, 4:]
        temp_features_2 = data_df.values[labels == 1, 4:]
        temp_features_3 = data_df.values[labels == 2, 4:]
        temp_features_4 = data_df.values[labels == 3, 4:]
        temp_features_5 = data_df.values[labels == 4, 4:]
        # temp_features_6 = data_df.values[labels == 5, 4:]

        # print("Dysplasie - ", temp_features_1.shape[0])
        # print("Metaplasie - ", temp_features_1.shape[0])
        print("EAC - ", temp_features_2.shape[0])
        print("Plattenepithel - ", temp_features_3.shape[0])
        print("Stroma - ", temp_features_4.shape[0])
        print("Blank - ", temp_features_5.shape[0])

        return data_df, group0, group1, group2, group3, group4, group5, patientnumber

    #nicht genutzt
    def plot_spectra(self, data, title):
        x_axis = np.arange(500, 1000, 5)
        labels = np.array(data['label'])
        temp_features_1 = data.transpose().values[1:, labels == 1]
        temp_features_2 = data.transpose().values[1:, labels == 2]

        plt.figure()
        # plt.plot(x_axis, temp_features_0[:,:10], color = 'C4')
        plt.plot(x_axis, temp_features_1[:, :10], color='C3')
        plt.plot(x_axis, temp_features_2[:, :10], color='C5')
        plt.title(title)
        custom_lines = [  # Line2D([0], [0], color='C4', lw=4),
            Line2D([0], [0], color='C3', lw=4),
            Line2D([0], [0], color='C5', lw=4), ]
        plt.legend(custom_lines, ['healthy', 'carcinom'])
        plt.xlabel('wavelenght, nm')
        plt.ylabel('reflectance')
        plt.show()

        # smoothings/filtering of data

    #nicht genutzt
    def Savitzky_Golay_smoothing(self):
        data, group0, group1, group2, group3, group4, group5, patientnumber = self.ordered_data()
        # to test which window_length is best --> size of 9 looks good enough
        '''
        for i in range(3,11,2):
            SG_data = savgol_filter(data.values[:,1:], i, 2, mode='nearest', axis=1)
            SG_data1 = pd.DataFrame(SG_data)
            labels = data.values[:,0]
            SG_data1.insert(loc=0, column = 'label', value = labels)
            self.plot_spectra(SG_data1, title = 'Savitzy-Golay smoothing', legend = '%i'%i)
        '''
        dataSG = data.values[:, 3:]
        dataSG = np.float_(dataSG)
        SG_data = savgol_filter(dataSG, 9, 2, mode='nearest', axis=1)
        SG_data1 = pd.DataFrame(SG_data)
        labels = data.values[:, 0]
        labels = np.int_(labels)
        patients = data.values[:, 1]
        patients = np.int_(patients)
        patientPath = data.values[:, 2]
        SG_data1.insert(loc=0, column='label', value=labels)
        SG_data1.insert(loc=1, column='patients', value=patients)
        SG_data1.insert(loc=2, column='patientpath', value=patientPath)
        # self.plot_spectra(SG_data1, title = 'Savitzy-Golay smoothing')
        # tissue_types = {0: "Gauze", 1: "Instrument", 2: "Skin", 3: "Thyroid", 4: "Parathyroid", 5: "Muscle"}
        # SG_data1['label'] = [tissue_types[x] for x in SG_data1['label']]
        SG_data1.columns = ['label', 'patients', '500nm', '505nm', '510nm', '515nm', '520nm', '525nm', '530nm', '535nm',
                            '540nm',
                            '545nm', '550nm', '555nm', '560nm', '565nm', '570nm', '575nm', '580nm', '585nm', '590nm',
                            '595nm',
                            '600nm', '605nm', '610nm', '615nm', '620nm', '625nm', '630nm', '635nm', '640nm', '645nm',
                            '650nm', '655nm', '660nm', '665nm', '670nm', '675nm', '680nm', '685nm', '690nm', '695nm',
                            '700nm', '705nm', '710nm', '715nm', '720nm', '725nm', '730nm', '735nm', '740nm', '745nm',
                            '750nm', '755nm', '760nm', '765nm', '770nm', '775nm', '780nm', '785nm', '790nm', '795nm',
                            '800nm']  # , '805nm', '810nm', '815nm', '820nm', '825nm', '830nm', '835nm', '840nm', '845nm',
        # '850nm', '855nm', '860nm', '865nm', '870nm', '875nm', '880nm', '885nm', '890nm', '895nm',
        # '900nm', '905nm', '910nm', '915nm', '920nm', '925nm', '930nm', '935nm', '940nm', '945nm',
        # '950nm', '955nm', '960nm', '965nm', '970nm', '975nm', '980nm', '985nm', '990nm', '995nm', ]

        ### gaussian pca ###

        SG_data1.columns = ['label', 'patients', 'patientpath', '500nm', '505nm', '510nm', '515nm', '520nm',
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

        return SG_data1, group0, group1, group2, group3, group4, group5, patientnumber

        # standardarisation of data

    #nicht genutzt
    def SNV_alg(self, style=None, numbers=None):
        # my implementation of SNV, limited to all features
        print('SNV Standardsclaer transponse')
        if style == 'Savitzky_Golay_smoothing':
            data, group0, group1, group2, group3, group4, group5, patientnumber = self.Savitzky_Golay_smoothing()
        else:
            data, group0, group1, group2, group3, group4, group5, patientnumber = self.ordered_data()

        # data = self.ordered_data()
        labels = data.values[:, 0]
        labels = np.int_(labels)
        patients = data.values[:, 1]
        patients = np.int_(patients)
        pathName = data.values[:, 2]
        data_values = data.values[:, 3:]
        data_values = np.float_(data_values)
        mean = np.ravel(data_values.mean(axis=1))
        print(mean)
        # mean = np.ravel(data_values.mean(axis=1))
        # std = np.ravel(data_values.std(axis=1))
        # SNV_column = np.zeros((data.shape[0],100))
        # for n in range(1,100):
        #    for i in range(data.shape[0]):
        #        SNV_column[i,0] = data.iat[i,0]
        #        SNV_column[i,n] = (data.iat[i,n]-mean[i])/std[i]
        # SNV_data = pd.DataFrame(SNV_column)
        ############################################
        from sklearn.preprocessing import StandardScaler
        data = np.transpose(data_values)
        scaler = StandardScaler()
        scaler.fit(data)
        SNV_Scale = scaler.transform(data)
        SNV_Scale = np.transpose(SNV_Scale)
        np.nan_to_num(SNV_Scale, nan=0, posinf=0, neginf=0)
        ####################################################
        SNV_data = pd.DataFrame(SNV_Scale)

        # tissue_types = {0: "Gauze", 1: "Instrument", 2: "Skin", 3: "Thyroid", 4: "Parathyroid", 5: "Muscle"}
        ### gaussian pca #####
        SNV_data.columns = ['500nm', '505nm', '510nm', '515nm', '520nm',
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
        SNV_data.insert(loc=0, column='label', value=labels)
        SNV_data.insert(loc=1, column='patients', value=patients)
        SNV_data.insert(loc=2, column='patientpath', value=pathName)
        patients = np.array(SNV_data['patients'])
        # for i in range(1, 29):
        #    value = SNV_data.values[patients == i, 2:]
        #    value = -1 * (np.log10(abs(value)))
        #    np.nan_to_num(value, nan=0, posinf=0, neginf=0)
        #    value[value > 255] = 5
        #    SNV_data.values[patients == i, 2:] = value

        if numbers == 'All':
            labels = np.array(SNV_data['label'])
            temp_features_1 = SNV_data.values[labels == 1, 1:]
            temp_features_2 = SNV_data.values[labels == 2, 1:]

            print("cancer - ", temp_features_1.shape[0])
            print("healthy          - ", temp_features_2.shape[0])

        return SNV_data, group0, group1, group2, group3, group4, group5, patientnumber


    def Reflectance_alg(self, style=None):
        print('Reflectance')
        # using the preprocessing option from scikit learn along the samples and not the features (axis-1)
        if style == 'Savitzky_Golay_smoothing':
            data, group0, group1, group2, group3, group4, group5, patientnumber = self.Savitzky_Golay_smoothing()
        else:
            data, group0, group1, group2, group3, group4, group5, patientnumber = self.ordered_data()
        #SNV_data1 = data

        return data, group0, group1, group2, group3, group4, group5, patientnumber

    #nicht genutzt
    def SNV_second_try(self, style=None):
        # using the preprocessing option from scikit learn along the samples and not the features (axis-1)
        if style == 'Savitzky_Golay_smoothing':
            data, group0, group1, group2, group3, group4, group5, patientnumber = self.Savitzky_Golay_smoothing()
        else:
            data, group0, group1, group2, group3, group4, group5, patientnumber = self.ordered_data()
        SNV_draft = data.values[:, 1:]
        labels = data.values[:, 0]
        SNV_data = preprocessing.scale(SNV_draft, axis=1)
        SNV_data1 = pd.DataFrame(SNV_data)
        SNV_data1.columns = ['500nm', '505nm', '510nm', '515nm', '520nm', '525nm', '530nm', '535nm', '540nm', '545nm',
                             '550nm', '555nm', '560nm', '565nm', '570nm', '575nm', '580nm', '585nm', '590nm', '595nm',
                             '600nm', '605nm', '610nm', '615nm', '620nm', '625nm', '630nm', '635nm', '640nm', '645nm',
                             '650nm', '655nm', '660nm', '665nm', '670nm', '675nm', '680nm', '685nm', '690nm', '695nm',
                             '700nm', '705nm', '710nm', '715nm', '720nm', '725nm', '730nm', '735nm', '740nm', '745nm',
                             '750nm', '755nm', '760nm', '765nm', '770nm', '775nm', '780nm', '785nm', '790nm', '795nm',
                             '800nm']  # , '805nm', '810nm', '815nm', '820nm', '825nm', '830nm', '835nm', '840nm', '845nm',
        # '850nm', '855nm', '860nm', '865nm', '870nm', '875nm', '880nm', '885nm', '890nm', '895nm',
        # '900nm', '905nm', '910nm', '915nm', '920nm', '925nm', '930nm', '935nm', '940nm', '945nm',
        # '950nm', '955nm', '960nm', '965nm', '970nm', '975nm', '980nm', '985nm', '990nm', '995nm', ]
        SNV_data1.insert(loc=0, column='label', value=labels)
        return SNV_data1, group0, group1, group2, group3, group4, group5, patientnumber

    #nicht genutzt
    def Absorbance_alg(self, style=None):
        print('Absorbance')
        # using the preprocessing option from scikit learn along the samples and not the features (axis-1)
        if style == 'Savitzky_Golay_smoothing':
            data, group0, group1, group2, group3, group4, group5, patientnumber = self.Savitzky_Golay_smoothing()
        else:
            data, group0, group1, group2, group3, group4, group5, patientnumber = self.ordered_data()
        SNV_draft = data.values[:, 1:]
        labels = data.values[:, 0]
        SNV_data = -1 * (np.log10(abs(SNV_draft)))
        SNV_data1 = pd.DataFrame(SNV_data)
        SNV_data1.columns = ['500nm', '505nm', '510nm', '515nm', '520nm', '525nm', '530nm', '535nm', '540nm', '545nm',
                             '550nm', '555nm', '560nm', '565nm', '570nm', '575nm', '580nm', '585nm', '590nm', '595nm',
                             '600nm', '605nm', '610nm', '615nm', '620nm', '625nm', '630nm', '635nm', '640nm', '645nm',
                             '650nm', '655nm', '660nm', '665nm', '670nm', '675nm', '680nm', '685nm', '690nm', '695nm',
                             '700nm', '705nm', '710nm', '715nm', '720nm', '725nm', '730nm', '735nm', '740nm', '745nm',
                             '750nm', '755nm', '760nm', '765nm', '770nm', '775nm', '780nm', '785nm', '790nm', '795nm',
                             '800nm']  # , '805nm', '810nm', '815nm', '820nm', '825nm', '830nm', '835nm', '840nm', '845nm',
        # '850nm', '855nm', '860nm', '865nm', '870nm', '875nm', '880nm', '885nm', '890nm', '895nm',
        # '900nm', '905nm', '910nm', '915nm', '920nm', '925nm', '930nm', '935nm', '940nm', '945nm',
        # '950nm', '955nm', '960nm', '965nm', '970nm', '975nm', '980nm', '985nm', '990nm', '995nm', ]
        SNV_data1.insert(loc=0, column='label', value=labels)
        return SNV_data1, group0, group1, group2, group3, group4, group5, patientnumber

    #nicht genutzt
    def AbsorbanceSNV_alg(self, style=None):
        print('SNV Absorbance')
        # using the preprocessing option from scikit learn along the samples and not the features (axis-1)
        if style == 'Savitzky_Golay_smoothing':
            data, group0, group1, group2, group3, group4, group5, patientnumber = self.Savitzky_Golay_smoothing()
        else:
            data, group0, group1, group2, group3, group4, group5, patientnumber = self.ordered_data()
        SNV_draft = data.values[:, 1:]
        labels = data.values[:, 0]
        SNV_data = -1 * (np.log1p(SNV_draft))
        from sklearn.preprocessing import StandardScaler
        data = np.transpose(SNV_data)
        scaler = StandardScaler()
        scaler.fit(data)
        SNV_Scale = scaler.transform(data)
        SNV_Scale = np.transpose(SNV_Scale)

        SNV_data1 = pd.DataFrame(SNV_Scale)
        SNV_data1.columns = ['500nm', '505nm', '510nm', '515nm', '520nm', '525nm', '530nm', '535nm', '540nm', '545nm',
                             '550nm', '555nm', '560nm', '565nm', '570nm', '575nm', '580nm', '585nm', '590nm', '595nm',
                             '600nm', '605nm', '610nm', '615nm', '620nm', '625nm', '630nm', '635nm', '640nm', '645nm',
                             '650nm', '655nm', '660nm', '665nm', '670nm', '675nm', '680nm', '685nm', '690nm', '695nm',
                             '700nm', '705nm', '710nm', '715nm', '720nm', '725nm', '730nm', '735nm', '740nm', '745nm',
                             '750nm', '755nm', '760nm', '765nm', '770nm', '775nm', '780nm', '785nm', '790nm', '795nm',
                             '800nm']
        SNV_data1.insert(loc=0, column='label', value=labels)
        return SNV_data1, group0, group1, group2, group3, group4, group5, patientnumber
