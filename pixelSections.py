import numpy as np
import csv
from LoadMarker import LoadMarker
from hypercube_data import Cube_Read
import os

class LoadPatients:
    all_values = []
    liste = np.zeros([0, 3, 3, 61])

    def __init__(self, ordner, groupname0, groupname1, groupname2, groupname3, groupname4, groupname5):
        self.groupname0 = groupname0
        self.groupname1 = groupname1
        self.groupname2 = groupname2
        self.groupname3 = groupname3
        self.groupname4 = groupname4
        self.groupname5 = groupname5
        self.ordner = ordner

    def start(self):
        print(str(self.ordner[0][0]) + " starte")
        for folder in self.ordner:
            self.load_section_data_patient(folder[0], folder[1], folder[2])
        ort = self.zwischenspeichern(self.ordner[0,0])
        print(ort)
        return ort

    def load_section_data_patient(self, patientnumber, newpath, file):
        file_listnew = os.listdir(newpath)
        with open(os.path.join(newpath, file), newline='') as fileM:
            filenameMarker = fileM.name

            MarkerName, Leftx, Topx, Radiusx, index_Marker = LoadMarker(
                file_address=filenameMarker).load()
            for fileDat in file_listnew:
                if fileDat.endswith(".dat"):
                    with open(os.path.join(newpath, fileDat), newline='') as fileDatx:
                        spectrum_data, pixel = Cube_Read(fileDatx, wavearea=100,
                                                         Firstnm=0,
                                                         Lastnm=100).cube_matrix()

                        spectrum_data = spectrum_data[:, :, 0:61]
                        i_wave_length = 61

            for n in range(len(MarkerName)):
                if MarkerName[n] == 'EAC':
                    group = self.groupname2
                    self.getMarkerData(group, n, Radiusx, Leftx, Topx, patientnumber,
                                       newpath, i_wave_length, spectrum_data)

                if MarkerName[n] == 'Stroma':
                    group = self.groupname3
                    self.getMarkerData(group, n, Radiusx, Leftx, Topx,
                                       patientnumber,
                                       newpath, i_wave_length, spectrum_data)

                if MarkerName[n] == 'Plattenepithel':
                    group = self.groupname4
                    self.getMarkerData(group, n, Radiusx, Leftx, Topx,
                                       patientnumber,
                                       newpath, i_wave_length, spectrum_data)

            if MarkerName[n] == 'Blank' or MarkerName[n] == 'blank':
                group = self.groupname5
                self.getMarkerData(group, n, Radiusx, Leftx, Topx,
                                   patientnumber,
                                   newpath, i_wave_length, spectrum_data)

        length_array = 3

        values = np.asarray(self.all_values)
        values = values.reshape((int(len(values) / length_array), length_array))

        print(str(patientnumber) + " Loaded")
        # [1][0] vorne für patienten
        return self.liste, values

    def zwischenspeichern(self, patientnumber):
        length_array = 3
        values = np.asarray(self.all_values)
        values = values.reshape((int(len(values) / length_array), length_array))
        path = 'C:/Users/Anna/Desktop/Masterarbeit/npz/eac' + str(patientnumber)
        np.savez_compressed(path, x=self.liste, y=values)
        self.all_values = []
        self.liste = np.zeros([0, 3, 3, 61])
        return path

    def search_in_neighborhood(self, arr, x, y, d=1):
        nachbarn = []
        start_x = x - d
        start_y = y - d
        leange = d * 2 + 1
        for i in range(leange):
            for j in range(leange):
                try:
                    nachbarn.append(arr[start_x, start_y, :])
                except:
                    nachbarn.append(np.zeros(61))
                start_x += 1
            start_x = x - d
            start_y += 1
        nachbarn = np.reshape(nachbarn, (3, 3, 61))
        return nachbarn

    def get_section(self, im, i, j, d=1):
        n = im[i - d:i + d + 1, j - d:j + d + 1, :]
        size = np.shape(n)[0] * np.shape(n)[1]
        if size < ((1 + d * 2) ** 2):
            n = self.search_in_neighborhood(im, i, j, d)
        return n

    def getMarkerData(self, group, n, Radiusx, Leftx, Topx, patientnumber, newpath, i_wave_length, spectrum_data):
        Radius = Radiusx[n]
        Left = Leftx[n]
        Top = Topx[n]

        x_ = np.arange(Left - Radius - 1, Left + Radius + 1, dtype=int)
        y_ = np.arange(Top - Radius - 1, Top + Radius + 1, dtype=int)
        # alle Pixel aus rundem Kreis
        x, y = np.where(
            (x_[:, np.newaxis] - Left) ** 2 + (y_ - Top) ** 2 <= Radius ** 2)
        for x, y in zip(x_[x], y_[y]):
            self.all_values.append(group)
            self.all_values.append(patientnumber)
            self.all_values.append(newpath)
            section3d = self.get_section(spectrum_data[:, :, :], x, y, 1)
            self.liste = np.append(self.liste, [section3d], axis=0)
        print('Marker Loaded')

class LoadData:
    group0, group1, group2, group3, group4, group5 = 0, 0, 0, 0, 0, 0

    def __init__(self, path, groupname0, groupname1, groupname2, groupname3, groupname4, groupname5):
        self.path = path
        self.groupname0 = groupname0
        self.groupname1 = groupname1
        self.groupname2 = groupname2
        self.groupname3 = groupname3
        self.groupname4 = groupname4
        self.groupname5 = groupname5

    def auslesen_offenbach(self):
        ordner = np.zeros([0, 3])
        f = 'zur Klassifizierung TNM 2014-2017_Barrett-CA, Offenbach.csv'  # 'TNM 2014-2015_Barrett-CA, Offenbach (1).csv'
        patientnumber = 0
        linenew = []
        with open(f, newline='') as f:
            reader = csv.reader(f, delimiter=';')
            row1 = next(reader)
            for i, line in enumerate(reader):
                linenew.append(line)
        print(linenew[1][0])
        # auslesen der Dateien
        for path in self.path:
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
                                   new = [patientnumber, newpath, file]
                                   ordner = np.append(ordner, [new], axis=0)
        return ordner


