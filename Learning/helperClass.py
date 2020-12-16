import numpy as np
import pandas as pd
import os

def balanced_dataset(x, y, classes):
    smallest = 0
    for c in classes:
        i = np.size(np.where(y == c))
        if(i<smallest or smallest == 0 and not i == 0):
            smallest = i
    for c in classes:
        i = np.size(np.where(y == c))
        uarray = np.random.choice(np.arange(0, i), replace=False, size=(i-smallest))
        to_delete = np.asarray(np.where(y == c))[0, uarray]
        y = np.delete(y, to_delete)
        x = np.delete(x, to_delete, axis=0)
    return x, y


#Load data

def load1Ddata(bin):
    # pathall = ['C:/Users/Anna/Desktop/Masterarbeit/data']
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    #
    # pathall = ['C:/Users/Anna/Desktop/Masterarbeit/data']
    # print(
    #     '##############################################################################################################')
    # print('EAC- Source Offenbach')

    # Daten aus Ordner laden
    # EAC, count0, count1, count2, count3, count4, count5, patientnumber = LoadData(pathall, groupname0=0, groupname1=0,
    #                                                                               groupname2=1, groupname3=2, groupname4=3,
    #                                                                               groupname5=4).Reflectance_alg(style=None)

    # loaded = np.load('C:/Users/Anna/Desktop/Masterarbeit/npz/eac.npz')
    # print(np.array_equal(x, loaded['x']))
    # print(np.array_equal(y, loaded['y']))
    # EAC.to_pickle('C:/Users/Anna/Desktop/Masterarbeit/data/a.pkl')# where to save it, usually as a .pkl
    # LoadData(pathall, groupname0=0, groupname1=0, groupname2=1,groupname3=1, groupname4=2, groupname5=3).all_data()

    EAC = pd.read_pickle('C:/Users/Anna/Desktop/Masterarbeit/pkl/a.pkl')
    patientnumber = 94
    i = 0
    patients = np.array(EAC['patients'])
    patients = np.int_(patients)
    datastest = EAC.values[patients >= i, 3:]
    datastest = datastest[:, :61]
    labelstest = EAC.values[patients >= i, 0]

    if(bin):
        for j in range(np.size(labelstest)):
            if labelstest[j] == '3':
                labelstest[j] = 1
            else:
                labelstest[j] = 0

    datastest = np.asarray(datastest).astype(np.float32)
    labelstest = np.asarray(labelstest).astype(np.float32)
    return datastest, labelstest

def everaged1D():
    x, y = load3Ddata()
    return np.mean(x, axis=(1, 2)), y

def load3Ddata():
    npzpath = 'C:/Users/Anna/Desktop/Masterarbeit/npz'
    #npzpath = '/home/sc.uni-leipzig.de/ay312doty/npz'
    file_list = os.listdir(npzpath)
    loaded = np.load(npzpath + '/eac1.npz')
    x = loaded['x']
    y = np.array(loaded['y'][:, 0]).astype(np.integer) - 1

    for file in file_list:
        if not file.endswith('eac1.npz'):
            name = npzpath + '/' + file
            x_ = np.load(name)['x']
            y_ = np.array(np.load(name)['y'][:, 0]).astype(np.integer) - 1
            x = np.append(x, x_, axis=0)
            y = np.append(y, y_, axis=0)
    return x, y