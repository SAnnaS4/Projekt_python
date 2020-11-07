import numpy as np
from hypercube_data import Cube_Read
import os
import tensorflow
import Learning.customizedPooling as cp
import matplotlib.pyplot as plt
import math
import multiprocessing as mp

path = "C:/Users/Anna/Desktop/Masterarbeit/data/2018_07_12_17_38_25"
file = "2018_07_12_17_38_25_SpecCube.dat"
with open(os.path.join(path, file), newline='') as fileDatx:
    spectrum_data, pixel = Cube_Read(fileDatx, wavearea=100,
                                     Firstnm=0,
                                     Lastnm=100).cube_matrix()
modelPath = "C:/Users/Anna/Desktop/Masterarbeit/SlurmLog/test/hparam/20201023-105910run-" + str(44) + ".h5"
model = tensorflow.keras.models.load_model(modelPath,
                                           custom_objects={'CustomizedPooling': cp.CustomizedPooling})
shape = np.shape(spectrum_data)
new_image = np.zeros([(math.floor((shape[0] - 2) / 3)), (math.floor((shape[1] - 2) / 3))])
size = math.floor((shape[0] - 2) / 3)
cpus = mp.cpu_count()
length = math.floor(size/cpus)
liste = np.zeros((0, 2))
pos = 0
for l in range(cpus - 1):
    liste = np.append(liste, [pos, pos + length - 1])
    pos = pos + length
liste = np.append(liste, [pos, size-1])
ordner_list = np.array_split(liste, cpus)
all_pixelSections = np.zeros((0, 3, 3, 61))

def calcLines(raum):
    print("starte" + str(raum))
    pixelSections = np.zeros((0, 3, 3, 61))
    for x in range(int(raum[0]), int(raum[1] + 1)):
        x = x * 3
        print("new Line: " + str(x))
        for y in range(math.floor((shape[1] - 2) / 3)):
            y = y * 3
            pixel = spectrum_data[(x):(x + 3), (y):(y + 3), 0:61]
            try:
                pixelSections = np.append(pixelSections, pixel.reshape(1, 3, 3, 61), axis=0)
            except:
                print(np.shape(pixel))
    predictions = model.predict_classes(pixelSections)
    predictions = np.reshape(predictions, (int(raum[1] - raum[0] + 1),
                                           math.floor((shape[1] - 2) / 3)))
    predictions = predictions / 3
    return raum[0], raum[1], predictions

if __name__ == '__main__':
    print("main")
    pool = mp.Pool(cpus)
    results = [pool.map(calcLines, [row for row in ordner_list])]
    print("ready")
    for re in results:
        for r in re:
            for x in range(int(r[0]), int(r[1]+1)):
                prediction = r[2][int(x-r[0])]
                new_image[x] = prediction
    pool.close()
    plt.imshow(new_image)
    plt.show()

def search_in_neighborhood(arr, x, y, d=1):
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

def get_section(im, i, j, d=1):
    n = im[i - d:i + d + 1, j - d:j + d + 1, :]
    size = np.shape(n)[0] * np.shape(n)[1]
    if (size) < ((1 + d * 2) ** 2):
        n = search_in_neighborhood(im, i, j, d)
    return n

def getPixelSection(x, y):
        sectionT = get_section(spectrum_data[:, :, :], x, y, 1)
        return sectionT
