import numpy as np
from hypercube_data import Cube_Read
import os
import tensorflow
import Learning.customizedPooling as cp
import matplotlib.pyplot as plt
import math
import multiprocessing as mp
import timeit

#Tensorflow patches
#oder numpy to patches ! Überlagern

path = "C:/Users/Anna/Desktop/Masterarbeit/data/2018_07_27_14_23_29/"
file = "2018_07_27_14_23_29_SpecCube.dat"
with open(os.path.join(path, file), newline='') as fileDatx:
    spectrum_data, pixel = Cube_Read(fileDatx, wavearea=100,
                                     Firstnm=0,
                                     Lastnm=100).cube_matrix()
model_blank = tensorflow.keras.models.load_model('C:/Users/Anna/Desktop/Masterarbeit/logs/blank_class.h5')
modelPath = "C:/Users/Anna/Desktop/Masterarbeit/SlurmLog/hparam/20201108-094422run-" + str(22) + ".h5"
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

#Todo: schleifen vermeiden
def calcLines(raum):
    print("starte" + str(raum))
    pixelSections = np.zeros((0, 3, 3, 61))
    #evaluate by mlp
    predictions = np.zeros((0, 61))
    for x in range(int(raum[0]), int(raum[1] + 1)):
        x = x * 3
        for y in range(math.floor((shape[1] - 2) / 3)):
            y = y * 3
            predictions = np.append(predictions, np.reshape(spectrum_data[x+1, y+1, 0:61], (1, 61)), axis=0)
    predictions = model_blank.predict_classes(predictions)
    predictions = np.reshape(predictions, (int(raum[1] - raum[0] + 1), math.floor((shape[1] - 2) / 3)))
    for x in range(int(raum[0]), int(raum[1] + 1)):
        x = x * 3
        for y in range(math.floor((shape[1] - 2) / 3)):
            y = y * 3
            if predictions[int((x/3)-(raum[0] + 1))][int(y/3)-1] == 0: #wenn kein blank
                pixel = spectrum_data[x:(x + 3), y:(y + 3), 0:61]
                try:
                    pixelSections = np.append(pixelSections, pixel.reshape(1, 3, 3, 61), axis=0)
                except:
                    print(np.shape(pixel))
    #for x und y --> einsortieren
    predict = model.predict_classes(pixelSections)
    i = 0
    for x in range(int(raum[0]), int(raum[1] + 1)):
        x = x * 3
        for y in range(math.floor((shape[1] - 2) / 3)):
            y = y * 3
            if predictions[int((x/3)-(raum[0] + 1))][int(y/3)-1] == 0: #wenn kein blank
                predictions[int((x/3)-(raum[0] + 1))][int(y/3)-1] = predict[i]
                i += 1

    #predictions = np.reshape(predictions, (int(raum[1] - raum[0] + 1),
    #                                       math.floor((shape[1] - 2) / 3)))
    #predictions = predictions # / 3
    return raum[0], raum[1], predictions

if __name__ == '__main__':
    calcLines([182, 211])
    print("main")
    start = timeit.default_timer()
    pool = mp.Pool(cpus)
    results = [pool.map(calcLines, [row for row in ordner_list])]
    print("ready")
    for re in results:
        for r in re:
            for x in range(int(r[0]), int(r[1]+1)):
                prediction = r[2][int(x-r[0])]
                new_image[x] = prediction
    pool.close()
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    plt.imshow(new_image)
    plt.show()
