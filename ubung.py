import numpy as np
import pixelSections

pathall = ['C:/Users/Anna/Desktop/Masterarbeit/data']
import multiprocessing as mp
cpus = mp.cpu_count()
print("Number of processors: ", cpus)
ordner = pixelSections.LoadData(pathall, groupname0=0, groupname1=0, groupname2=1,
                                groupname3=2, groupname4=3, groupname5=4).auslesen_offenbach()
ordner_list = np.array_split(ordner, cpus)

def parallel(ordner):
    print(str(ordner[0][0]) + " starte")
    pixelSections.LoadPatients(ordner, groupname0=0, groupname1=0, groupname2=1,
                               groupname3=2, groupname4=3, groupname5=4).start()
    return 1

if __name__ == '__main__':
    print("main")
    pool = mp.Pool(cpus)
    results = [pool.map(parallel, [row for row in ordner_list])]
    print("ready")
    pool.close()
