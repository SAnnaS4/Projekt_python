import numpy as np
from time import time

# Prepare data
import pixelSections

#np.random.RandomState(100)
#arr = np.random.randint(0, 10, size=[200000, 5])
#data = arr.tolist()

# Parallelizing using Pool.map()

def howmany_within_range_rowonly(row, minimum=4, maximum=8):
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    return count

# if __name__ == '__main__':
#     pool = mp.Pool(mp.cpu_count())
#
#     results = pool.map(howmany_within_range_rowonly, [row for row in data])
#
#     pool.close()
#
#     print(results[:10])

# Redefine, with only 1 mandatory argument.

#> [3, 1, 4, 4, 4, 2, 1, 1, 3, 3]
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
    # Step 1: Init multiprocessing.Pool()
    pool = mp.Pool(mp.cpu_count())

    # Step 2: `pool.apply` the `howmany_within_range()`
    results = [pool.map(parallel, [row for row in ordner_list])]
    print("ready")
    pool.close()
    #self.parallel(ordner)

