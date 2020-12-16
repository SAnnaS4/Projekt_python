import numpy as np
import matplotlib.pyplot as plt
import math


def showSomeData(x_test, y_test):
    eac = x_test[np.where(y_test == 0)[0][0]]
    eac_mean = x_test[np.where(y_test == 0)].mean(axis=0)
    eac_std = x_test[np.where(y_test == 0)].std(axis=0)
    pe = x_test[np.where(y_test == 1)[0][0]]
    pe_mean = x_test[np.where(y_test == 1)].mean(axis=0)
    pe_std = x_test[np.where(y_test == 1)].std(axis=0)
    # stroma = x_test[np.where(y_test == "2")[0][0]]
    # stroma_mean = x_test[np.where(y_test == "2")].mean(axis=0)
    # stroma_std = x_test[np.where(y_test == "2")].std(axis=0)
    # blank = x_test[np.where(y_test == "3")[0][0]]
    # blank_mean = x_test[np.where(y_test == "3")].mean(axis=0)
    # blank_std = x_test[np.where(y_test == "3")].std(axis=0)

    fig, axs = plt.subplots(4, 3, figsize=(10, 10))
    axs[0, 0].plot(eac)
    axs[0, 1].plot(eac_mean)
    axs[0, 2].plot(eac_std)
    axs[1, 0].plot(pe)
    axs[1, 1].plot(pe_mean)
    axs[1, 2].plot(pe_std)
    # axs[2, 0].plot(stroma)
    # axs[2, 1].plot(stroma_mean)
    # axs[2, 2].plot(stroma_std)
    # axs[3, 0].plot(blank)
    # axs[3, 1].plot(blank_mean)
    # axs[3, 2].plot(blank_std)
    plt.show()


def analysePCA_kmeans():
    path = "C:/Users/Anna/Desktop/Masterarbeit/data/pcaValues2D.npz"
   # path = "C:/Users/Anna/Desktop/Masterarbeit/data/pcaValues.npz"
    values = np.load(path)['median_values']
    epochs = values.shape[2]
    folds = values.shape[1]
    tests = values.shape[0]
    values = values[:, :, epochs - 6:epochs - 1]
    objects = ('6', '11', '16', '21', '26', '31', '36', '41', '46', '51', '56', '61')
    max_epoch_values = np.mean(values, axis=2)
    median_values = np.median(max_epoch_values, axis=1)
    max_values = np.mean(max_epoch_values, axis=1)

    #    plt.show()

    # data to plot
    n_groups = len(objects)

    # create plot
    plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    #  opacity = 0.8

    plt.bar(index, median_values, bar_width,
            alpha=0.5,
            color='b',
            label='median')

    plt.bar(index + bar_width, max_values, bar_width,
            alpha=0.5,
            color='g',
            label='mean')
    min = np.min(max_values)
    plt.ylim([0.5, 1])
    plt.xlabel('number of Components')
    plt.ylabel('Scores')
    plt.xticks(index + bar_width, objects)
    plt.legend()

    plt.tight_layout()
    plt.show()

analysePCA_kmeans()