import tensorflow
import Learning.customizedPooling as cp
import numpy as np
import matplotlib.pyplot as plt


def loadModel_custPooling(number):
    modelPath = "C:/Users/Anna/Desktop/Masterarbeit/SlurmLog/balanced/hparam/20201128-235319run-" + str(number) + ".h5"
    model = tensorflow.keras.models.load_model(modelPath, custom_objects={'CustomizedPooling': cp.CustomizedPooling})
    return model


def loadModel_3D(number):
    modelPath = "C:/Users/Anna/Desktop/Masterarbeit/SlurmLog/balanced/hparam3D/20201201-054803run-" + str(
        number) + ".h5"
    model = tensorflow.keras.models.load_model(modelPath)
    return model


model = loadModel_custPooling(36)
weights_start, biases = model.layers[2].get_weights()
# weights = np.where(weights<0, weights*(-1), weights)
for layer_size in range(1, 14):
    print(layer_size)
    weights = weights_start[:, :, :, ((layer_size - 1) * 10):(layer_size * 10)]
    weights_sum = np.sum(weights, axis=(0, 1))
    weights = np.mean(weights, axis=3)

    plt.style.use('seaborn-darkgrid')
    # create a color palette
    palette = plt.get_cmap('Set1')

    # plt.plot(np.mean(weights_sum, axis=1))

    # multiple line plot
    num = 0
    for i in range(3):
        for j in range(3):
            num += 1
            plt.subplot(3, 3, num)
            for a in range(3):
                for b in range(3):
                    plt.plot(weights[a, b, :], marker='', color='grey', linewidth=0.6, alpha=0.3)
            plt.plot(weights[i, j, :], marker='', color=palette(num),
                     linewidth=1, alpha=0.9, label="(" + str(i) + "," + str(j) + ")")
            # Same limits for everybody!
            plt.xlim(0, 43)
            plt.ylim(-0.4, 0.2)

            # Not ticks everywhere
            if num in range(7):
                plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            if num not in [1, 4, 7]:
                plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

            # Add title
            plt.title("(" + str(i) + "," + str(j) + ")", loc='left', fontsize=12, fontweight=0, color=palette(num))
    # plt.legend(loc=2, ncol=2)
    # plt.ylabel('weights')
    # plt.text(0.5, 0.02, 'spectra', ha='center', va='center')
    # plt.text(0.06, 0.5, 'weights', ha='center', va='center', rotation='vertical')
    plt.show()

# versuche 9 graphen (1 pro pixel)
# versuche herrauszufinden wie wichtig unterschiedliche layer sind
