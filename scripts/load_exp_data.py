
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import os

exp_name = "Multiclass_Report_Results"
folds = 5
epochs = 50
for task in ['RP']:
    for arch in ['[100,100]']:
        model_name = "MUL_" + task + "_" + arch + "_WD0_EP" + str(epochs)
        avg = np.zeros(3)

        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(12, 6))
        for i in range(1,folds+1,1):
            tr_curves = pd.read_csv(exp_name + "/OUT/" + model_name + '_' + str(i) +'.tr_curves')
            val_curves = pd.read_csv(exp_name + "/OUT/" + model_name + '_' + str(i) +'.val_curves')
            tr_losses = tr_curves[["tr_loss"]].values[:,0]
            val_losses = val_curves[["val_loss"]].values[:,0]

            tr_accuracies = tr_curves[["tr_acc"]].values[:,0]
            val_accuracies = val_curves[["val_acc"]].values[:,0]

            tr_auc = tr_curves[["tr_auc"]].values[:,0]
            val_auc = val_curves[["val_auc"]].values[:,0]
            #avg += curves.T[29].values[1:]

            axes[0, 0].plot(np.arange(len(tr_losses)), tr_losses, lw=1, linewidth=2, label="TRAIN, FOLD {}".format(i))
            axes[0, 0].plot(np.arange(len(val_losses)), val_losses, lw=1, linewidth=2, label="VALID, FOLD {}".format(i))
            axes[0, 0].legend()
            axes[0, 0].set_xlabel('epoch', size=15)
            axes[0, 0].set_ylabel('NLLloss', size=15)
            plt.sca(axes[0, 0])
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

            # plot accuracies
            axes[0, 1].plot(np.arange(len(tr_accuracies)), tr_accuracies, lw=1, linewidth=2, label="TRAIN, FOLD {}".format(i))
            axes[0, 1].plot(np.arange(len(val_accuracies)), val_accuracies, lw=1, linewidth=2, label="VALID, FOLD {}".format(i))
            axes[0, 1].legend()
            axes[0, 1].set_xlabel('epoch', size=15)
            axes[0, 1].set_ylabel('Accuracies', size=15)
            plt.sca(axes[0, 1])
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

            # plot AUCS
            axes[1, 0].plot(np.arange(len(tr_auc)), tr_auc, lw=1, linewidth=2, label="TRAIN, FOLD {}".format(i))
            axes[1, 0].plot(np.arange(len(val_auc)), val_auc, lw=1, linewidth=2, label="VALID, FOLD {}".format(i))
            axes[1, 0].legend()
            axes[1, 0].set_xlabel('epoch', size=15)
            axes[1, 0].set_ylabel('AUC', size=15)
            plt.sca(axes[1, 0])
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

        with open(os.path.join(exp_name + "/OUT/" + model_name + "_CONF_MAT.npy"), 'rb') as f:
            total_conf_matrix = np.load(f)

        RFs = [path for path in os.listdir("data") if os.path.isdir(os.path.join("data", path))]

        plt.sca(axes[1, 1])
        conf_df = pd.DataFrame(total_conf_matrix, RFs, RFs)

        sn.set(font_scale=1)
        sn.heatmap(conf_df, annot=True, annot_kws={"size": 15}, fmt='g', cbar=False)
        plt.ylabel("True family", size=15)
        plt.xlabel("Predicted Family", size=15)

        plt.sca(axes[1, 1])
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        fig.tight_layout(pad=0.2)

        plt.savefig("out.png", dpi=300)
        plt.show()
        # print(avg[2]/folds)