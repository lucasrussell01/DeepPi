import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import seaborn as sn
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="-1" # don't need to use GPU
mpl.rcParams.update({'font.size': 12})
sn.set(font_scale=1.4)
# path_to_mlflow = "../../Training/python/mlruns/"


# expID = "3"
# runID = "c481a804abcf441aaaf23e2a5870c98f"


class cm_plotter:

    def __init__(self, expID, runID):
        self.expID = expID
        self.runID = runID
        path_to_mlflow = "../../Training/python/mlruns/"
        self.path_to_pred = path_to_mlflow + self.expID + "/" + self.runID + "/artifacts/predictions/pred_ggH.pkl"
        self.df = pd.read_pickle(self.path_to_pred)
        self.where_threeprong = np.concatenate((np.where(self.df["truthDM"]==10)[0], np.where(self.df["truthDM"]==11)[0]))
        self.where_oneprong = np.concatenate(( np.where(self.df["truthDM"]==0)[0], np.where(self.df["truthDM"]==1)[0],
                                                 np.where(self.df["truthDM"]==2)[0]))
        self.axis_lab = ["0 $\pi^0$", "1 $\pi^0$", "2 $\pi^0$"]

    def plot_cm(self):
        cm = confusion_matrix(self.df["truth"], self.df["max_pred"], normalize='true')
        # print(cm)
        plt.figure(figsize=(8,6.4))
        plt.axhline(y = 0, color='k',linewidth = 3)
        plt.axhline(y = 3, color = 'k', linewidth = 3)
        plt.axvline(x = 0, color = 'k',linewidth = 3)
        plt.axvline(x = 3, color = 'k', linewidth = 3)
        sn.heatmap(cm, annot=True, cmap='Blues', xticklabels = self.axis_lab, yticklabels = self.axis_lab, annot_kws={"fontsize":12})
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title("All DMs", loc='right')
        plt.text(0.03, -0.05, "CMS ", fontsize=18, fontweight="bold")
        plt.text(0.4, -0.05, "Work in progess ", fontsize=18, fontstyle="italic")
        plt.show() 
    
    def plot_raw_cm(self):
        cm = confusion_matrix(self.df["truth"], self.df["max_pred"])
        # print(cm)
        plt.figure(figsize=(8,6.4))
        plt.axhline(y = 0, color='k',linewidth = 3)
        plt.axhline(y = 3, color = 'k', linewidth = 3)
        plt.axvline(x = 0, color = 'k',linewidth = 3)
        plt.axvline(x = 3, color = 'k', linewidth = 3)
        sn.heatmap(cm, annot=True, cmap='Blues', xticklabels = self.axis_lab, yticklabels = self.axis_lab, annot_kws={"fontsize":12})
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title("All DMs", loc='right')
        print(f"Fraction of DM0: {len(np.where(self.df["truthDM"]==0)[0])/10000}")
        print(f"Fraction of DM1: {len(np.where(self.df["truthDM"]==1)[0])/10000}")
        print(f"Fraction of DM2: {len(np.where(self.df["truthDM"]==2)[0])/10000}")
        print(f"Fraction of DM10: {len(np.where(self.df["truthDM"]==10)[0])/10000}")
        print(f"Fraction of DM11: {len(np.where(self.df["truthDM"]==11)[0])/10000}")
        plt.text(0.03, -0.05, "CMS ", fontsize=18, fontweight="bold")
        plt.text(0.4, -0.05, "Work in progess ", fontsize=18, fontstyle="italic")
        plt.show()
    
    def plot_cm_oneP(self):
        cm = confusion_matrix(self.df["truth"][self.where_oneprong], self.df["max_pred"][self.where_oneprong], normalize='true')
        # print(cm)
        plt.figure(figsize=(8,6.4))
        plt.axhline(y = 0, color='k',linewidth = 3)
        plt.axhline(y = 3, color = 'k', linewidth = 3)
        plt.axvline(x = 0, color = 'k',linewidth = 3)
        plt.axvline(x = 3, color = 'k', linewidth = 3)
        sn.heatmap(cm, annot=True, cmap='Blues', xticklabels = self.axis_lab, yticklabels = self.axis_lab, annot_kws={"fontsize":12})
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title("Single Prong DMs", loc='right')
        plt.text(0.03, -0.05, "CMS ", fontsize=18, fontweight="bold")
        plt.text(0.4, -0.05, "Work in progess ", fontsize=18, fontstyle="italic")
        plt.show()
    
    def plot_cm_threeP(self):
        cm = confusion_matrix(self.df["truth"][self.where_threeprong], self.df["max_pred"][self.where_threeprong], normalize='true')
        # print(cm)
        plt.figure(figsize=(8,6.4))
        plt.axhline(y = 0, color='k',linewidth = 3)
        plt.axhline(y = 3, color = 'k', linewidth = 3)
        plt.axvline(x = 0, color = 'k',linewidth = 3)
        plt.axvline(x = 3, color = 'k', linewidth = 3)
        plt.text(0.03, -0.05, "CMS ", fontsize=18, fontweight="bold")
        sn.heatmap(cm, annot=True, cmap='Blues', xticklabels = self.axis_lab, yticklabels = self.axis_lab, annot_kws={"fontsize":12})
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title("Three Prong DMs", loc='right')
        plt.text(0.03, -0.05, "CMS ", fontsize=18, fontweight="bold")
        plt.text(0.4, -0.05, "Work in progess ", fontsize=18, fontstyle="italic")
        plt.show()

    def plot_raw_cm_oneP(self):
        cm = confusion_matrix(self.df["truth"][self.where_oneprong], self.df["max_pred"][self.where_oneprong])
        # print(cm)
        plt.figure(figsize=(8,6.4))
        plt.axhline(y = 0, color='k',linewidth = 3)
        plt.axhline(y = 3, color = 'k', linewidth = 3)
        plt.axvline(x = 0, color = 'k',linewidth = 3)
        plt.axvline(x = 3, color = 'k', linewidth = 3)
        plt.text(0.03, -0.05, "CMS ", fontsize=18, fontweight="bold")
        sn.heatmap(cm, annot=True, cmap='Blues', xticklabels = self.axis_lab, yticklabels = self.axis_lab, annot_kws={"fontsize":12})
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title("Single Prong DMs", loc='right')
        plt.text(0.03, -0.05, "CMS ", fontsize=18, fontweight="bold")
        plt.text(0.4, -0.05, "Work in progess ", fontsize=18, fontstyle="italic")
        plt.show()
    
    def plot_raw_cm_threeP(self):
        cm = confusion_matrix(self.df["truth"][self.where_threeprong], self.df["max_pred"][self.where_threeprong])
        # print(cm)
        plt.figure(figsize=(8,6.4))
        plt.axhline(y = 0, color='k',linewidth = 3)
        plt.axhline(y = 3, color = 'k', linewidth = 3)
        plt.axvline(x = 0, color = 'k',linewidth = 3)
        plt.axvline(x = 3, color = 'k', linewidth = 3)
        plt.text(0.03, -0.05, "CMS ", fontsize=18, fontweight="bold")
        sn.heatmap(cm, annot=True, cmap='Blues', xticklabels = self.axis_lab, yticklabels = self.axis_lab, annot_kws={"fontsize":12})
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title("Three Prong DMs", loc='right')
        plt.text(0.4, -0.05, "Work in progess ", fontsize=18, fontstyle="italic")
        plt.show()

# plotter = cm_plotter(expID, runID)
# plotter.plot_cm()
# plotter.plot_cm_oneP()