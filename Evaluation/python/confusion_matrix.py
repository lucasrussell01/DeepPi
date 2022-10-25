import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import seaborn as sn
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="-1" # don't need to use GPU
matplotlib.rcParams.update({'font.size': 12})
sn.set(font_scale=1.4)
# path_to_mlflow = "../../Training/python/mlruns/"





class cm_plotter:

    def __init__(self, expID, runID, HPSonly = True):
        self.expID = expID
        self.runID = runID
        path_to_mlflow = "../../Training/python/mlruns/"
        if not HPSonly:
            self.path_to_pred = path_to_mlflow + self.expID + "/" + self.runID + "/artifacts/predictions/pred_DM.pkl"
        else:
            self.path_to_pred = path_to_mlflow + self.expID + "/" + self.runID + "/artifacts/predictions/pred_DM_HPS_only.pkl"
        self.savepath = path_to_mlflow + self.expID + "/" + self.runID + "/artifacts/predictions"
        self.df = pd.read_pickle(self.path_to_pred)
        self.df['MVA_pred'][np.where(self.df["MVA_pred"]==-1)[0]] = 100
        self.where_threeprong = np.concatenate((np.where(self.df['truthDM']==10)[0], np.where(self.df['truthDM']==11)[0]))
        self.where_oneprong = np.concatenate(( np.where(self.df['truthDM']==0)[0], np.where(self.df['truthDM']==1)[0],
                                                 np.where(self.df['truthDM']==2)[0]))
        self.axis_lab = ["0 $\pi^0$", "1 $\pi^0$", "2 $\pi^0$"]

    def plot_cm(self, pred="CNN_pred", type="efficiency"):
        if type=="efficiency":
            cm = confusion_matrix(self.df["truth"], self.df[pred], normalize='true')
        elif type=="purity":
            cm = confusion_matrix(self.df["truth"], self.df[pred], normalize='pred') 
        # print(cm)
        plt.figure(figsize=(8,6.4))
        plt.axhline(y = 0, color='k',linewidth = 3)
        # if pred!="MVA_pred":
        plt.axhline(y = 3, color = 'k', linewidth = 3)
        # else:
            # plt.axhline(y = 4, color = 'k', linewidth = 3)
        plt.axvline(x = 0, color = 'k',linewidth = 3)
        # if pred!="MVA_pred":
        plt.axvline(x = 3, color = 'k', linewidth = 3)
        # else:
            # plt.axvline(x = 4, color = 'k', linewidth = 3)
        if pred=="MVA_pred":
            sn.heatmap(cm, annot=True, cmap='Blues', xticklabels = ["0 $\pi^0$", "1 $\pi^0$", "2 $\pi^0$", "other"], yticklabels = ["0 $\pi^0$", "1 $\pi^0$", "2 $\pi^0$", "other"], annot_kws={"fontsize":12})
        else:
            sn.heatmap(cm, annot=True, cmap='Blues', xticklabels = self.axis_lab, yticklabels = self.axis_lab, annot_kws={"fontsize":12})
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title("All DMs", loc='right')
        plt.text(0.03, -0.05, pred, fontsize=18, fontweight="bold")
        # plt.text(0.03, -0.05, "CMS ", fontsize=18, fontweight="bold")
        # plt.text(0.4, -0.05, "Work in progess ", fontsize=18, fontstyle="italic")
        plt.show() 
    
    def plot_cm_oneP(self, pred="CNN_pred", type="efficiency"):
        if type=="efficiency":
            cm = confusion_matrix(self.df["truth"][self.where_oneprong], self.df[pred][self.where_oneprong], normalize='true')
        elif type=="purity":
            cm = confusion_matrix(self.df["truth"][self.where_oneprong], self.df[pred][self.where_oneprong], normalize='pred')
        # print(cm)
        plt.figure(figsize=(8,6.4))
        plt.axhline(y = 0, color='k',linewidth = 3)
        # if pred!="MVA_pred":
        plt.axhline(y = 3, color = 'k', linewidth = 3)
        # else:
        #     plt.axhline(y = 4, color = 'k', linewidth = 3)
        plt.axvline(x = 0, color = 'k',linewidth = 3)
        # if pred!="MVA_pred":
        plt.axvline(x = 3, color = 'k', linewidth = 3)
        # else:
        #     plt.axvline(x = 4, color = 'k', linewidth = 3)
        # if pred=="MVA_pred":
        #     sn.heatmap(cm, annot=True, cmap='Blues', xticklabels = ["0 $\pi^0$", "1 $\pi^0$", "2 $\pi^0$", "other"], yticklabels = ["0 $\pi^0$", "1 $\pi^0$", "2 $\pi^0$", "other"], annot_kws={"fontsize":12})
        # else:
        sn.heatmap(cm, annot=True, cmap='Blues', xticklabels = self.axis_lab, yticklabels = self.axis_lab, annot_kws={"fontsize":12})
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title("Single Prong DMs", loc='right')
        plt.text(0.03, -0.05, pred, fontsize=18, fontweight="bold")
        # plt.text(0.03, -0.05, "CMS ", fontsize=18, fontweight="bold")
        # plt.text(0.4, -0.05, "Work in progess ", fontsize=18, fontstyle="italic")
        
        plt.savefig(f"{self.savepath}/{pred}_one_prong_{type}.pdf", bbox_inches="tight")
        # plt.show()
    
    def plot_cm_threeP(self, pred="CNN_pred", type="efficiency"):

        threeP_pred = np.array(self.df[pred][self.where_threeprong])
        threeP_pred[np.where(threeP_pred==2)[0]] = 1
        if type=="efficiency":
            cm = confusion_matrix(self.df["truth"][self.where_threeprong], threeP_pred, normalize='true')
        elif type=="purity":
            cm = confusion_matrix(self.df["truth"][self.where_threeprong], threeP_pred, normalize='pred')
        # print(cm)
        plt.figure(figsize=(8,6.4))
        plt.axhline(y = 0, color='k',linewidth = 3)
        # if pred!="MVA_pred":
        plt.axvline(x = 2, color = 'k', linewidth = 3)
        # else:
            # plt.axvline(x = 3, color = 'k', linewidth = 3)
        plt.axvline(x = 0, color = 'k',linewidth = 3)
        # if pred!="MVA_pred":
        plt.axhline(y = 2, color = 'k', linewidth = 3)
        # else:
            # plt.axhline(y = 3, color = 'k', linewidth = 3)
        # if pred=="MVA_pred":
        #     sn.heatmap(cm, annot=True, cmap='Blues', xticklabels = ["0 $\pi^0$", "$\geq$ 1 $\pi^0$", "other"], yticklabels = ["0 $\pi^0$", " 1 $\pi^0$","other"], annot_kws={"fontsize":12})
        # else:
        sn.heatmap(cm, annot=True, cmap='Blues', xticklabels = ["0 $\pi^0$", "$\geq$ 1 $\pi^0$"], yticklabels = ["0 $\pi^0$", "1 $\pi^0$"], annot_kws={"fontsize":12})
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title("Three Prong DMs", loc='right')
        plt.text(0.03, -0.05, pred, fontsize=18, fontweight="bold")
        plt.savefig(f"{self.savepath}/{pred}_three_prong_{type}.pdf", bbox_inches="tight")
        # plt.text(0.03, -0.05, "CMS ", fontsize=18, fontweight="bold")
        # plt.text(0.4, -0.05, "Work in progess ", fontsize=18, fontstyle="italic")
        # plt.show()


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
        print(f"Fraction of DM0: {len(np.where(self.df['truthDM']==0)[0])/10000}")
        print(f"Fraction of DM1: {len(np.where(self.df['truthDM']==1)[0])/10000}")
        print(f"Fraction of DM2: {len(np.where(self.df['truthDM']==2)[0])/10000}")
        print(f"Fraction of DM10: {len(np.where(self.df['truthDM']==10)[0])/10000}")
        print(f"Fraction of DM11: {len(np.where(self.df['truthDM']==11)[0])/10000}")
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