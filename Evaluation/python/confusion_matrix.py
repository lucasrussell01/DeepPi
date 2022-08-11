import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="-1" # don't need to use GPU

# path_to_mlflow = "../../Training/python/mlruns/"


expID = "3"
runID = "c481a804abcf441aaaf23e2a5870c98f"


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

    def plot_cm(self):
        cm = confusion_matrix(self.df["truth"], self.df["max_pred"], normalize='true')
        print(cm)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1, 2])
        cm_display.plot()
        plt.title("All DMs")
        plt.show() 
    
    def plot_raw_cm(self):
        cm = confusion_matrix(self.df["truth"], self.df["max_pred"])
        print(type(cm))
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1, 2])
        cm_display.plot()
        plt.title("All DMs")
        plt.show()
    
    def plot_cm_oneP(self):
        cm = confusion_matrix(self.df["truth"][self.where_oneprong], self.df["max_pred"][self.where_oneprong], normalize='true')
        print(cm)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1, 2])
        cm_display.plot()
        plt.title("Single Prong DMs")
        plt.show()
    
    def plot_cm_threeP(self):
        cm = confusion_matrix(self.df["truth"][self.where_threeprong], self.df["max_pred"][self.where_threeprong], normalize='true')
        print(cm)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1, 2])
        cm_display.plot()
        plt.title("Three Prong DMs")
        plt.show()

    def plot_raw_cm_oneP(self):
        cm = confusion_matrix(self.df["truth"][self.where_oneprong], self.df["max_pred"][self.where_oneprong])
        print(cm)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1, 2])
        cm_display.plot()
        plt.title("Single Prong DMs")
        plt.show()
    
    def plot_raw_cm_threeP(self):
        cm = confusion_matrix(self.df["truth"][self.where_threeprong], self.df["max_pred"][self.where_threeprong])
        print(cm)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1, 2])
        cm_display.plot()
        plt.title("Three Prong DMs")
        plt.show()

# plotter = cm_plotter(expID, runID)
# plotter.plot_cm()
# plotter.plot_cm_oneP()