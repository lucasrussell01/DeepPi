import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# path_to_mlflow = "../../Training/python/mlruns/"
# expID = "3"
# runID = "c481a804abcf441aaaf23e2a5870c98f"


class metric_plotter:

    def __init__(self, expID, runID):
        self.expID = expID
        self.runID = runID
        path_to_mlflow = "../../Training/python/mlruns/"
        self.path_to_metrics = path_to_mlflow + self.expID + "/" + self.runID + "/metrics"

    def plot_metric(self, metric_name, verbose=False):
        metric = []
        val_metric = []
        with open(f"{self.path_to_metrics}/{metric_name}") as file:
            for e in file:
                vals = e.split(" ")
                metric.append(float(vals[1]))
        with open(f"{self.path_to_metrics}/val_{metric_name}") as file:
            for e in file:
                vals = e.split(" ")
                val_metric.append(float(vals[1]))
        epochs = range(len(metric))
        if verbose:
            print("Training: ", metric)
            print("Validation: ", val_metric)
        plt.figure()
        plt.plot(epochs, metric, marker="o", color="blue", label = metric_name + " (Training)")
        plt.plot(epochs, val_metric, marker="o", color="red", label = metric_name + " (Validation)")
        plt.xlabel("Epoch")
        plt.legend()
        plt.show()
        # plt.savefig(f"{path_to_metrics}/{metric_name}.pdf")
