import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

# path_to_mlflow = "../../Training/python/mlruns/"
# expID = "3"
# runID = "c481a804abcf441aaaf23e2a5870c98f"


class metric_plotter:

    def __init__(self, expID, runID):
        self.expID = expID
        self.runID = runID
        self.path_to_mlflow = "../../Training/python/mlruns/"
        self.path_to_metrics = self.path_to_mlflow + self.expID + "/" + self.runID + "/metrics"
        with open(f'{self.path_to_mlflow}{self.expID}/{self.runID}/artifacts/input_cfg/hydra/config.yaml') as file:
            self.cfg = yaml.full_load(file)

    def plot_metric(self, metric_name, verbose=False):
        metric = []
        val_metric = []
        # if started from pre trained model, load those params
        if self.cfg["pretrained"] is not None:
            expID = self.cfg["pretrained"]["experiment_id"]
            runID = self.cfg["pretrained"]["run_id"]
            print(f"Note: Model training was started from {runID} ({expID})")
            path_to_old = f"{self.path_to_mlflow}{expID}/{runID}/metrics"
            # check if pretrained model was pretrained :0
            with open(f'{self.path_to_mlflow}{expID}/{runID}/artifacts/input_cfg/hydra/config.yaml') as file:
                pre_cfg = yaml.full_load(file)
            if pre_cfg["pretrained"] is not None:
                pre_expID = pre_cfg["pretrained"]["experiment_id"]
                pre_runID = pre_cfg["pretrained"]["run_id"]
                print(f"Which was started from {pre_runID} ({pre_expID})")
                pre_path = f"{self.path_to_mlflow}{pre_expID}/{pre_runID}/metrics"
                with open(f"{pre_path}/{metric_name}") as file:
                    for e in file:
                        vals = e.split(" ")
                        metric.append(float(vals[1]))
                with open(f"{pre_path}/val_{metric_name}") as file:
                    for e in file:
                        vals = e.split(" ")
                        val_metric.append(float(vals[1]))

            with open(f"{path_to_old}/{metric_name}") as file:
                for e in file:
                    vals = e.split(" ")
                    metric.append(float(vals[1]))
            with open(f"{path_to_old}/val_{metric_name}") as file:
                for e in file:
                    vals = e.split(" ")
                    val_metric.append(float(vals[1]))
        # add metrics from the training
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
