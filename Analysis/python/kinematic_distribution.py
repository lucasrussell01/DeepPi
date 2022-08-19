import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd

class distribution_plotter:

    def __init__(self, file_dir):
        self.file_list = glob.glob(file_dir + "/*.pkl")
        self.eta = []
        self.phi = []
        self.p = []
        self.DM = []
        i = 0
        for f in self.file_list:
            df = pd.read_pickle(f)
            i_max = [np.where(df["relp"][i] == np.max(df["relp"][i]))[0] for i in range(10000)]
            for r in range(10000):
                self.eta.append(df["releta"][r][i_max[r]])
                self.phi.append(df["relphi"][r][i_max[r]])
                self.p.append(df["relp"][r][i_max[r]])
                self.DM.append(df["DM"][r])
            i += 1
            print(f"File {i} out of {len(self.file_list)} processed")
        print("All files loaded")
    
    def plot_eta(self, bins, DM = None):
        plt.figure()
        w_plot = bins[1] - bins[0]
        bins = np.arange(bins[0], bins[1] + w_plot/30,  w_plot/30) # histo with 30 bins
        if DM:
            where = np.empty(shape=1, dtype=int)
            for i in DM:
                where = np.concatenate((where, np.where(np.array(self.DM)==i)[0]))
            counts = np.histogram(np.array(self.eta)[where], bins=bins)
        else:
            counts = np.histogram(self.eta, bins=bins)
        bin_width = counts[1][1] - counts[1][0]
        bin_centre = counts[1][:-1] + bin_width/2
        plt.bar(bin_centre, height=counts[0], width = bin_width)
        plt.ylim(0, np.max(counts[0]) + 0.1*np.max(counts[0]))
        plt.xlim(counts[1][0], counts[1][-1])
        plt.title("Relative eta")
        plt.show()

    def plot_phi(self, bins, DM = None):
        plt.figure()
        w_plot = bins[1] - bins[0]
        bins = np.arange(bins[0], bins[1] + w_plot/30,  w_plot/30) # histo with 30 bins
        if DM:
            where = np.empty(shape=1, dtype=int)
            for i in DM:
                where = np.concatenate((where, np.where(np.array(self.DM)==i)[0]))
            counts = np.histogram(np.array(self.phi)[where], bins=bins)
        else:
            counts = np.histogram(self.phi, bins=bins)
        bin_width = counts[1][1] - counts[1][0]
        bin_centre = counts[1][:-1] + bin_width/2
        plt.bar(bin_centre, height=counts[0], width = bin_width)
        plt.ylim(0, np.max(counts[0]) + 0.1*np.max(counts[0]))
        plt.xlim(counts[1][0], counts[1][-1])
        plt.title("Relative phi")
        plt.show()

    def plot_p(self, bins, DM = None):
        plt.figure()
        w_plot = bins[1] - bins[0]
        bins = np.arange(bins[0], bins[1] + w_plot/30,  w_plot/30) # histo with 30 bins
        if DM:
            where = np.empty(shape=1, dtype=int)
            for i in DM:
                where = np.concatenate((where, np.where(np.array(self.DM)==i)[0]))
            counts = np.histogram(np.array(self.p)[where], bins=bins)
        else:
            counts = np.histogram(self.p, bins=bins)
        bin_width = counts[1][1] - counts[1][0]
        bin_centre = counts[1][:-1] + bin_width/2
        plt.bar(bin_centre, height=counts[0], width = bin_width)
        plt.ylim(0, np.max(counts[0]) + 0.1*np.max(counts[0]))
        plt.xlim(counts[1][0], counts[1][-1])
        plt.title("Relative momentum")
        plt.show()


# a = distribution_plotter("/vols/cms/lcr119/Images/Kinematics/Evaluation")
