import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class error_plotter:

    def __init__(self, expID, runID):
        self.expID = expID
        self.runID = runID
        path_to_mlflow = "../../Training/python/mlruns/"
        self.path_to_pred = path_to_mlflow + self.expID + "/" + self.runID + "/artifacts/predictions/kinematic_pred_ggH.pkl"
        self.df = pd.read_pickle(self.path_to_pred)
    
    def plot_momentum(self):
        p = self.df["relp"]
        p_pred = self.df["relp_pred"]
        err = p-p_pred
        w = 1
        bins = np.arange(np.min(err), np.max(err)+w, w)
        plt.figure()
        plt.hist(err, bins = bins, edgecolor='black')
        plt.xlabel("p-p_pred")
        plt.xlim(-75, 75)
        plt.show()

    def compare_momentum(self):
        p = self.df["relp"]
        p_pred = self.df["relp_pred"]
        p_HPS = self.df["pi0_p_HPS"]
        err = p-p_pred
        HPS_err = p - p_HPS
        w = 1
        bins = np.arange(np.min(err), np.max(err)+w, w)
        plt.figure()
        plt.hist(err, bins = bins, histtype="step", color = "blue", label= "DeepPi")
        plt.hist(HPS_err, bins = bins, histtype="step", color = "red", label= "HPS")
        plt.legend()
        plt.xlabel(r"$p$-$p_{pred}$")
        plt.xlim(-40, 40)
        plt.show()

    def compare_momentum_pce(self):
        p = self.df["relp"]
        p_pred = self.df["relp_pred"]
        p_HPS = self.df["pi0_p_HPS"]
        err = (p-p_pred)/p
        HPS_err = (p-p_HPS)/p
        w = 0.05
        bins = np.arange(np.min(err), np.max(err)+w, w)
        plt.figure()
        plt.hist(err, bins = bins, histtype="step", color = "blue", label= "DeepPi")
        plt.hist(HPS_err, bins = bins, histtype="step", color = "red", label= "HPS")
        plt.legend()
        plt.xlabel(r"($p$-$p_{pred}$)/$p$ (Percentage error)")
        plt.xlim(-1, 1)
        plt.show()

    def compare_eta(self):
        eta = self.df["pi0_eta"]
        eta_pred = self.df["pi0_eta_pred"]
        eta_HPS = self.df["pi0_eta_HPS"]
        err = eta-eta_pred
        HPS_err = eta - eta_HPS
        w = 0.005
        bins = np.arange(np.min(err), np.max(err)+w, w)
        plt.figure()
        plt.hist(err, bins = bins, histtype="step", color = "blue", label= "DeepPi")
        plt.hist(HPS_err, bins = bins, histtype="step", color = "red", label= "HPS")
        plt.legend()
        plt.xlabel(r"$\eta$-$\eta_{pred}$")
        plt.xlim(-0.125, 0.125)
        plt.show()

    def compare_phi(self):
        phi = self.df["pi0_phi"]
        phi_pred = self.df["pi0_phi_pred"]
        phi_HPS = self.df["pi0_phi_HPS"]
        err = phi-phi_pred
        HPS_err = phi - phi_HPS
        w = 0.005
        bins = np.arange(np.min(err), np.max(err)+w, w)
        plt.figure()
        plt.hist(err, bins = bins, histtype="step", color = "blue", label= "DeepPi")
        plt.hist(HPS_err, bins = bins, histtype="step", color = "red", label= "HPS")
        plt.legend()
        plt.xlabel(r"$\phi$-$\phi_{pred}$")
        plt.xlim(-0.125, 0.125)
        plt.show()

    def plot_eta(self, name="DeepPi"):
        eta = self.df["pi0_eta"]
        eta_pred = self.df["pi0_eta_pred"]
        err = eta-eta_pred
        w = 0.005
        bins = np.arange(np.min(err), np.max(err)+w, w)
        plt.figure()
        plt.hist(err, bins = bins, edgecolor='black')
        plt.xlabel(r"$\eta$-$\eta_{pred}$")
        plt.text(0.05, 500, name, fontweight="bold", fontsize = 16)
        plt.xlim(-0.125, 0.125)
        plt.show()

    def plot_phi(self, name="DeepPi"):
        phi = self.df["pi0_phi"]
        phi_pred = self.df["pi0_phi_pred"]
        err = phi-phi_pred
        w = 0.005
        bins = np.arange(np.min(err), np.max(err)+w, w)
        plt.figure()
        plt.hist(err, bins = bins, edgecolor='black')
        plt.xlabel(r"$\phi$-$\phi_{pred}$")
        plt.text(0.05, 350, name, fontweight="bold", fontsize = 16)
        plt.xlim(-0.125, 0.125)
        plt.show()

    def plot_x(self):
        x = self.df["pi0_x"]
        x_pred = self.df["pi0_x_pred"]
        err = x-x_pred
        w = 1
        bins = np.arange(np.min(err), np.max(err)+w, w)
        plt.figure()
        plt.hist(err, bins = bins)
        plt.xlabel("x-x_pred")
        plt.xlim(-50, 50)
        plt.show()

    def plot_y(self):
        y = self.df["pi0_y"]
        y_pred = self.df["pi0_y_pred"]
        err = y-y_pred
        w = 1
        bins = np.arange(np.min(err), np.max(err)+w, w)
        plt.figure()
        plt.hist(err, bins = bins)
        plt.xlabel("y-y_pred")
        plt.xlim(-50, 50)
        plt.show()

    def plot_z(self):
        z = self.df["pi0_z"]
        z_pred = self.df["pi0_z_pred"]
        err = z-z_pred
        w = 1
        bins = np.arange(np.min(err), np.max(err)+w, w)
        plt.figure()
        plt.hist(err, bins = bins)
        plt.xlabel("z-z_pred")
        plt.xlim(-50, 50)
        plt.show()