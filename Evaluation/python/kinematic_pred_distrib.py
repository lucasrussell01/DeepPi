import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import iqr

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
        bins = np.arange(-40, 40+w, w)
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
        bins = np.arange(-40, 40+w, w)
        plt.figure()
        plt.hist(err, bins = bins, histtype="step", color = "blue", label= "CNN")
        plt.hist(HPS_err, bins = bins, histtype="step", color = "red", label= "HPS")
        plt.legend()
        plt.xlabel(r"$p$-$p_{pred}$")
        plt.xlim(-40, 40)
        plt.show()

    def compare_eta(self, p_range = None):
        if p_range:
            df = self.df.loc[(self.df['relp'] >= p_range[0]) & (self.df['relp'] < p_range[1])]
            print("Momentum capped between: ", p_range)
            eta = df["pi0_eta"]
            eta_pred = df["pi0_eta_pred"]
            eta_HPS = df["pi0_eta_HPS"]
        else:
            eta = self.df["pi0_eta"]
            eta_pred = self.df["pi0_eta_pred"]
            eta_HPS = self.df["pi0_eta_HPS"]
        err = eta-eta_pred
        HPS_err = eta - eta_HPS
        w = 0.001
        bins = np.arange(np.min(err), np.max(err)+w, w)
        plt.figure()
        plt.hist(HPS_err, bins = bins, histtype="step", color = "red", label= f"HPS $\mu$={np.mean(HPS_err):.4f} IQR={iqr(HPS_err):.4f}")
        plt.hist(err, bins = bins, histtype="step", color = "blue", label= f"CNN $\mu$={np.mean(err):.4f} IQR={iqr(err):.4f}")
        plt.legend()
        plt.xlabel(r"$\eta$-$\eta_{pred}$")
        plt.xlim(-0.05, 0.05)
        plt.show()

    def plot_releta(self):
        eta = self.df["releta"]
        eta_pred = self.df["releta_pred"]
        err = eta-eta_pred
        w = 0.001
        bins = np.arange(np.min(err), np.max(err)+w, w)
        plt.figure()
        plt.hist(err, bins = bins, histtype="step", color = "blue", label= f"CNN $\mu$={np.mean(err):.4f} IQR={iqr(err):.4f}")
        plt.legend()
        plt.xlabel(r"Rel $\eta$- Rel $\eta_{pred}$")
        plt.xlim(-0.05, 0.05)
        plt.show()

    def compare_phi(self):
        phi = self.df["pi0_phi"]
        phi_pred = self.df["pi0_phi_pred"]
        phi_HPS = self.df["pi0_phi_HPS"]
        err = phi-phi_pred
        HPS_err = phi - phi_HPS
        w = 0.0025
        bins = np.arange(np.min(err), np.max(err)+w, w)
        plt.figure()
        plt.hist(HPS_err, bins = bins, histtype="step", color = "red", label= f"HPS $\mu$={np.mean(HPS_err):.4f} IQR={iqr(HPS_err):.4f}")
        plt.hist(err, bins = bins, histtype="step", color = "blue", label= f"CNN $\mu$={np.mean(err):.4f} IQR={iqr(err):.4f}")
        plt.legend()
        plt.xlabel(r"$\phi$-$\phi_{pred}$")
        plt.xlim(-0.125, 0.125)
        plt.show()

    def compare_profile(self, distrib, save_indv = False):
        p_range = np.concatenate((np.arange(0, 60, 2.5),np.arange(60, 80, 5),np.arange(80, 100, 10), np.arange(100, 170, 20)))
        print(p_range)
        p_centre = p_range[:-1] + np.diff(p_range)/2
        print(p_centre)
        mean_err = []
        mean_HPS_err = []
        std = []
        std_HPS = []
        viqr = []
        viqr_HPS = []
        for i in range(len(p_range)-1):

            # print(f"Processing energies between {p_range[i]} and {p_range[i+1]}")
            df_slice = self.df.loc[(self.df['relp'] >= p_range[i]) & (self.df['relp'] < p_range[i+1])]
            if distrib=="p":
                err = df_slice["pi0_p"] - df_slice["pi0_p_pred"]
                HPS_err = df_slice["pi0_p"] - df_slice["pi0_p_HPS"]
                lab = "$p$-$p_{pred}$"
            elif distrib=="eta":
                err = df_slice["pi0_eta"] - df_slice["pi0_eta_pred"]
                HPS_err = df_slice["pi0_eta"] - df_slice["pi0_eta_HPS"]
                lab = "$\eta$-$\eta_{pred}$"
            elif distrib=="phi":
                err = df_slice["pi0_phi"] - df_slice["pi0_phi_pred"]
                HPS_err = df_slice["pi0_phi"] - df_slice["pi0_phi_HPS"]
                lab = "$\phi$-$\phi_{pred}$"
            mean_err.append(np.mean(err))
            mean_HPS_err.append(np.mean(HPS_err))
            std.append(np.std(err))
            std_HPS.append(np.std(HPS_err))
            viqr.append(iqr(err)) # replace with IQR
            viqr_HPS.append(iqr(HPS_err))
            if save_indv:
                # print(f"Generating {len(p_range)-1} histos")
                bins = np.arange(-40, 41)
                plt.figure()
                plt.hist(err, bins = bins, histtype="step", color = "blue", label= "CNN")
                plt.hist(HPS_err, bins = bins, histtype="step", color = "red", label= "HPS")
                plt.legend()
                plt.xlabel(r"$p$-$p_{pred}$")
                plt.title(f"Momentum Range: {p_range[i]}-{p_range[i+1]} GeV")
                plt.xlim(-40, 40)
                plt.savefig(f"/home/hep/lcr119/Plots/mom_hists/momentum_hist_weighted_{i}.pdf")
                # plt.show()
                plt.close()
        plt.figure()
        plt.plot(p_centre, mean_err, label = "CNN")
        plt.plot(p_centre, mean_HPS_err, label = "HPS")
        plt.legend()
        plt.ylabel(f"Mean {lab}")
        plt.xlabel("Pi0 Momentum")
        plt.show()
        plt.figure()
        plt.plot(p_centre, viqr, label = f"CNN")
        plt.plot(p_centre, viqr_HPS, label = f"HPS")
        plt.legend()
        plt.ylabel(f"IQR {lab}")
        plt.xlabel("Pi0 Momentum")
        plt.show()
        plt.figure()
        plt.plot(p_centre, std, label = f"CNN")
        plt.plot(p_centre, std_HPS, label = f"HPS")
        plt.legend()
        plt.ylabel(f"Standard dev. {lab}")
        plt.xlabel("Pi0 Momentum")
        plt.show()

    def plot_orig_releta(self, p_range=None):
        if p_range:
            df = self.df.loc[(self.df['relp'] >= p_range[0]) & (self.df['relp'] < p_range[1])]
            print("P RANGE: ", p_range)
        else:
            df = self.df
        eta = df["releta"]
        bins = np.arange(-0.2, 0.21, 0.01)
        plt.figure()
        plt.hist(eta, bins=bins, edgecolor='black', label = f" $\mu$={np.mean(eta):.4f} IQR={iqr(eta):.4f}")
        plt.xlabel(r"Rel $\eta$")
        plt.legend()
        plt.show()

    def plot_phi(self, name="CNN"):
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