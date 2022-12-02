import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import iqr

class kin_pred:

    def __init__(self, expID, runID, DM=[1, 11], HPSDM=[1, 11], p_range = None):
        self.expID = expID
        self.runID = runID
        path_to_mlflow = "../../Training/python/mlruns/"
        self.path_to_pred = path_to_mlflow + self.expID + "/" + self.runID + "/artifacts/predictions/kinematic_pred.pkl"
        self.HPSDM = HPSDM

        # Select only certain true DM 
        if DM==[1, 11]:
            self.df = pd.read_pickle(self.path_to_pred)
        elif DM == [1]:
            self.df = pd.read_pickle(self.path_to_pred).loc[pd.read_pickle(self.path_to_pred)['DM']==1]
        elif DM == [11]:
            self.df = pd.read_pickle(self.path_to_pred).loc[pd.read_pickle(self.path_to_pred)['DM']==11]
        else:
            raise Exception("DM format not supported")
        print(f"Included True DMs: {DM}")

        # Select only certain HPS DM
        if HPSDM == [1,11]:
            self.df = self.df.loc[(self.df['HPSDM']==1) | (self.df['HPSDM']==11)]
        elif HPSDM == [-1]:
            self.df = self.df.loc[self.df['HPSDM']==-1]
        elif HPSDM == [1]:
            self.df = self.df.loc[self.df['HPSDM']==1]
        elif HPSDM == [0]:
            self.df = self.df.loc[self.df['HPSDM']==0]
        elif HPSDM == [0, 10]:
            self.df = self.df.loc[(self.df['HPSDM']==0) | (self.df['HPSDM']==10)]
        else:
            raise Exception("HPS DM format not supported")
        print(f"Included HPS DMs: {HPSDM}")

        if p_range:
            self.df = self.df.loc[self.df["relp"]>p_range[0]]
            self.df = self.df.loc[self.df["relp"]<p_range[1]]
            print(f"Momentum cut applied, p>{p_range[0]}, p<{p_range[1]}")

        print(f"Number of taus available: {len(self.df['DM'])}")

    def momentum(self):
        p = self.df["relp"]
        p_pred = self.df["relp_pred"]
        err = p-p_pred
        w = 1
        bins = np.arange(-40, 40+w, w)
        plt.figure()
        if self.HPSDM == [1, 11] or self.HPSDM==[1] or self.HPSDM==[11]:
            p_HPS = self.df["pi0_p_HPS"]
            HPS_err = p - p_HPS
            plt.hist(HPS_err, bins = bins, histtype="step", color = "red", label= f"HPS $\mu$={np.mean(HPS_err):.4f} IQR={iqr(HPS_err):.4f}")
        plt.hist(err, bins = bins, histtype="step", color = "blue", label= f"CNN $\mu$={np.mean(err):.4f} IQR={iqr(err):.4f}")
        plt.legend()
        plt.xlabel(r"$p$-$p_{pred}$")
        plt.xlim(-40, 40)
        
        plt.show()

    def eta(self):
        eta = self.df["pi0_eta"]
        eta_pred = self.df["pi0_eta_pred"] 
        err = eta-eta_pred
        w = 0.001
        bins = np.arange(-0.02, 0.021, 0.001)

        plt.figure()
        if self.HPSDM == [1, 11] or self.HPSDM==[1] or self.HPSDM==[11]:
            eta_HPS = self.df["pi0_eta_HPS"]
            HPS_err = eta - eta_HPS
            plt.hist(HPS_err, bins = bins, histtype="step", color = "red", label= f"HPS $\mu$={np.mean(HPS_err):.4f} IQR={iqr(HPS_err):.4f}")
        plt.hist(err, bins = bins, histtype="step", color = "blue", label= f"CNN $\mu$={np.mean(err):.4f} IQR={iqr(err):.4f}")
        plt.legend()
        plt.xlabel(r"$\eta$-$\eta_{pred}$")
        plt.xlim(-0.02, 0.02)
        plt.show()

    def phi(self):
        phi = self.df["pi0_phi"]
        phi_pred = self.df["pi0_phi_pred"]
        err = phi-phi_pred
        w = 0.0025
        bins = np.arange(-0.03, 0.0325, 0.0025)
        
        plt.figure()
        if self.HPSDM == [1, 11] or self.HPSDM==[1] or self.HPSDM==[11]:
            phi_HPS = self.df["pi0_phi_HPS"]
            HPS_err = phi - phi_HPS
            plt.hist(HPS_err, bins = bins, histtype="step", color = "red", label= f"HPS $\mu$={np.mean(HPS_err):.4f} IQR={iqr(HPS_err):.4f}")
        plt.hist(err, bins = bins, histtype="step", color = "blue", label= f"CNN $\mu$={np.mean(err):.4f} IQR={iqr(err):.4f}")
        plt.legend()
        plt.xlabel(r"$\phi$-$\phi_{pred}$")
        plt.xlim(-0.03, 0.03)


    def releta(self, p_range = None):
        
        eta = self.df["releta"]
        eta_pred = self.df["releta_pred"]
        err = eta-eta_pred
        w = 0.001
        bins = np.arange(-0.02, 0.021, 0.001)

        plt.figure()
        if self.HPSDM == [1, 11] or self.HPSDM==[1] or self.HPSDM==[11]:
            eta_HPS = self.df["rel_eta_HPS"]
            HPS_err = eta - eta_HPS
            plt.hist(HPS_err, bins = bins, histtype="step", color = "red", label= f"HPS $\mu$={np.mean(HPS_err):.4f} IQR={iqr(HPS_err):.4f}")
        plt.hist(err, bins = bins, histtype="step", color = "blue", label= f"CNN $\mu$={np.mean(err):.4f} IQR={iqr(err):.4f}")
        plt.legend()
        plt.xlabel(r"rel $\eta$- rel $\eta_{pred}$")
        plt.xlim(-0.02, 0.02)
        plt.show()
        

    def relphi(self, p_range = None):
        
        phi = self.df["relphi"]
        phi_pred = self.df["relphi_pred"]
        err = phi-phi_pred
        w = 0.0025
        bins = np.arange(-0.03, 0.0325, 0.0025)

        plt.figure()
        if self.HPSDM == [1, 11] or self.HPSDM==[1] or self.HPSDM==[11]:
            phi_HPS = self.df["rel_phi_HPS"]
            HPS_err = phi - phi_HPS
            plt.hist(HPS_err, bins = bins, histtype="step", color = "red", label= f"HPS $\mu$={np.mean(HPS_err):.4f} IQR={iqr(HPS_err):.4f}")
        plt.hist(err, bins = bins, histtype="step", color = "blue", label= f"CNN $\mu$={np.mean(err):.4f} IQR={iqr(err):.4f}")
        plt.legend()
        plt.xlabel(r"rel $\phi$- rel $\phi_{pred}$")
        plt.xlim(-0.05, 0.05)
        plt.show()


    def profile(self, distrib):

        if distrib=="phi" or distrib=="eta":
            p_range = np.concatenate((np.arange(0, 100, 10), np.arange(110, 170, 20))) #np.arange(0, 60, 5),np.arange(60, 80, 5), 
        elif distrib=="p":
            p_range = np.concatenate((np.arange(0, 60, 2.5),np.arange(60, 80, 5), np.arange(80, 100, 10), np.arange(110, 170, 20)))
        
        p_centre = p_range[:-1] + np.diff(p_range)/2
        width = np.diff(p_range)
        mean_err = []
        mean_HPS_err = []
        std = []
        std_HPS = []
        viqr = []
        viqr_HPS = []
        mean_pce = []
        mean_HPS_pce = []
        pce_iqr = []
        HPS_pce_iqr = []
        pce_std = []
        HPS_pce_std = []
        for i in range(len(p_range)-1):
            df_slice = self.df.loc[(self.df['relp'] >= p_range[i]) & (self.df['relp'] < p_range[i+1])]
            if distrib=="p":
                err = df_slice["relp"] - df_slice["relp_pred"]
                HPS_err = df_slice["relp"] - df_slice["pi0_p_HPS"]
                lab = "$p$-$p_{pred}$"
                pce = err/df_slice["relp"]
                HPS_pce = HPS_err/df_slice["relp"]
            elif distrib=="eta":
                err = df_slice["pi0_eta"] - df_slice["pi0_eta_pred"]
                HPS_err = df_slice["pi0_eta"] - df_slice["pi0_eta_HPS"]
                lab = "$\eta$-$\eta_{pred}$"
                pce = err/df_slice["pi0_eta"]
                HPS_pce = HPS_err/df_slice["pi0_eta"]
            elif distrib=="phi":
                err = df_slice["pi0_phi"] - df_slice["pi0_phi_pred"]
                HPS_err = df_slice["pi0_phi"] - df_slice["pi0_phi_HPS"]
                lab = "$\phi$-$\phi_{pred}$"
                pce = err/df_slice["pi0_phi"]
                HPS_pce = HPS_err/df_slice["pi0_phi"]
            mean_err.append(np.mean(err))
            mean_HPS_err.append(np.mean(HPS_err))
            std.append(np.std(err))
            std_HPS.append(np.std(HPS_err))
            viqr.append(iqr(err)) # replace with IQR
            viqr_HPS.append(iqr(HPS_err))
            mean_pce.append(np.mean(pce))
            mean_HPS_pce.append(np.mean(HPS_pce))
            pce_iqr.append(iqr(pce))
            HPS_pce_iqr.append(iqr(HPS_pce))
            pce_std.append(np.std(pce))
            HPS_pce_std.append(np.std(HPS_pce))

        fig, ax1 = plt.subplots(1, 1, figsize=(6,6))
        ax1.minorticks_on()
        ax1.grid()
        
        ax1.errorbar(p_centre, (mean_err), xerr=width/2, marker = "o", linestyle="", label = f"CNN")
        if self.HPSDM == [1, 11] or self.HPSDM==[1] or self.HPSDM==[11]:
            ax1.errorbar(p_centre, (mean_HPS_err), xerr=width/2, marker = "o", linestyle="", label = f"HPS")
        ax1.legend()
        ax1.set_ylabel(f"Mean {lab}")
        ax1.set_xlabel(r"$\pi^0$ Momentum")
        ax1.set_xlim(0, 150)
        plt.show()

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex="col", gridspec_kw={'height_ratios': [6, 1]}, figsize=(6,6))
        fig.subplots_adjust(hspace=0.05) 
        ax1.minorticks_on()
        ax1.grid()
        ax1.errorbar(p_centre, viqr, xerr=width/2, marker = "o", linestyle="", label = f"CNN")
        if self.HPSDM == [1, 11] or self.HPSDM==[1] or self.HPSDM==[11]:
            ax1.errorbar(p_centre, viqr_HPS, xerr=width/2, marker = "o", linestyle="", label = f"HPS")
            return mean_err, viqr, mean_HPS_err, viqr_HPS
        else:
            return  mean_err, viqr
        
            # ax1.errorbar([200], [1], xerr=[1], label = "CNN/HPS Ratio", marker="o", linestyle="", color="black")
        ax1.legend()
        ax1.set_ylabel(f"IQR {lab}")
        ax2.set_xlabel(r"$\pi^0$ Momentum")
        ax2.grid()

        if distrib=="phi" or distrib=="eta":
            # ax1.set_ylim(0, 0.03)
            ax2.set_ylim(0.5, 2.5)
        if self.HPSDM == [1, 11] or self.HPSDM==[1] or self.HPSDM==[11]:
            ax2.errorbar(p_centre, np.array(viqr)/np.array(viqr_HPS), xerr=width/2, marker="o", linestyle="", color="black")
            ax2.set_ylabel("CNN/HPS")

        ax1.set_xlim(0, 150)
        plt.show()

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex="col", gridspec_kw={'height_ratios': [6, 1]}, figsize=(6,6))
        fig.subplots_adjust(hspace=0.05) 
        ax1.minorticks_on()
        ax1.grid()
        ax1.errorbar(p_centre, std, xerr=width/2, marker = "o", linestyle="", label = f"CNN")
        if self.HPSDM == [1, 11] or self.HPSDM==[1] or self.HPSDM==[11]:
            ax1.errorbar(p_centre, std_HPS, xerr=width/2, marker = "o", linestyle="", label = f"HPS")
            ax1.errorbar([200], [1], xerr=[1], label = "CNN/HPS Ratio", marker="o", linestyle="", color="black")
            ax1.scatter([200], [1], label = "CNN/HPS Ratio", color="black")
        ax1.legend()
        ax1.set_ylabel(f"Standard dev. {lab}")
        ax2.set_xlabel(r"$\pi^0$ Momentum")
        ax2.grid()
        if self.HPSDM == [1, 11] or self.HPSDM==[1] or self.HPSDM==[11]:
            ax2.errorbar(p_centre, np.array(std)/np.array(std_HPS), xerr=width/2, marker="o", linestyle="", color="black")
            ax1.set_xlim(0, 150)
            ax2.set_ylabel("CNN/HPS")
        plt.show()



    