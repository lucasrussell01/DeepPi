import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import iqr

class error_plotter:

    def __init__(self, expID, runID):
        self.expID = expID
        self.runID = runID
        path_to_mlflow = "../../Training/python/mlruns/"
        self.path_to_pred = path_to_mlflow + self.expID + "/" + self.runID + "/artifacts/predictions/debug_kinematic_pred_ggH.pkl"
        self.df = pd.read_pickle(self.path_to_pred)
    
    def gen_momentum(self, density=False):
        p = self.df["relp"]
        plt.figure()
        bins = np.arange(0, 155, 5)
        hist = plt.hist(p, bins= np.arange(0, 205, 5), density=density, edgecolor='black')
        print(f"{len(bins)-1} Bins, target count in each bin: {1/(len(bins)-1)}")
        plt.xlabel(r"$\pi^0$ Momentum")
        plt.xlim(0, 200)
        
        plt.savefig(f"/vols/cms/lcr119/Plots/etaphi/genmom.pdf", bbox_inches="tight")
        plt.show()
        return hist

    def plot_image(self, entry):
        plt.figure()
        plt.xlim(-0.2871, 0.2971)
        plt.ylim(-0.2871, 0.2971)
        print(f"Pi0 momentum for this entry is: ", self.df["relp"][entry])
        plt.scatter(self.df["relphi"][entry], self.df["releta"][entry], label="True position")
        plt.scatter(self.df["relphi_pred"][entry], self.df["releta_pred"][entry], label="CNN pred")
        plt.scatter(self.df["rel_phi_HPS"][entry], self.df["rel_eta_HPS"][entry], label="HPS pred")
        plt.legend()
        plt.show()

        plt.figure()
        plt.xlim(self.df["relphi"][entry]-0.02, self.df["relphi"][entry]+0.02)
        plt.ylim(self.df["releta"][entry]-0.02, self.df["releta"][entry]+0.02)
        plt.title("ZOOM")
        plt.scatter(self.df["relphi"][entry], self.df["releta"][entry], label="True position")
        plt.scatter(self.df["relphi_pred"][entry], self.df["releta_pred"][entry], label="CNN pred")
        plt.scatter(self.df["rel_phi_HPS"][entry], self.df["rel_eta_HPS"][entry], label="HPS pred")
        plt.legend()
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
        plt.hist(HPS_err, bins = bins, histtype="step", color = "red", label= f"HPS $\mu$={np.mean(HPS_err):.4f} IQR={iqr(HPS_err):.4f}")
        plt.hist(err, bins = bins, histtype="step", color = "blue", label= f"CNN $\mu$={np.mean(err):.4f} IQR={iqr(err):.4f}")

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
            eta_HPSPV = self.df["pi0_eta_HPSPV"]
        err = eta-eta_pred
        HPS_err = eta - eta_HPS
        HPSPV_err = eta- eta_HPSPV
        w = 0.001
        bins = np.arange(np.min(err), np.max(err)+w, w)
        plt.figure()
        plt.hist(HPS_err, bins = bins, histtype="step", color = "red", label= f"HPS $\mu$={np.mean(HPS_err):.4f} IQR={iqr(HPS_err):.4f}")
        # plt.hist(HPSPV_err, bins = bins, histtype="step", color = "green", label= f"HPSPV $\mu$={np.mean(HPSPV_err):.4f} IQR={iqr(HPSPV_err):.4f}")
        plt.hist(err, bins = bins, histtype="step", color = "blue", label= f"CNN $\mu$={np.mean(err):.4f} IQR={iqr(err):.4f}")
        plt.legend()
        plt.xlabel(r"$\eta$-$\eta_{pred}$")
        plt.xlim(-0.02, 0.02)
        # plt.ylim(0, 75000)
        plt.savefig(f"/vols/cms/lcr119/Plots/etaphi/eta.pdf", bbox_inches="tight")
        plt.show()
        plt.figure()
        diff = np.arange(-0.001, 0.001, 0.0001)
        plt.hist(eta_HPS-eta_HPSPV, bins=diff, histtype="step")
        plt.title("HPS-HPS PV")
        plt.show()

    def compare_releta(self, p_range = None):
        
        eta = self.df["releta"]
        eta_pred = self.df["releta_pred"]
        eta_HPS = self.df["rel_eta_HPS"]
        err = eta-eta_pred
        HPS_err = eta - eta_HPS
        w = 0.001
        bins = np.arange(np.min(err), np.max(err)+w, w)
        plt.figure()
        plt.hist(HPS_err, bins = bins, histtype="step", color = "red", label= f"HPS $\mu$={np.mean(HPS_err):.4f} IQR={iqr(HPS_err):.4f}")
        plt.hist(err, bins = bins, histtype="step", color = "blue", label= f"CNN $\mu$={np.mean(err):.4f} IQR={iqr(err):.4f}")
        plt.legend()
        plt.xlabel(r"rel $\eta$- rel $\eta_{pred}$")
        plt.xlim(-0.02, 0.02)
        # plt.ylim(0, 75000)
        # plt.savefig(f"/vols/cms/lcr119/Plots/etaphi/eta.pdf", bbox_inches="tight")
        plt.show()
        

    def compare_relphi(self, p_range = None):
        
        phi = self.df["relphi"]
        phi_pred = self.df["relphi_pred"]
        phi_HPS = self.df["rel_phi_HPS"]
        err = phi-phi_pred
        HPS_err = phi - phi_HPS
        w = 0.0025
        bins = np.arange(np.min(err), np.max(err)+w, w)
        plt.figure()
        plt.hist(HPS_err, bins = bins, histtype="step", color = "red", label= f"HPS $\mu$={np.mean(HPS_err):.4f} IQR={iqr(HPS_err):.4f}")
        plt.hist(err, bins = bins, histtype="step", color = "blue", label= f"CNN $\mu$={np.mean(err):.4f} IQR={iqr(err):.4f}")
        plt.legend()
        plt.xlabel(r"rel $\phi$- rel $\phi_{pred}$")
        plt.xlim(-0.05, 0.05)
        # plt.ylim(0, 105000)
        plt.savefig(f"/vols/cms/lcr119/Plots/etaphi/phi.pdf", bbox_inches="tight")
        plt.show()


    def compare_phi(self):
        phi = self.df["pi0_phi"]
        phi_pred = self.df["pi0_phi_pred"]
        phi_HPS = self.df["pi0_phi_HPS"]
        phi_HPSPV = self.df["pi0_phi_HPSPV"]
        err = phi-phi_pred
        HPS_err = phi - phi_HPS
        HPSPV_err = phi- phi_HPSPV
        w = 0.0025
        bins = np.arange(np.min(err), np.max(err)+w, w)
        plt.figure()
        plt.hist(HPS_err, bins = bins, histtype="step", color = "red", label= f"HPS $\mu$={np.mean(HPS_err):.4f} IQR={iqr(HPS_err):.4f}")
        # plt.hist(np.array(HPSPV_err), bins = bins, histtype="step", color = "green", label= f"HPS PV $\mu$={np.mean(HPSPV_err):.4f} IQR={iqr(HPSPV_err):.4f}")
        plt.hist(err, bins = bins, histtype="step", color = "blue", label= f"CNN $\mu$={np.mean(err):.4f} IQR={iqr(err):.4f}")
        plt.legend()
        plt.xlabel(r"$\phi$-$\phi_{pred}$")
        # plt.xlim(-0.05, 0.05)
        # plt.ylim(0, 105000)
        # plt.savefig(f"/vols/cms/lcr119/Plots/etaphi/phi.pdf", bbox_inches="tight")

        plt.figure()
        diff = np.arange(-0.001, 0.001, 0.0001)
        plt.hist(phi_HPS-phi_HPSPV,bins=diff, histtype="step")
        plt.title("HPS-HPS PV")
        plt.show()

    def compare_profile(self, distrib, save_indv = False):

        if distrib=="phi" or distrib=="eta":
            p_range = np.concatenate((np.arange(0, 100, 10), np.arange(110, 170, 20))) #np.arange(0, 60, 5),np.arange(60, 80, 5), 
        elif distrib=="p":
            p_range = np.concatenate((np.arange(0, 60, 2.5),np.arange(60, 80, 5), np.arange(80, 100, 10), np.arange(110, 170, 20)))
        
        print(p_range)
        p_centre = p_range[:-1] + np.diff(p_range)/2
        # print(p_centre)
        width = np.diff(p_range)
        print(width)
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

            # print(f"Processing energies between {p_range[i]} and {p_range[i+1]}")
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
        fig, ax1 = plt.subplots(1, 1, figsize=(6,6))
        ax1.minorticks_on()
        ax1.grid()
        
        ax1.errorbar(p_centre, (mean_err), xerr=width/2, marker = "o", linestyle="", label = f"CNN")
        ax1.errorbar(p_centre, (mean_HPS_err), xerr=width/2, marker = "o", linestyle="", label = f"HPS")
        ax1.legend()
        ax1.set_ylabel(f"Mean {lab}")
        ax1.set_xlabel(r"$\pi^0$ Momentum")
        ax1.set_xlim(0, 150)
        # ax1.set_ylim(-10, 15)
        # plt.savefig(f"/vols/cms/lcr119/Plots/etaphi/{distrib}MEAN.pdf", bbox_inches="tight")
        plt.show()

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex="col", gridspec_kw={'height_ratios': [6, 1]}, figsize=(6,6))
        fig.subplots_adjust(hspace=0.05) 
        ax1.minorticks_on()
        ax1.grid()
        ax1.errorbar(p_centre, viqr, xerr=width/2, marker = "o", linestyle="", label = f"CNN")
        ax1.errorbar(p_centre, viqr_HPS, xerr=width/2, marker = "o", linestyle="", label = f"HPS")
        ax1.errorbar([200], [1], xerr=[1], label = "CNN/HPS Ratio", marker="o", linestyle="", color="black")
        ax1.legend()
        ax1.set_ylabel(f"IQR {lab}")
        ax2.set_xlabel(r"$\pi^0$ Momentum")
        ax2.grid()
        # ax2.set_ylim(0.4, 1)

        if distrib=="phi" or distrib=="eta":
            ax1.set_ylim(0, 0.03)
            ax2.set_ylim(0.5, 2.5)
        ax2.errorbar(p_centre, np.array(viqr)/np.array(viqr_HPS), xerr=width/2, marker="o", linestyle="", color="black")
        ax2.set_ylabel("CNN/HPS")
        
        # if distrib=="phi":
        #     ax1.set_ylim(0, 0.05)
        ax1.set_xlim(0, 150)
        # plt.savefig(f"/vols/cms/lcr119/Plots/etaphi/{distrib}IQR.pdf", bbox_inches="tight")
        plt.show()

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex="col", gridspec_kw={'height_ratios': [6, 1]}, figsize=(6,6))
        fig.subplots_adjust(hspace=0.05) 
        ax1.minorticks_on()
        ax1.grid()
        ax1.errorbar(p_centre, std, xerr=width/2, marker = "o", linestyle="", label = f"CNN")
        ax1.errorbar(p_centre, std_HPS, xerr=width/2, marker = "o", linestyle="", label = f"HPS")
        ax1.errorbar([200], [1], xerr=[1], label = "CNN/HPS Ratio", marker="o", linestyle="", color="black")
        ax1.scatter([200], [1], label = "CNN/HPS Ratio", color="black")
        ax1.legend()
        ax1.set_ylabel(f"Standard dev. {lab}")
        ax2.set_xlabel(r"$\pi^0$ Momentum")
        ax2.grid()
        # ax2.set_ylim(0.55, 1)
        ax2.errorbar(p_centre, np.array(std)/np.array(std_HPS), xerr=width/2, marker="o", linestyle="", color="black")
        ax1.set_xlim(0, 150)
        ax2.set_ylabel("CNN/HPS")
        # if distrib=="phi":
            # ax1.set_ylim(0, 0.5)
        # plt.savefig(f"/vols/cms/lcr119/Plots/Momentum/STD.pdf", bbox_inches="tight")
        plt.show()


        # fig, ax1 = plt.subplots(1, 1, figsize=(6,6))
        # fig.subplots_adjust(hspace=0.05) 
        # ax1.minorticks_on()
        # ax1.grid()
        # ax1.errorbar(p_centre, mean_pce, xerr=width/2, marker = "o", linestyle="", label = f"CNN")
        # ax1.errorbar(p_centre, mean_HPS_pce, xerr=width/2, marker = "o", linestyle="", label = f"HPS")
        # ax1.legend()
        # ax1.set_ylabel(f"Mean {lab}")
        # # ax1.set_ylim(-0.06, 0.1)
        # ax1.set_xlim(0, 150)
        # ax1.set_xlabel(r"$\pi^0$ Momentum")
        # if distrib=="phi":
        #     ax1.set_ylim(0, 0.5)
        # # plt.savefig(f"/vols/cms/lcr119/Plots/Momentum/PCE.pdf", bbox_inches="tight")
        # plt.show()


        # fig, (ax1, ax2) = plt.subplots(2, 1, sharex="col", gridspec_kw={'height_ratios': [6, 1]}, figsize=(6,6))
        # fig.subplots_adjust(hspace=0.05) 
        # ax1.minorticks_on()
        # ax1.grid()
        # ax1.errorbar(p_centre, pce_iqr, xerr=width/2, marker = "o", linestyle="", label = f"CNN")
        # ax1.errorbar(p_centre, HPS_pce_iqr, xerr=width/2, marker = "o", linestyle="", label = f"HPS")
        # ax1.errorbar([200], [1], xerr=[1], label = "CNN/HPS Ratio", marker="o", linestyle="", color="black")
        # ax1.legend()
        # if distrib=="p":
        #     ax1.set_ylabel("IQR ($p$-$p_{pred}$)/$p$")
        # ax2.set_xlabel(r"$\pi^0$ Momentum")
        # # ax1.set_ylim(0,0.3)
        # ax1.set_xlim(0, 150)
        # ax2.grid()
        # # ax2.set_ylim(0.4, 1)
        # ax2.errorbar(p_centre, np.array(pce_iqr)/np.array(HPS_pce_iqr), xerr=width/2, marker="o", linestyle="", color="black")
        # ax2.set_ylabel("CNN/HPS")
        # # plt.savefig(f"/vols/cms/lcr119/Plots/Momentum/PCEIQR.pdf", bbox_inches="tight")
        # plt.show()


        # fig, (ax1, ax2) = plt.subplots(2, 1, sharex="col", gridspec_kw={'height_ratios': [6, 1]}, figsize=(6,6))
        # fig.subplots_adjust(hspace=0.05) 
        # ax1.minorticks_on()
        # ax1.grid()
        # ax1.errorbar(p_centre, pce_std, xerr=width/2, marker = "o", linestyle="", label = f"CNN")
        # ax1.errorbar(p_centre, HPS_pce_std, xerr=width/2, marker = "o", linestyle="", label = f"HPS")
        # ax1.errorbar([200], [1], xerr=[1], label = "CNN/HPS Ratio", marker="o", linestyle="", color="black")
        # ax1.legend()
        # if distrib=="p":
        #     ax1.set_ylabel("Standard dev. ($p$-$p_{pred}$)/$p$")
        # ax2.set_xlabel(r"$\pi^0$ Momentum")
        # # ax1.set_ylim(0, 0.5)
        # ax1.set_xlim(0, 150)
        # ax2.grid()
        # # ax2.set_ylim(0.55, 1.05)
        # ax2.errorbar(p_centre, np.array(pce_std)/np.array(HPS_pce_std), xerr=width/2, marker="o", linestyle="", color="black")
        # ax2.set_ylabel("CNN/HPS")
        # # plt.savefig(f"/vols/cms/lcr119/Plots/Momentum/PCESTD.pdf", bbox_inches="tight")
        # plt.show()


        