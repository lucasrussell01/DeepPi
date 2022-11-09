import numpy as np
import ROOT as R
import os
import glob
import pandas as pd
from tqdm import tqdm 
import argparse
import gc 

# This script will generate images for all taus in a given ROOT file

parser = argparse.ArgumentParser(description='Generate images from RHTree ROOT file')
parser.add_argument('--file', required=True, type=str, help="Path to file in dcache")
parser.add_argument('--alias', required=True, type=str, help="Sample alias")
parser.add_argument('--save_path', required=False, default="/vols/cms/lcr119/Images/TestBatch", type=str, help="Save path")

args = parser.parse_args()

# check if save directory exists
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

def crop_channel(image, centre_eta, centre_phi):
    pad = 16 # padding on each side of centre
    centre_eta += 85 # add 85 to crop array by index
    # Add extra blank cells to eta for edge cases:
    if centre_eta - pad < 0: # Add below
        add_cells = np.abs(centre_eta-pad)
        padding = np.zeros((add_cells, 360)) # Extra padding
        image = np.concatenate((padding, image), axis = 0)
        centre_eta += add_cells # Adjust eta index for cropping (the number added)
    elif centre_eta + pad >= 170: # Add above
        add_cells = centre_eta+pad-169 # (169 as index 169 is 170th entry)
        padding = np.zeros((add_cells, 360)) # Extra padding
        image = np.concatenate((image, padding), axis = 0)
    # Wrap phi for overlapping cases:
    if centre_phi < pad: # Wrap-around on left side
        diff = pad-centre_phi
        image_crop = np.concatenate((image[centre_eta-pad:centre_eta+pad+1,-diff:],
                                    image[centre_eta-pad:centre_eta+pad+1,:centre_phi+pad+1]), axis=-1)
    elif 360-centre_phi <= pad: # Wrap-around on right side
        diff = pad - (360-centre_phi)
        image_crop = np.concatenate((image[centre_eta-pad:centre_eta+pad+1,centre_phi-pad:],
                                    image[centre_eta-pad:centre_eta+pad+1,:diff+1]), axis=-1)
    else:
        image_crop = image[centre_eta-pad:centre_eta+pad+1,centre_phi-pad:centre_phi+pad+1]
    return image_crop

def crop_image(Tracks, ECAL, PF_HCAL, PF_ECAL, addTracks, centre_eta, centre_phi):
    Tracks_crop = crop_channel(Tracks, centre_eta, centre_phi)
    ECAL_crop = crop_channel(ECAL, centre_eta, centre_phi)
    PF_HCAL_crop = crop_channel(PF_HCAL, centre_eta, centre_phi)
    PF_ECAL_crop = crop_channel(PF_ECAL, centre_eta, centre_phi)
    addTracks_crop = crop_channel(addTracks, centre_eta, centre_phi)
    return Tracks_crop, ECAL_crop, PF_HCAL_crop, PF_ECAL_crop, addTracks_crop


targetDM = [0, 1, 2, 10, 11]

Tracks_list = []
ECAL_list = []
PF_HCAL_list = []
PF_ECAL_list = []
addTracks_list = []
DM_list = []
releta_list = []
relphi_list = []
relp_list = []
prong_releta_list = []
prong_relphi_list = []
prong_relp_list = []
# MVA/DeepTau information
MVA_DM_list = []
deeptauVSjet_list = []
deeptauVSmu_list = []
deeptauVSe_list = []

# General info variables:
jet_eta_list = []
jet_phi_list = []
jet_pt_list = []
jet_mass_list = []
pi0_centre_eta_list = []
pi0_centre_phi_list = []
tau_centre_eta_list = []
tau_centre_phi_list = []
centre2_eta_list = []
centre2_phi_list = []
# PV information:
PV_list = []
# High level variables:
list_pi0_releta = []
list_pi0_relphi = []
list_tau_dm = []
list_tau_pt = []
list_tau_E = []
list_tau_eta = []
list_tau_mass = []
list_pi_px = []
list_pi_py = []
list_pi_pz = []
list_pi_E = []
list_pi0_px = []
list_pi0_py = []
list_pi0_pz = []
list_pi0_E = []
list_pi0_dEta = []
list_pi0_dPhi = []
list_strip_mass = []
list_strip_pt = []
list_rho_mass = []
list_pi2_px = []
list_pi2_py = []
list_pi2_pz = []
list_pi2_E = []
list_pi3_px = []
list_pi3_py = []
list_pi3_pz = []
list_pi3_E = []
list_mass0 = []
list_mass1 = []
list_mass2 = []


print("Attempting to open file")

# Given file must have path to crab job in dcache + file path from crab
file = "root://gfe02.grid.hep.ph.ic.ac.uk:1097" + args.file 

Rfile = R.TFile.Open(file, "READ")
rhTree = Rfile.Get("recHitAnalyzer/RHTree")

nEvts = int(rhTree.GetEntries())


print("Beginning image creation")

# ADD: jet eta, phi, pT, mass, tau centre, pi0_centre

for event in range(nEvts):
    rhTree.GetEntry(event)
    # Load truth values
    truthDM = np.array(rhTree.jet_truthDM)
    # Load the detector images
    # ECAL_barrel = np.reshape(np.array(rhTree.EB_energy), (170, 360))
    # Tracks_barrel = np.reshape(np.array(rhTree.TracksE_EB), (170, 360))
    # PF_HCAL_barrel = np.reshape(np.array(rhTree.PF_HCAL_EB), (170, 360))
    # PF_ECAL_barrel = np.reshape(np.array(rhTree.PF_ECAL_EB), (170, 360))
    # addTracks_barrel = np.reshape(np.array(rhTree.FailedTracksE_EB), (170, 360))
    # Load jet centre coordinates
    # ieta = np.array(rhTree.jet_centre2_ieta) # old egamma convention
    # iphi = np.array(rhTree.jet_centre2_iphi)
    ieta = np.array(rhTree.pi0_centre_ieta) # Tau/HPS pi0
    iphi = np.array(rhTree.pi0_centre_iphi)
    # Load neutral kinematics
    releta = np.array(rhTree.jet_neutral_indv_releta, dtype=object)
    relphi = np.array(rhTree.jet_neutral_indv_relphi, dtype=object)
    relp = np.array(rhTree.jet_neutral_indv_relp, dtype=object)
    # Load prong kinematics
    prong_releta = np.array(rhTree.jet_charged_indv_releta, dtype=object)
    prong_relphi = np.array(rhTree.jet_charged_indv_relphi, dtype=object)
    prong_relp = np.array(rhTree.jet_charged_indv_relp, dtype=object)
    # Primary vertex:
    PV = np.array(rhTree.PrimaryVertex)
    # Iterate over the taus in the event
    n_taus = len(truthDM)
    for i in range(n_taus):
        if truthDM[i] in targetDM: # check if genuine tau    
            centre_phi = int(iphi[i])
            centre_eta = int(ieta[i]) 
            if centre_phi>=360 or centre_phi<0:
                continue # don't use these events
            elif centre_eta>=80 or centre_eta<-80:
                continue # too close to edge

            image = crop_image(np.reshape(np.array(rhTree.TracksE_EB), (170, 360)), np.reshape(np.array(rhTree.EB_energy), (170, 360)),
                                np.reshape(np.array(rhTree.PF_HCAL_EB), (170, 360)), np.reshape(np.array(rhTree.PF_ECAL_EB), (170, 360)),
                                np.reshape(np.array(rhTree.FailedTracksE_EB), (170, 360)), centre_eta, centre_phi)

            # image = crop_image(Tracks_barrel, ECAL_barrel, PF_HCAL_barrel, PF_ECAL_barrel, addTracks_barrel, centre_eta, centre_phi)
            if (image[1].shape != (33, 33)):
                print(image[1].shape)
                print(event)
                print(centre_eta, centre_phi)
                raise Exception ("incorrect image shape")
            Tracks_list.append(image[0])
            ECAL_list.append(image[1])
            PF_HCAL_list.append(image[2])
            PF_ECAL_list.append(image[3])
            addTracks_list.append(image[4])
            DM_list.append(truthDM[i])
            releta_list.append(np.array(releta[i]))
            relphi_list.append(np.array(relphi[i]))
            relp_list.append(np.array(relp[i]))
            prong_releta_list.append(np.array(prong_releta[i]))
            prong_relphi_list.append(np.array(prong_relphi[i]))
            prong_relp_list.append(np.array(prong_relp[i]))
            PV_list.append(PV)
            # General info variables:
            jet_eta_list.append(np.array(rhTree.jetEta)[i])
            jet_phi_list.append(np.array(rhTree.jetPhi)[i])
            jet_pt_list.append(np.array(rhTree.jetPt)[i])
            jet_mass_list.append(np.array(rhTree.jetM)[i])
            pi0_centre_eta_list.append(np.array(rhTree.pi0_centre_eta)[i])
            pi0_centre_phi_list.append(np.array(rhTree.pi0_centre_phi)[i])
            tau_centre_eta_list.append(np.array(rhTree.tau_centre_eta)[i])
            tau_centre_phi_list.append(np.array(rhTree.tau_centre_phi)[i])
            centre2_eta_list.append(np.array(rhTree.jet_centre2_eta)[i])
            centre2_phi_list.append(np.array(rhTree.jet_centre2_phi)[i])
            MVA_DM_list.append(np.array(rhTree.tau_mva_dm)[i])
            deeptauVSjet_list.append(np.array(rhTree.tau_deeptau_id)[i])
            deeptauVSmu_list.append(np.array(rhTree.tau_deeptau_id_vs_mu)[i])
            deeptauVSe_list.append(np.array(rhTree.tau_deeptau_id_vs_e)[i])
            # add HPS variables:
            list_pi0_releta.append(np.array(rhTree.HPSpi0_releta)[i]) # + np.array(rhTree.jet_centre2_eta)[i] - np.array(rhTree.pi0_centre_eta)[i]) # adjust relative to new centering
            list_pi0_relphi.append(np.array(rhTree.HPSpi0_relphi)[i]) # + np.array(rhTree.jet_centre2_phi)[i] - np.array(rhTree.pi0_centre_phi)[i])
            list_tau_dm.append(np.array(rhTree.tau_dm)[i])
            list_tau_pt.append(np.array(rhTree.tau_pt)[i])
            list_tau_E.append(np.array(rhTree.tau_E)[i])
            list_tau_eta.append(np.array(rhTree.tau_eta)[i])
            list_tau_mass.append(np.array(rhTree.tau_mass)[i])
            list_pi_px.append(np.array(rhTree.pi_px)[i])
            list_pi_py.append(np.array(rhTree.pi_py)[i])
            list_pi_pz.append(np.array(rhTree.pi_pz)[i])
            list_pi_E.append(np.array(rhTree.pi_E)[i])
            list_pi0_px.append(np.array(rhTree.pi0_px)[i])
            list_pi0_py.append(np.array(rhTree.pi0_py)[i])
            list_pi0_pz.append(np.array(rhTree.pi0_pz)[i])
            list_pi0_E.append(np.array(rhTree.pi0_E)[i])
            list_pi0_dEta.append(np.array(rhTree.pi0_dEta)[i])
            list_pi0_dPhi.append(np.array(rhTree.pi0_dPhi)[i])
            list_strip_mass.append(np.array(rhTree.strip_mass)[i])
            list_strip_pt.append(np.array(rhTree.strip_pt)[i])
            list_rho_mass.append(np.array(rhTree.rho_mass)[i])
            list_pi2_px.append(np.array(rhTree.pi2_px)[i])
            list_pi2_py.append(np.array(rhTree.pi2_py)[i])
            list_pi2_pz.append(np.array(rhTree.pi2_pz)[i])
            list_pi2_E.append(np.array(rhTree.pi2_E)[i])
            list_pi3_px.append(np.array(rhTree.pi3_px)[i])
            list_pi3_py.append(np.array(rhTree.pi3_py)[i])
            list_pi3_pz.append(np.array(rhTree.pi3_pz)[i])
            list_pi3_E.append(np.array(rhTree.pi3_E)[i])
            list_mass0.append(np.array(rhTree.mass0)[i])
            list_mass1.append(np.array(rhTree.mass1)[i])
            list_mass2.append(np.array(rhTree.mass2)[i])

df = pd.DataFrame()
df["Tracks"] = Tracks_list
df["ECAL"] = ECAL_list
df["PF_HCAL"] = PF_HCAL_list
df["PF_ECAL"] = PF_ECAL_list
df["addTracks"] = addTracks_list
df["DM"] = DM_list
df["releta"] = releta_list
df["relphi"] = relphi_list
df["relp"] = relp_list
df["prong_releta"] = prong_releta_list
df["prong_relphi"] = prong_relphi_list
df["prong_relp"] = prong_relp_list
df["PV"] = PV_list
# General information
df["jet_eta"] = jet_eta_list
df["jet_phi"] = jet_phi_list
df["jet_pt"] = jet_pt_list
df["jet_mass"] = jet_mass_list
df["pi0_centre_eta"] = pi0_centre_eta_list
df["pi0_centre_phi"] = pi0_centre_phi_list
df["tau_centre_eta"] = tau_centre_eta_list
df["tau_centre_phi"] = tau_centre_phi_list
df["centre2_eta"] = centre2_eta_list
df["centre2_phi"] = centre2_phi_list
# Tau ID info
df["MVA_DM"] = MVA_DM_list
df["deeptauVSjet"] = deeptauVSjet_list
df["deeptauVSmu"] = deeptauVSmu_list
df["deeptauVSe"] = deeptauVSjet_list
# HPS variables:
df["HPS_pi0_releta"] = list_pi0_releta
df["HPS_pi0_relphi"] = list_pi0_relphi
df["HPS_tau_dm"] = list_tau_dm
df["HPS_tau_pt"] = list_tau_pt
df["HPS_tau_E"] = list_tau_E
df["HPS_tau_eta"] = list_tau_eta
df["HPS_tau_mass"] = list_tau_mass
df["HPS_pi_px"] = list_pi_px
df["HPS_pi_py"] = list_pi_py
df["HPS_pi_pz"] = list_pi_pz
df["HPS_pi_E"] = list_pi_E
df["HPS_pi0_px"] = list_pi0_px
df["HPS_pi0_py"] = list_pi0_py
df["HPS_pi0_pz"] = list_pi0_pz
df["HPS_pi0_E"] = list_pi0_E
df["HPS_pi0_dEta"] = list_pi0_dEta
df["HPS_pi0_dPhi"] = list_pi0_dPhi
df["HPS_strip_mass"] = list_strip_mass
df["HPS_strip_pt"] = list_strip_pt
df["HPS_rho_mass"] = list_rho_mass
df["HPS_pi2_px"] = list_pi2_px
df["HPS_pi2_py"] = list_pi2_py
df["HPS_pi2_pz"] = list_pi2_pz
df["HPS_pi2_E"] = list_pi2_E
df["HPS_pi3_px"] = list_pi3_px
df["HPS_pi3_py"] = list_pi3_py
df["HPS_pi3_pz"] = list_pi3_pz
df["HPS_pi3_E"] = list_pi3_E
df["HPS_mass0"] = list_mass0
df["HPS_mass1"] = list_mass1
df["HPS_mass2"] = list_mass2

savepath = args.save_path + "/" + args.alias + ".pkl"
print("Saving dataframe at: ", savepath)
df.to_pickle(savepath)



Rfile.Close()
gc.collect() # collect released memory
        