import numpy as np
import ROOT as R
import os
import glob
from tqdm import tqdm 
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
rgb = np.array([[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,255, 255, 255, 255, 255, 254, 253, 252, 251, 249, 248, 247, 246,244, 243, 241, 240, 238, 237, 235, 233, 231, 230, 228, 226, 224,222, 220, 218, 216, 213, 211, 209, 207, 204, 202, 199, 197, 194,192, 189, 187, 184, 181, 178, 175, 172, 170, 168, 167, 166, 165, 164, 163, 162, 161, 160, 159, 158, 157, 156, 155, 154, 153, 152, 151, 150, 149, 148, 147, 146, 145, 144, 143, 142, 141, 140, 139, 138, 137, 136, 135, 134, 133, 132, 131, 130, 129, 128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 85, 84, 84, 88, 91, 94, 97, 100, 103, 107, 110, 114, 117, 121, 124, 128, 131, 135, 139, 143, 146, 150, 154, 158, 162, 166, 170, 174, 179, 183, 187, 191, 196, 200, 205, 209, 214, 218, 223, 228, 233, 237, 242, 247, 252, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],[253, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241, 240, 239, 238, 237, 236, 235, 234, 233, 232, 231, 230, 229, 228, 227, 226, 225, 224, 223, 222, 221, 220, 219, 218, 217, 216, 215, 214, 213, 212, 211, 210, 209, 208, 207, 206, 205, 204, 203, 202, 201, 200, 199, 198, 197, 196, 195, 194, 193, 192, 191, 190, 189, 188, 187, 186, 185, 184, 183, 182, 181, 180, 179, 178, 177, 176, 175, 174, 173, 172, 171, 170, 169, 169, 169, 170, 171, 172, 173, 174, 176, 177, 178, 180, 181, 183, 184, 186, 188, 189, 191, 193, 195, 197, 199, 201, 203, 205, 207, 209, 211, 214, 216, 218, 221, 223, 226, 228, 231, 233, 236, 239, 241, 244, 247, 250, 253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 252, 247, 242, 237, 232, 226, 221, 216, 211, 205, 200, 194, 189, 183, 178, 172, 166, 160, 155, 149, 143, 137, 131, 125, 119, 113, 107, 100, 94, 88, 81, 75, 69, 62, 56, 49, 42, 36, 29, 29],[253, 253, 252, 251, 250, 249, 248, 248, 247, 247, 246, 246, 245, 245, 244, 244, 244, 244, 243, 243, 243, 243, 243, 243, 243, 243, 244, 244, 244, 244, 245, 245, 246, 246, 247, 247, 248, 249, 250, 250, 251, 252, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 253, 250, 247, 244, 241, 237, 234, 231, 227, 224, 220, 217, 213, 210, 206, 203, 199, 195, 191, 187, 183, 179, 175, 171, 167, 163, 159, 155, 150, 146, 142, 137, 133, 128, 124, 119, 114, 110, 105, 100, 95, 90, 86, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 13, 9, 8, 7, 6, 5, 4, 3, 2, 2]], dtype=float)/256
rgb = np.transpose(rgb)
newcmp = colors.ListedColormap(rgb)
import argparse

parser = argparse.ArgumentParser(description='Generate images from RHTree ROOT files')
parser.add_argument('--n_tau', required=True, type=int, help="Number of taus to select")
parser.add_argument('--sample', required=True, type=str, help="Sample Name")
parser.add_argument('--split', required=True, type=str, help="Sample section")
parser.add_argument('--save_path', required=False, default="/vols/cms/lcr119/Images/", type=str, help="Save path")
parser.add_argument('--max_events', required=False, default=10000000, type=int, help="Max events to process")
args = parser.parse_args()




sample = args.sample #"GluGluHToTauTau_M125", "DYJetsToLL-LO"
if sample == "GluGluHToTauTau_M125":
    alias = "ggHTT_" + args.split
else:
    alias = "unkwn"
path_to_filelist = "/home/hep/lcr119/DeepPi/Analysis/scripts/HPS_MC_106X_" + sample + ".dat"


# Max number of events to process #int(rhTree.GetEntries())
n_tau_target = args.n_tau # Max number of taus to select
plot = False


shard = 0 # number of file if split into several required
save_folder = args.save_path + "Test"
savepath = save_folder + "/" + alias + "_" + str(shard) + ".pkl"
# check if directory exists
if not os.path.exists(save_folder):
    os.makedirs(save_folder)


count = 0
with open(path_to_filelist) as f:
    lines = f.readlines()
    lines = [line.rstrip('\n') for line in lines]
    if args.split == "A":
        files = lines[:125]
    elif args.split == "B":
        files = lines[125:250]
    elif args.split == "C":
        files = lines[250:375]
    elif args.split == "D":
        files = lines[375:500]
    elif args.split == "E":
        files = lines[500:]
    else: 
        raise Exception("Split not recognised")
    

def crop_channel(image, centre_eta, centre_phi):
    pad = 16 # padding on each side of centre
    centre_eta += 85 # add 85 to crop array by index

    # Add extra blank cells to eta for edge cases:
    if centre_eta - pad < 0: # Add below
        # print("Eta out of bounds, extra padding")
        add_cells = np.abs(centre_eta-pad)
        padding = np.zeros((add_cells, 360)) # Extra padding
        image = np.concatenate((padding, image), axis = 0)
        centre_eta += add_cells # Adjust eta index for cropping (the number added)
    elif centre_eta + pad >= 170: # Add above
        # print("Eta out of bounds, extra padding")
        add_cells = centre_eta+pad-169 # (169 as index 169 is 170th entry)
        padding = np.zeros((add_cells, 360)) # Extra padding
        image = np.concatenate((image, padding), axis = 0)

    # Wrap phi for overlapping cases:
    if centre_phi < pad: # Wrap-around on left side
        # print("Wrapping left")
        diff = pad-centre_phi
        image_crop = np.concatenate((image[centre_eta-pad:centre_eta+pad+1,-diff:],
                                    image[centre_eta-pad:centre_eta+pad+1,:centre_phi+pad+1]), axis=-1)
    elif 360-centre_phi <= pad: # Wrap-around on right side
        # print("Wrapping right")
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
# HPS variables:
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


n_selected = 0 # Number of taus selected
if args.n_tau != -1:
    pbar = tqdm(total = n_tau_target)
complete = False # flag to say when all taus selected

os.system('~/scripts/t-notify.sh Beginning image creation')

file_pbar = tqdm(total = len(files))

for f in files:
    file = "root://gfe02.grid.hep.ph.ic.ac.uk:1097/store/user/lrussell/DetectorImages_HPS_MC_106X_2018/" + f
    rhTree = R.TChain("recHitAnalyzer/RHTree")
    rhTree.Add(file)
    nEvts = int(rhTree.GetEntries())
    if complete:
        break
    for event in range(nEvts):
        rhTree.GetEntry(event)
        # Load truth values
        truthDM = np.array(rhTree.jet_truthDM)
        # Load the detector images
        ECAL_barrel = np.reshape(np.array(rhTree.EB_energy), (170, 360))
        Tracks_barrel = np.reshape(np.array(rhTree.TracksE_EB), (170, 360))
        PF_HCAL_barrel = np.reshape(np.array(rhTree.PF_HCAL_EB), (170, 360))
        PF_ECAL_barrel = np.reshape(np.array(rhTree.PF_ECAL_EB), (170, 360))
        addTracks_barrel = np.reshape(np.array(rhTree.FailedTracksE_EB), (170, 360))
        # Load jet centre coordinates
        ieta = np.array(rhTree.jet_centre2_ieta)
        iphi = np.array(rhTree.jet_centre2_iphi)
        # Load neutral kinematics
        releta = np.array(rhTree.jet_neutral_indv_releta, dtype=object)
        relphi = np.array(rhTree.jet_neutral_indv_relphi, dtype=object)
        relp = np.array(rhTree.jet_neutral_indv_relp, dtype=object)
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
                image = crop_image(Tracks_barrel, ECAL_barrel, PF_HCAL_barrel, PF_ECAL_barrel, addTracks_barrel, centre_eta, centre_phi)
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
                # add HPS variables:
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
                n_selected += 1
                if args.n_tau != -1:
                    if n_selected%10 == 0:
                        pbar.update(10)
                    # print(n_selected)
                    if n_selected>=n_tau_target:
                        complete = True
                        break
                if (n_selected%10000)==0: # save in multiple files for safety
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
                    # HPS variables:
                    df["tau_dm"] = list_tau_dm
                    df["tau_pt"] = list_tau_pt
                    df["tau_E"] = list_tau_E
                    df["tau_eta"] = list_tau_eta
                    df["tau_mass"] = list_tau_mass
                    df["pi_px"] = list_pi_px
                    df["pi_py"] = list_pi_py
                    df["pi_pz"] = list_pi_pz
                    df["pi_E"] = list_pi_E
                    df["pi0_px"] = list_pi0_px
                    df["pi0_py"] = list_pi0_py
                    df["pi0_pz"] = list_pi0_pz
                    df["pi0_E"] = list_pi0_E
                    df["pi0_dEta"] = list_pi0_dEta
                    df["pi0_dPhi"] = list_pi0_dPhi
                    df["strip_mas"] = list_strip_mass
                    df["strip_pt"] = list_strip_pt
                    df["rho_mass"] = list_rho_mass
                    df["pi2_px"] = list_pi2_px
                    df["pi2_py"] = list_pi2_py
                    df["pi2_pz"] = list_pi2_pz
                    df["pi2_E"] = list_pi2_E
                    df["pi3_px"] = list_pi3_px
                    df["pi3_py"] = list_pi3_py
                    df["pi3_pz"] = list_pi3_pz
                    df["pi3_E"] = list_pi3_E
                    df["mass0"] = list_mass0
                    df["mass1"] = list_mass1
                    df["mass2"] = list_mass2
                    print("Saving dataframe at: ", savepath)
                    df.to_pickle(savepath)
                    shard+=1 # new shard
                    savepath = save_folder + "/" + alias + "_" + str(shard) + ".pkl"
                    # os.system('~/scripts/t-notify.sh shard saved')
                    print("After event: ", event)
                    Tracks_list = []
                    ECAL_list = []
                    PF_HCAL_list = []
                    PF_ECAL_list = []
                    addTracks_list = []
                    DM_list = []
                    releta_list = []
                    relphi_list = []
                    relp_list = []
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
    file_pbar.update(1)
    print("Changing file")
    del rhTree
               

savepath = save_folder + "/" + alias + "_" + str(shard) + "_end.pkl"
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
df["tau_dm"] = list_tau_dm
df["tau_pt"] = list_tau_pt
df["tau_E"] = list_tau_E
df["tau_eta"] = list_tau_eta
df["tau_mass"] = list_tau_mass
df["pi_px"] = list_pi_px
df["pi_py"] = list_pi_py
df["pi_pz"] = list_pi_pz
df["pi_E"] = list_pi_E
df["pi0_px"] = list_pi0_px
df["pi0_py"] = list_pi0_py
df["pi0_pz"] = list_pi0_pz
df["pi0_E"] = list_pi0_E
df["pi0_dEta"] = list_pi0_dEta
df["pi0_dPhi"] = list_pi0_dPhi
df["strip_mas"] = list_strip_mass
df["strip_pt"] = list_strip_pt
df["rho_mass"] = list_rho_mass
df["pi2_px"] = list_pi2_px
df["pi2_py"] = list_pi2_py
df["pi2_pz"] = list_pi2_pz
df["pi2_E"] = list_pi2_E
df["pi3_px"] = list_pi3_px
df["pi3_py"] = list_pi3_py
df["pi3_pz"] = list_pi3_pz
df["pi3_E"] = list_pi3_E
df["mass0"] = list_mass0
df["mass1"] = list_mass1
df["mass2"] = list_mass2
print("Saving dataframe at ", savepath)
df.to_pickle(savepath)

os.system('~/scripts/t-notify.sh Finished')


