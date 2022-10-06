import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors, ticker, cm
import ROOT as R
rgb = np.array([[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,255, 255, 255, 255, 255, 254, 253, 252, 251, 249, 248, 247, 246,244, 243, 241, 240, 238, 237, 235, 233, 231, 230, 228, 226, 224,222, 220, 218, 216, 213, 211, 209, 207, 204, 202, 199, 197, 194,192, 189, 187, 184, 181, 178, 175, 172, 170, 168, 167, 166, 165, 164, 163, 162, 161, 160, 159, 158, 157, 156, 155, 154, 153, 152, 151, 150, 149, 148, 147, 146, 145, 144, 143, 142, 141, 140, 139, 138, 137, 136, 135, 134, 133, 132, 131, 130, 129, 128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 85, 84, 84, 88, 91, 94, 97, 100, 103, 107, 110, 114, 117, 121, 124, 128, 131, 135, 139, 143, 146, 150, 154, 158, 162, 166, 170, 174, 179, 183, 187, 191, 196, 200, 205, 209, 214, 218, 223, 228, 233, 237, 242, 247, 252, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],[253, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241, 240, 239, 238, 237, 236, 235, 234, 233, 232, 231, 230, 229, 228, 227, 226, 225, 224, 223, 222, 221, 220, 219, 218, 217, 216, 215, 214, 213, 212, 211, 210, 209, 208, 207, 206, 205, 204, 203, 202, 201, 200, 199, 198, 197, 196, 195, 194, 193, 192, 191, 190, 189, 188, 187, 186, 185, 184, 183, 182, 181, 180, 179, 178, 177, 176, 175, 174, 173, 172, 171, 170, 169, 169, 169, 170, 171, 172, 173, 174, 176, 177, 178, 180, 181, 183, 184, 186, 188, 189, 191, 193, 195, 197, 199, 201, 203, 205, 207, 209, 211, 214, 216, 218, 221, 223, 226, 228, 231, 233, 236, 239, 241, 244, 247, 250, 253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 252, 247, 242, 237, 232, 226, 221, 216, 211, 205, 200, 194, 189, 183, 178, 172, 166, 160, 155, 149, 143, 137, 131, 125, 119, 113, 107, 100, 94, 88, 81, 75, 69, 62, 56, 49, 42, 36, 29, 29],[253, 253, 252, 251, 250, 249, 248, 248, 247, 247, 246, 246, 245, 245, 244, 244, 244, 244, 243, 243, 243, 243, 243, 243, 243, 243, 244, 244, 244, 244, 245, 245, 246, 246, 247, 247, 248, 249, 250, 250, 251, 252, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 253, 250, 247, 244, 241, 237, 234, 231, 227, 224, 220, 217, 213, 210, 206, 203, 199, 195, 191, 187, 183, 179, 175, 171, 167, 163, 159, 155, 150, 146, 142, 137, 133, 128, 124, 119, 114, 110, 105, 100, 95, 90, 86, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 13, 9, 8, 7, 6, 5, 4, 3, 2, 2]], dtype=float)/256
rgb = np.transpose(rgb)
newcmp = colors.ListedColormap(rgb)
mpl.rcParams.update({'font.size': 12})


class image_plotter:

    def __init__(self, file):
        self.rhTree = R.TChain("recHitAnalyzer/RHTree")
        self.rhTree.Add(file)
        self.nEvts = self.rhTree.GetEntries()
        self.branches = self.rhTree.GetListOfBranches()

    def get_barrel(self, entry):
        self.rhTree.GetEntry(entry)
        # Barrel dimensions should be 360*170, up to eta = 1.48
        ECAL_barrel = np.reshape(np.array(self.rhTree.EB_energy), (170, 360))
        Tracks_barrel = np.reshape(np.array(self.rhTree.TracksE_EB), (170, 360))
        PF_HCAL_barrel = np.reshape(np.array(self.rhTree.PF_HCAL_EB), (170, 360))
        PF_ECAL_barrel = np.reshape(np.array(self.rhTree.PF_ECAL_EB), (170, 360))
        FailedTracks_barrel = np.reshape(np.array(self.rhTree.FailedTracksE_EB), (170, 360))
        return Tracks_barrel, ECAL_barrel, PF_ECAL_barrel, PF_HCAL_barrel, FailedTracks_barrel

    def centre(self, entry):
        self.rhTree.GetEntry(entry)
        # centre on egamma
        # ieta = np.array(self.rhTree.jet_centre2_ieta)
        # iphi = np.array(self.rhTree.jet_centre2_iphi)
        # on HPS tau if exists
        ieta = np.array(self.rhTree.pi0_centre_ieta)
        iphi = np.array(self.rhTree.pi0_centre_iphi)
        return iphi, ieta

    def plot_full(self, Tracks, ECAL, PFECAL, PFHCAL, event):
    
        self.rhTree.GetEntry(event)
        print("Event ID: ", self.rhTree.eventId)
        centres = self.centre(event)
        truthDM = np.array(self.rhTree.jet_truthDM)
        neutral_ECAL = np.array(self.rhTree.neutralsum_ECAL)
        jet_pt = np.array(self.rhTree.jetPt)
        truthLabel = np.array(self.rhTree.jet_truthLabel) 
        
        x = np.arange(361)
        y = np.arange(-85, 86)
        x_, y_ = np.meshgrid(x, y)
        
        # Load neutral gen positions
        gen_neutral_ieta = np.array(self.rhTree.jet_neutral_indv_releta_crystal, dtype=object)
        gen_neutral_iphi = np.array(self.rhTree.jet_neutral_indv_relphi_crystal, dtype=object)
        # Load charged gen positions
        gen_charged_ieta = np.array(self.rhTree.jet_charged_indv_releta_crystal, dtype=object)
        gen_charged_iphi = np.array(self.rhTree.jet_charged_indv_relphi_crystal, dtype=object)
        
        fig, ax = plt.subplots(2, 3, figsize = (20,15), gridspec_kw={'width_ratios': [1, 1.2, 0.3]}) 
        
        # plot the images
        cbar = ax[0][1].pcolormesh(x_, y_, ECAL, cmap=newcmp, norm = colors.LogNorm(vmin=1e-4, vmax=np.max(ECAL)))
        ax[0][0].pcolormesh(x_, y_, Tracks, cmap=newcmp, norm = colors.LogNorm(vmin=1e-4, vmax=np.max(ECAL)))
        ax[1][0].pcolormesh(x_, y_, PFHCAL, cmap=newcmp, norm = colors.LogNorm(vmin=1e-4, vmax=np.max(ECAL)))
        ax[1][1].pcolormesh(x_, y_, PFECAL, cmap=newcmp, norm = colors.LogNorm(vmin=1e-4, vmax=np.max(ECAL)))
        
        ax[0][0].scatter(centres[0], centres[1], color = "black", marker = ".", label = "Centre")
        ax[0][1].scatter(centres[0], centres[1], color = "black", marker = ".", label = "Centre")
        
        index = 0
        for i in truthDM:
            if i == 1 or i==2 or i == 11:
                # plot neutral gen for DM 1, 2, 11 taus
                ax[0][0].scatter(gen_neutral_iphi[index], gen_neutral_ieta[index], marker = "*", label = "Gen Neutral Pion", color= "tab:brown")
                ax[0][1].scatter(gen_neutral_iphi[index], gen_neutral_ieta[index], marker = "*", label = "Gen Neutral Pion", color= "tab:brown")
            if i != -1:
                # plot charged gen for true taus
                ax[0][0].scatter(gen_charged_iphi[index], gen_charged_ieta[index], marker = "^", label = "Gen Charged Pion", color= "blue")
                ax[0][1].scatter(gen_charged_iphi[index], gen_charged_ieta[index], marker = "^", label = "Gen Charged Pion", color= "blue")
            index+=1
            
        ax[0][0].set_xlim(0, 360)
        ax[0][1].set_xlim(0, 360)
        ax[1][0].set_xlim(0, 360)
        ax[1][1].set_xlim(0, 360)
        
        ax[0][0].text(0.07, 0.88, "Tracks @ECAL", fontsize=14, transform=ax[0][0].transAxes, fontweight="bold")
        ax[0][1].text(0.07, 0.88, "ECAL RecHits", fontsize=14, transform=ax[0][1].transAxes, fontweight="bold")
        ax[1][0].text(0.07, 0.88, "PF_HCAL Energy", fontsize=14, transform=ax[1][0].transAxes, fontweight="bold")
        ax[1][1].text(0.07, 0.88, "PF_ECAL Energy", fontsize=14, transform=ax[1][1].transAxes, fontweight="bold")
        
        ax[0][0].set_xlabel("i$\phi$ ECAL Barrel")
        ax[0][0].set_ylabel("i$\eta$ ECAL Barrel")
        ax[0][1].set_xlabel("i$\phi$ ECAL Barrel")
        ax[0][1].set_ylabel("i$\eta$ ECAL Barrel")
        ax[1][0].set_xlabel("i$\phi$ ECAL Barrel")
        ax[1][0].set_ylabel("i$\eta$ ECAL Barrel")
        ax[1][1].set_xlabel("i$\phi$ ECAL Barrel")
        ax[1][1].set_ylabel("i$\eta$ ECAL Barrel")
        
        ax[0][0].legend(loc = "lower right")
        
        plt.colorbar(cbar, ax=ax[0][1])
        plt.colorbar(cbar, ax=ax[1][1])
        
        ax[0][2].axis("off")
        ax[1][2].axis("off")
        
        ax[0][2].text(-0.2, 0.95, "Candidates in Event:", fontsize=16, fontweight="bold", transform=ax[0][2].transAxes)
        h =0.95
        for i in range(len(truthDM)):
            sout = "Jet $p_T$: " + str(round(jet_pt[i], 2)) + " Truth: " + str(truthLabel[i]) + " DM: " + str(truthDM[i])
            h += - 0.08
            ax[0][2].text(-0.2, h, sout, fontsize=16, transform=ax[0][2].transAxes)
            sout2 = "Total neutral ECAL deposit: " + str(round(neutral_ECAL[i], 2))
            h += - 0.08
            ax[0][2].text(-0.2, h, sout2, fontsize=16, transform=ax[0][2].transAxes)
            
        plt.show()

    def crop_image(self, Tracks, ECAL, PFECAL, PFHCAL, FailedTracks, centre_eta, centre_phi):

        pad = 16 # padding on each side of centre
        centre_eta += 85 # add 85 to crop array by index
        cell_shift = 0

        # Add extra blank cells to eta for edge cases:
        if centre_eta - pad < 0: # Add below
            print("Eta out of bounds, extra padding (bottom)")
            add_cells = np.abs(centre_eta-pad)
            cell_shift = add_cells
            padding = np.ones((add_cells, 360)) # Extra padding
            ECAL = np.concatenate((padding, ECAL), axis = 0)
            Tracks = np.concatenate((padding, Tracks), axis = 0)
            FailedTracks = np.concatenate((padding, FailedTracks), axis = 0)
            PFHCAL = np.concatenate((padding, PFHCAL), axis = 0)
            PFECAL = np.concatenate((padding, PFECAL), axis = 0)
            centre_eta += add_cells # Adjust eta index for cropping (the number added)
        elif centre_eta + pad >= 170: # Add above
            print("Eta out of bounds, extra padding (upper)")
            add_cells = centre_eta+pad-169 # (169 as index 169 is 170th entry)
            padding = np.ones((add_cells, 360)) # Extra padding
            ECAL = np.concatenate((ECAL, padding), axis = 0)
            Tracks = np.concatenate((Tracks, padding), axis = 0)
            FailedTracks = np.concatenate((FailedTracks, padding), axis = 0)
            PFHCAL = np.concatenate((PFHCAL, padding), axis = 0)  
            PFECAL = np.concatenate((PFECAL, padding), axis = 0)      

        # Wrap phi for overlapping cases:
        if centre_phi < pad: # Wrap-around on left side
            print("Wrapping left, iphi: ", centre_phi)
            diff = pad-centre_phi
            ECAL_crop = np.concatenate((ECAL[centre_eta-pad:centre_eta+pad+1,-diff:],
                                        ECAL[centre_eta-pad:centre_eta+pad+1,:centre_phi+pad+1]), axis=-1)
            Tracks_crop = np.concatenate((Tracks[centre_eta-pad:centre_eta+pad+1,-diff:],
                                        Tracks[centre_eta-pad:centre_eta+pad+1,:centre_phi+pad+1]), axis=-1)
            FailedTracks_crop = np.concatenate((FailedTracks[centre_eta-pad:centre_eta+pad+1,-diff:],
                                        FailedTracks[centre_eta-pad:centre_eta+pad+1,:centre_phi+pad+1]), axis=-1)
            PFHCAL_crop = np.concatenate((PFHCAL[centre_eta-pad:centre_eta+pad+1,-diff:],
                                        PFHCAL[centre_eta-pad:centre_eta+pad+1,:centre_phi+pad+1]), axis=-1)
            PFECAL_crop = np.concatenate((PFECAL[centre_eta-pad:centre_eta+pad+1,-diff:],
                                        PFECAL[centre_eta-pad:centre_eta+pad+1,:centre_phi+pad+1]), axis=-1)
        elif 360-centre_phi <= pad: # Wrap-around on right side
            print("Wrapping right, iphi:", centre_phi)
            diff = pad - (360-centre_phi)
            ECAL_crop = np.concatenate((ECAL[centre_eta-pad:centre_eta+pad+1,centre_phi-pad:],
                                        ECAL[centre_eta-pad:centre_eta+pad+1,:diff+1]), axis=-1)
            Tracks_crop = np.concatenate((Tracks[centre_eta-pad:centre_eta+pad+1,centre_phi-pad:],
                                        Tracks[centre_eta-pad:centre_eta+pad+1,:diff+1]), axis=-1)
            FailedTracks_crop = np.concatenate((FailedTracks[centre_eta-pad:centre_eta+pad+1,centre_phi-pad:],
                                        FailedTracks[centre_eta-pad:centre_eta+pad+1,:diff+1]), axis=-1)
            PFHCAL_crop = np.concatenate((PFHCAL[centre_eta-pad:centre_eta+pad+1,centre_phi-pad:],
                                        PFHCAL[centre_eta-pad:centre_eta+pad+1,:diff+1]), axis=-1)
            PFECAL_crop = np.concatenate((PFECAL[centre_eta-pad:centre_eta+pad+1,centre_phi-pad:],
                                        PFECAL[centre_eta-pad:centre_eta+pad+1,:diff+1]), axis=-1)
        else:
            ECAL_crop = ECAL[centre_eta-pad:centre_eta+pad+1,centre_phi-pad:centre_phi+pad+1]
            Tracks_crop = Tracks[centre_eta-pad:centre_eta+pad+1,centre_phi-pad:centre_phi+pad+1]
            FailedTracks_crop = FailedTracks[centre_eta-pad:centre_eta+pad+1,centre_phi-pad:centre_phi+pad+1]
            PFHCAL_crop = PFHCAL[centre_eta-pad:centre_eta+pad+1,centre_phi-pad:centre_phi+pad+1]
            PFECAL_crop = PFECAL[centre_eta-pad:centre_eta+pad+1,centre_phi-pad:centre_phi+pad+1]

        return Tracks_crop, ECAL_crop, PFECAL_crop, PFHCAL_crop, FailedTracks_crop, cell_shift 
    
    def plot_tau(self, Tracks, ECAL, PFECAL, PFHCAL, FailedTracks, event, DM = [0, 1, 2, 10, 11, -1]):
        
        self.rhTree.GetEntry(event)
        
        centres = self.centre(event)
        truthDM = np.array(self.rhTree.jet_truthDM)
        neutral_ECAL = np.array(self.rhTree.neutralsum_ECAL)
        jet_pt = np.array(self.rhTree.jetPt)
        truthLabel = np.array(self.rhTree.jet_truthLabel) 
            
        # Load neutral gen positions
        gen_neutral_ieta = np.array(self.rhTree.jet_neutral_indv_releta_crystal, dtype=object)
        gen_neutral_iphi = np.array(self.rhTree.jet_neutral_indv_relphi_crystal, dtype=object)
        # Load charged gen positions
        gen_charged_ieta = np.array(self.rhTree.jet_charged_indv_releta_crystal, dtype=object)
        gen_charged_iphi = np.array(self.rhTree.jet_charged_indv_relphi_crystal, dtype=object)
        # Load total neutral gen position
        gen_total_neutral_ieta = np.array(self.rhTree.jet_neutral_releta_crystal, dtype=object)
        gen_total_neutral_iphi = np.array(self.rhTree.jet_neutral_relphi_crystal, dtype=object) 

        for i in range(len(truthDM)):
            if truthDM[i] in DM:
                
                ptcharged_elem = np.array(np.array(self.rhTree.jet_charged_indv_p)[i]) # gen momentum
                ptneutral_elem = np.array(np.array(self.rhTree.jet_neutral_indv_p)[i])
                p_total_neutral_elem = np.array(self.rhTree.jet_neutral_relp)[i]
                mass_total_neutral_elem = np.array(self.rhTree.jet_neutral_relmass)[i]
                
                centre_phi = int(centres[0][i]) 
                centre_eta = int(centres[1][i]) 
                
                if centre_phi>=360:
                    continue
                    
                print("Event Number: ", event, " Phi centre: ", centre_phi, " Eta centre: ", centre_eta)
                
                Tracks_crop, ECAL_crop, PFECAL_crop, PFHCAL_crop, FailedTracks_crop, shift  = self.crop_image(Tracks, ECAL, PFECAL, PFHCAL, FailedTracks, centre_eta, centre_phi)
        
                fig, ax = plt.subplots(3, 3, figsize = (18,13), gridspec_kw={'width_ratios': [1, 1.2, 0.4]}) 
    
                cbar = ax[0][1].pcolormesh( ECAL_crop, cmap=newcmp, norm = colors.LogNorm(vmin=1e-5, vmax=np.max(ECAL)))
                ax[0][0].pcolormesh(Tracks_crop, cmap=newcmp, norm = colors.LogNorm(vmin=1e-5, vmax=np.max(ECAL)))
                ax[1][0].pcolormesh(PFHCAL_crop, cmap=newcmp, norm = colors.LogNorm(vmin=1e-4, vmax=np.max(ECAL)))
                ax[1][1].pcolormesh(PFECAL_crop, cmap=newcmp, norm = colors.LogNorm(vmin=1e-4, vmax=np.max(ECAL)))
                ax[2][0].pcolormesh(FailedTracks_crop, cmap=newcmp, norm = colors.LogNorm(vmin=1e-4, vmax=np.max(ECAL)))
                
                ax[1][0].set_xlabel("i$\phi$ SHIFTED")
                ax[1][0].set_ylabel("i$\eta$ SHIFTED")
                ax[1][1].set_xlabel("i$\phi$ SHIFTED")
                ax[1][1].set_ylabel("i$\eta$ SHIFTED")
                ax[0][1].set_xlabel("i$\phi$ SHIFTED")
                ax[0][1].set_ylabel("i$\eta$ SHIFTED")
                ax[0][0].set_xlabel("i$\phi$ SHIFTED")
                ax[0][0].set_ylabel("i$\eta$ SHIFTED")
                ax[2][0].set_xlabel("i$\phi$ SHIFTED")
                ax[2][0].set_ylabel("i$\eta$ SHIFTED")
                
                ax[0][0].text(0.07, 0.88, "Tracks @ECAL", fontsize=14, transform=ax[0][0].transAxes, fontweight="bold")
                ax[0][1].text(0.07, 0.88, "ECAL RecHits", fontsize=14, transform=ax[0][1].transAxes, fontweight="bold")
                ax[1][0].text(0.07, 0.88, "PF HCAL energy", fontsize=14, transform=ax[1][0].transAxes, fontweight="bold")
                ax[1][1].text(0.07, 0.88, "PF ECAL energy", fontsize=14, transform=ax[1][1].transAxes, fontweight="bold")
                ax[2][0].text(0.07, 0.88, "Tracks failing cuts @ECAL", fontsize=14, transform=ax[2][0].transAxes, fontweight="bold")
                
                plt.colorbar(cbar, ax=ax[0][1])
                plt.colorbar(cbar, ax=ax[1][1])
                #plt.colorbar(cbar, ax=ax[2][0])
                
                sumstr = "$\Sigma$E: " + str(round(np.sum(Tracks_crop),2)) + " GeV"
                ax[0][0].text(0.07, 0.07, sumstr, fontsize=18, transform=ax[0][0].transAxes)
                
                sumstr = "$\Sigma$E: " + str(round(np.sum(ECAL_crop),2)) + " GeV"
                ax[0][1].text(0.07, 0.07, sumstr, fontsize=18, transform=ax[0][1].transAxes)
                
                sumstr = "$\Sigma$E: " + str(round(np.sum(PFECAL_crop),2)) + " GeV"
                ax[1][0].text(0.07, 0.07, sumstr, fontsize=18, transform=ax[1][1].transAxes)

                sumstr = "$\Sigma$E: " + str(round(np.sum(PFHCAL_crop),2)) + " GeV"
                ax[1][0].text(0.07, 0.07, sumstr, fontsize=18, transform=ax[1][0].transAxes)
                
                sumstr = "$\Sigma$E: " + str(round(np.sum(FailedTracks_crop),2)) + " GeV"
                ax[2][0].text(0.07, 0.07, sumstr, fontsize=18, transform=ax[2][0].transAxes)
                
                
                ax[0][2].axis("off")
                ax[1][2].axis("off")
                ax[2][1].axis("off")
                ax[2][2].axis("off")
                
                ax[0][0].set_xlim(0, 33)
                ax[0][0].set_ylim(0, 33)
                ax[0][1].set_xlim(0, 33)
                ax[0][1].set_ylim(0, 33)
                ax[1][1].set_xlim(0, 33)
                ax[1][1].set_ylim(0, 33)
                ax[1][0].set_xlim(0, 33)
                ax[1][0].set_ylim(0, 33)
                ax[2][0].set_xlim(0, 33)
                ax[2][0].set_ylim(0, 33)
                
                
                pad = 16 # need to add pad to index
                if truthDM[i]==1 or truthDM[i]==2 or truthDM[i]==11:
                        
                    print("Info: Truth DM is: ", truthDM[i], " HPS reconstructed tau as DM: ", np.array(self.rhTree.tau_dm)[i])
                    rel_total_neutral_ieta = np.array(gen_total_neutral_ieta[i]) - centre_eta + pad #+ shift # images look better without shift
                    rel_total_neutral_iphi = np.array(gen_total_neutral_iphi[i]) - centre_phi + pad
                    
                    
                    rel_neutral_ieta = np.array(gen_neutral_ieta[i]) - centre_eta + pad #+ shift
                    rel_neutral_iphi = np.array(gen_neutral_iphi[i]) - centre_phi + pad
                    for n in range(len(rel_neutral_iphi)):
                        if rel_neutral_iphi[n]<0:
                            rel_neutral_iphi[n] += 360
            
                    # print("$\pi^0$ coords: ", rel_neutral_iphi, rel_neutral_ieta)
                    ax[0][0].plot(rel_neutral_iphi, rel_neutral_ieta, linestyle="", marker = "*", label = "Gen $\pi^0$", markersize=12)
                    ax[1][0].plot(rel_neutral_iphi, rel_neutral_ieta, linestyle="", marker = "*", label = "Gen $\pi^0$", markersize=12)
                    ax[1][1].plot(rel_neutral_iphi, rel_neutral_ieta, linestyle="", marker = "*", label = "Gen $\pi^0$", markersize=12)
                    ax[0][1].plot(rel_neutral_iphi, rel_neutral_ieta, linestyle="", marker = "*", label = "Gen $\pi^0$", markersize=12)
                    ax[2][0].plot(rel_neutral_iphi, rel_neutral_ieta, linestyle="", marker = "*", label = "Gen $\pi^0$", markersize=12)
                    
                    if np.array(self.rhTree.tau_dm)[i] == 1 or np.array(self.rhTree.tau_dm)[i] == 11:
                        # HPS_eta = np.array(self.rhTree.HPSpi0_releta)[i]/0.0174 + 16.5
                        # HPS_phi = np.array(self.rhTree.HPSpi0_relphi)[i]/0.0174 + 16.5
                         # adjust since new center convention
                        HPS_eta = (np.array(self.rhTree.HPSpi0_releta)[i] + np.array(self.rhTree.jet_centre2_eta)[i] - np.array(self.rhTree.pi0_centre_eta)[i])/0.0174 + 16.5
                        HPS_phi = (np.array(self.rhTree.HPSpi0_relphi)[i] + np.array(self.rhTree.jet_centre2_phi)[i] - np.array(self.rhTree.pi0_centre_phi)[i])/0.0174 + 16.5
                        # print("releta: ", np.array(self.rhTree.HPSpi0_releta)[i]/0.0174)
                        # print("relphi: ", np.array(self.rhTree.HPSpi0_relphi)[i]/0.0174)
                        ax[0][0].plot(HPS_phi, HPS_eta, linestyle="", marker = "o", label = "HPS pi0", color='grey')
                        ax[0][1].plot(HPS_phi, HPS_eta, linestyle="", marker = "o", label = "HPS pi0", color='grey')
                        ax[1][0].plot(HPS_phi, HPS_eta, linestyle="", marker = "o", label = "HPS pi0", color='grey')
                        ax[1][1].plot(HPS_phi, HPS_eta, linestyle="", marker = "o", label = "HPS pi0", color='grey')
                        ax[2][0].plot(HPS_phi, HPS_eta, linestyle="", marker = "o", label = "HPS pi0", color='grey')
                    else:    
                        print("Warning, no reco pi0 as HPS DM: ", np.array(self.rhTree.tau_dm)[i])

                    ax[0][0].plot(16.5, 16.5, linestyle="", marker = "x", label = "Image Centre", markersize=12, color='black')
                    ax[1][0].plot(16.5, 16.5, linestyle="", marker = "x", label = "Image Centre", markersize=12, color='black')
                    ax[1][1].plot(16.5, 16.5, linestyle="", marker = "x", label = "Image Centre", markersize=12, color='black')
                    ax[0][1].plot(16.5, 16.5, linestyle="", marker = "x", label = "Image Centre", markersize=12, color='black')
                    ax[2][0].plot(16.5, 16.5, linestyle="", marker = "x", label = "Image Centre", markersize=12, color='black')
                
                if truthDM[i]!=-1:
                    rel_charged_ieta = np.array(gen_charged_ieta[i]) - centre_eta + pad #+ shift
                    rel_charged_iphi = np.array(gen_charged_iphi[i]) - centre_phi + pad
                    for n in range(len(rel_charged_iphi)):
                        if rel_charged_iphi[n]<0:
                            rel_charged_iphi[n] += 360
                            
                    # print("$\pi^\pm$ coords: ", rel_charged_iphi, rel_charged_ieta)
                    ax[0][0].plot(rel_charged_iphi, rel_charged_ieta, linestyle="", marker = "*", label = "Gen $\pi^\pm$", markersize=12)
                    ax[0][1].plot(rel_charged_iphi, rel_charged_ieta, linestyle="", marker = "*", label = "Gen $\pi^\pm$", markersize=12)
                    ax[1][0].plot(rel_charged_iphi, rel_charged_ieta, linestyle="", marker = "*", label = "Gen $\pi^\pm$", markersize=12)
                    ax[1][1].plot(rel_charged_iphi, rel_charged_ieta, linestyle="", marker = "*", label = "Gen $\pi^\pm$", markersize=12)
                    ax[2][0].plot(rel_charged_iphi, rel_charged_ieta, linestyle="", marker = "*", label = "Gen $\pi^\pm$", markersize=12)

                # print gen info below:
                l = -0.3
                Truthstr = "Gen Truth: " + str(truthLabel[i])
                ax[0][2].text(l, 0.86, Truthstr, fontsize=16, transform=ax[0][2].transAxes)
                DMstr = "Decay Mode: " + str(truthDM[i])
                ax[0][2].text(l, 0.79, DMstr, fontsize=16, transform=ax[0][2].transAxes)
                ECALstr = "PF $\gamma$ ECAL: " + str(round(neutral_ECAL[i], 2)) + " GeV"
                ax[0][2].text(l, 0.65, ECALstr, fontsize=16, transform=ax[0][2].transAxes)
                if len(ptcharged_elem)==1:
                    if ptcharged_elem != -1:
                        ptctxt = "Charged prongs: " 
                        ax[0][2].text(l, 0.58, ptctxt, fontsize=16, transform=ax[0][2].transAxes)
                        ptctxt = "$E$ :"  + str(round(ptcharged_elem[0], 2)) + " GeV" 
                        ax[0][2].text(l, 0.51, ptctxt, fontsize=16, transform=ax[0][2].transAxes)
                elif len(ptcharged_elem)==3:
                    ptctxt = "Charged prongs: " 
                    ax[0][2].text(l, 0.58, ptctxt, fontsize=16, transform=ax[0][2].transAxes)
                    ptctxt = "$E$ :"  + str(round(ptcharged_elem[0], 2)) + " GeV " 
                    ax[0][2].text(l, 0.51, ptctxt, fontsize=16, transform=ax[0][2].transAxes)
                    ptctxt = "$E$ :"  + str(round(ptcharged_elem[1], 2)) + " GeV " 
                    ax[0][2].text(l, 0.44, ptctxt, fontsize=16, transform=ax[0][2].transAxes)
                    ptctxt = "$E$ :"  + str(round(ptcharged_elem[2], 2)) + " GeV  "
                    ax[0][2].text(l, 0.37, ptctxt, fontsize=16, transform=ax[0][2].transAxes)
                if ptneutral_elem[0]!= -1:
                    if len(ptneutral_elem)==1:
                        ptctxt = "Neutral pions: " 
                        ax[0][2].text(l, 0.3, ptctxt, fontsize=16, transform=ax[0][2].transAxes)
                        ptctxt = "$E$: "  + str(round(ptneutral_elem[0], 2)) + " GeV  " 
                        ax[0][2].text(l, 0.23, ptctxt, fontsize=16, transform=ax[0][2].transAxes)
                    elif len(ptneutral_elem)==2:
                        ptctxt = "Neutral pions: " 
                        ax[0][2].text(l, 0.3, ptctxt, fontsize=16, transform=ax[0][2].transAxes)
                        ptctxt = "$E$ :"  + str(round(ptneutral_elem[0], 2)) + " GeV  "
                        ax[0][2].text(l, 0.23, ptctxt, fontsize=16, transform=ax[0][2].transAxes)
                        ptctxt = "$E$ :"  + str(round(ptneutral_elem[1], 2)) + " GeV " 
                        ax[0][2].text(l, 0.16, ptctxt, fontsize=16, transform=ax[0][2].transAxes)
                if mass_total_neutral_elem>0.:
                    ptctxt = "Total neutral:" 
                    ax[0][2].text(l, 0.09, ptctxt, fontsize=16, transform=ax[0][2].transAxes)
                    ptctxt = "$E$: "  + str(round(p_total_neutral_elem, 2)) + " GeV  " 
                    ax[0][2].text(l, 0.02, ptctxt, fontsize=16, transform=ax[0][2].transAxes)
                    massctxt = "$Mass$: "  + str(round(mass_total_neutral_elem, 2)) + " GeV  " 
                    ax[0][2].text(l, -0.05, massctxt, fontsize=16, transform=ax[0][2].transAxes)
                        
                ax[0][0].legend() 

                plt.show()
        
    def save_tau(self, Tracks, ECAL, PFECAL, PFHCAL, FailedTracks, event, DM = [0, 1, 2, 10, 11, -1]):
        self.rhTree.GetEntry(event)
        
        centres = self.centre(event)
        truthDM = np.array(self.rhTree.jet_truthDM)
        neutral_ECAL = np.array(self.rhTree.neutralsum_ECAL)
        jet_pt = np.array(self.rhTree.jetPt)
        truthLabel = np.array(self.rhTree.jet_truthLabel) 
            
        # Load neutral gen positions
        gen_neutral_ieta = np.array(self.rhTree.jet_neutral_indv_releta_crystal, dtype=object)
        gen_neutral_iphi = np.array(self.rhTree.jet_neutral_indv_relphi_crystal, dtype=object)
        # Load charged gen positions
        gen_charged_ieta = np.array(self.rhTree.jet_charged_indv_releta_crystal, dtype=object)
        gen_charged_iphi = np.array(self.rhTree.jet_charged_indv_relphi_crystal, dtype=object)
        # Load total neutral gen position
        gen_total_neutral_ieta = np.array(self.rhTree.jet_neutral_releta_crystal, dtype=object)
        gen_total_neutral_iphi = np.array(self.rhTree.jet_neutral_relphi_crystal, dtype=object) 

        for i in range(len(truthDM)):
            if truthDM[i] in DM:
                
                ptcharged_elem = np.array(np.array(self.rhTree.jet_charged_indv_p)[i]) # gen momentum
                ptneutral_elem = np.array(np.array(self.rhTree.jet_neutral_indv_p)[i])
                p_total_neutral_elem = np.array(self.rhTree.jet_neutral_relp)[i]
                mass_total_neutral_elem = np.array(self.rhTree.jet_neutral_relmass)[i]
                
                centre_phi = int(centres[0][i]) 
                centre_eta = int(centres[1][i]) 
                
                if centre_phi>=360:
                    continue
                    
                print("Event Number: ", event, " Phi centre: ", centre_phi, " Eta centre: ", centre_eta)
                
                Tracks_crop, ECAL_crop, PFECAL_crop, PFHCAL_crop, FailedTracks_crop, shift  = self.crop_image(Tracks, ECAL, PFECAL, PFHCAL, FailedTracks, centre_eta, centre_phi)

                ##################################################################################
                plt.figure(figsize=(5,4.5))
                plt.pcolormesh( Tracks_crop, cmap=newcmp, norm = colors.LogNorm(vmin=1e-5, vmax=np.max(ECAL)))
                plt.xlabel("i$\phi$")
                plt.ylabel("i$\eta$")
                plt.text(1, 30, "Tracks @ECAL", fontsize=14, fontweight="bold")
                sumstr = "$\Sigma$E: " + str(round(np.sum(Tracks_crop),2)) + " GeV"
                plt.text(1, 1, sumstr, fontsize=18)
                plt.xlim(0, 33)
                plt.ylim(0, 33)
                pad = 16 # need to add pad to index
                if truthDM[i]!=-1:
                    rel_charged_ieta = np.array(gen_charged_ieta[i]) - centre_eta + pad #+ shift
                    rel_charged_iphi = np.array(gen_charged_iphi[i]) - centre_phi + pad
                    for n in range(len(rel_charged_iphi)):
                        if rel_charged_iphi[n]<0:
                            rel_charged_iphi[n] += 360           
                    plt.plot(rel_charged_iphi, rel_charged_ieta, linestyle="", marker = "*", label = "Gen $\pi^\pm$", markersize=12)
                if truthDM[i]==1 or truthDM[i]==2 or truthDM[i]==11:
                    print("Info: Truth DM is: ", truthDM[i], " HPS reconstructed tau as DM: ", np.array(self.rhTree.tau_dm)[i])
                    rel_total_neutral_ieta = np.array(gen_total_neutral_ieta[i]) - centre_eta + pad #+ shift # images look better without shift
                    rel_total_neutral_iphi = np.array(gen_total_neutral_iphi[i]) - centre_phi + pad
                    rel_neutral_ieta = np.array(gen_neutral_ieta[i]) - centre_eta + pad #+ shift
                    rel_neutral_iphi = np.array(gen_neutral_iphi[i]) - centre_phi + pad
                    for n in range(len(rel_neutral_iphi)):
                        if rel_neutral_iphi[n]<0:
                            rel_neutral_iphi[n] += 360
                    plt.plot(rel_neutral_iphi, rel_neutral_ieta, linestyle="", marker = "*", label = "Gen $\pi^0$", markersize=12)
                    if np.array(self.rhTree.tau_dm)[i] == 1 or np.array(self.rhTree.tau_dm)[i] == 11:
                        HPS_eta = np.array(self.rhTree.HPSpi0_releta)[i]/0.0174 + 16.5
                        HPS_phi = np.array(self.rhTree.HPSpi0_relphi)[i]/0.0174 + 16.5
                        plt.plot(HPS_phi, HPS_eta, linestyle="", marker = "o", label = "HPS pi0", color='grey')
                    else:    
                        print("Warning, no reco pi0 as HPS DM: ", np.array(self.rhTree.tau_dm)[i])
                plt.legend()
                plt.savefig(f"/vols/cms/lcr119/Plots/Tracks/Event_{event}_{i}_Tracks.pdf", bbox_inches="tight")
                plt.show()
                ##################################################################################
                plt.figure(figsize=(5,4.5))
                cbar = plt.pcolormesh(ECAL_crop, cmap=newcmp, norm = colors.LogNorm(vmin=1e-5, vmax=np.max(ECAL)))
                plt.xlabel("i$\phi$")
                plt.ylabel("i$\eta$")
                plt.text(1, 30, "ECAL RecHits (Reduced)", fontsize=14, fontweight="bold")
                sumstr = "$\Sigma$E: " + str(round(np.sum(ECAL_crop),2)) + " GeV"
                plt.text(1, 1, sumstr, fontsize=18)
                plt.xlim(0, 33)
                plt.ylim(0, 33)
                pad = 16 # need to add pad to index
                # if truthDM[i]!=-1:
                #     rel_charged_ieta = np.array(gen_charged_ieta[i]) - centre_eta + pad #+ shift
                #     rel_charged_iphi = np.array(gen_charged_iphi[i]) - centre_phi + pad
                #     for n in range(len(rel_charged_iphi)):
                #         if rel_charged_iphi[n]<0:
                #             rel_charged_iphi[n] += 360           
                #     plt.plot(rel_charged_iphi, rel_charged_ieta, linestyle="", marker = "*", label = "Gen $\pi^\pm$", markersize=12)
                # if truthDM[i]==1 or truthDM[i]==2 or truthDM[i]==11:
                #     print("Info: Truth DM is: ", truthDM[i], " HPS reconstructed tau as DM: ", np.array(self.rhTree.tau_dm)[i])
                #     rel_total_neutral_ieta = np.array(gen_total_neutral_ieta[i]) - centre_eta + pad #+ shift # images look better without shift
                #     rel_total_neutral_iphi = np.array(gen_total_neutral_iphi[i]) - centre_phi + pad
                #     rel_neutral_ieta = np.array(gen_neutral_ieta[i]) - centre_eta + pad #+ shift
                #     rel_neutral_iphi = np.array(gen_neutral_iphi[i]) - centre_phi + pad
                #     for n in range(len(rel_neutral_iphi)):
                #         if rel_neutral_iphi[n]<0:
                #             rel_neutral_iphi[n] += 360
                #     plt.plot(rel_neutral_iphi, rel_neutral_ieta, linestyle="", marker = "*", label = "Gen $\pi^0$", markersize=12)
                #     if np.array(self.rhTree.tau_dm)[i] == 1:
                #         HPS_eta = np.array(self.rhTree.HPSpi0_releta)[i]/0.0174 + 16.5
                #         HPS_phi = np.array(self.rhTree.HPSpi0_relphi)[i]/0.0174 + 16.5
                #         plt.plot(HPS_phi, HPS_eta, linestyle="", marker = "o", label = "HPS pi0", color='grey')
                #     else:    
                #         print("Warning, no reco pi0 as HPS DM: ", np.array(self.rhTree.tau_dm)[i])
                plt.savefig(f"/vols/cms/lcr119/Plots/ECALRechit/Event_{event}_{i}_ECAL.pdf", bbox_inches="tight")
                plt.show()
                ###################################################
                plt.figure(figsize=(5,4.5))
                plt.pcolormesh( PFECAL_crop, cmap=newcmp, norm = colors.LogNorm(vmin=1e-5, vmax=np.max(ECAL)))
                plt.xlabel("i$\phi$")
                plt.ylabel("i$\eta$")
                plt.text(1, 30, "PF ECAL Energy", fontsize=14, fontweight="bold")
                sumstr = "$\Sigma$E: " + str(round(np.sum(PFECAL_crop),2)) + " GeV"
                plt.text(1, 1, sumstr, fontsize=18)
                plt.xlim(0, 33)
                plt.ylim(0, 33)
                pad = 16 # need to add pad to index
                if truthDM[i]!=-1:
                    rel_charged_ieta = np.array(gen_charged_ieta[i]) - centre_eta + pad #+ shift
                    rel_charged_iphi = np.array(gen_charged_iphi[i]) - centre_phi + pad
                    for n in range(len(rel_charged_iphi)):
                        if rel_charged_iphi[n]<0:
                            rel_charged_iphi[n] += 360           
                    plt.plot(rel_charged_iphi, rel_charged_ieta, linestyle="", marker = "*", label = "Gen $\pi^\pm$", markersize=12)
                if truthDM[i]==1 or truthDM[i]==2 or truthDM[i]==11:
                    print("Info: Truth DM is: ", truthDM[i], " HPS reconstructed tau as DM: ", np.array(self.rhTree.tau_dm)[i])
                    rel_total_neutral_ieta = np.array(gen_total_neutral_ieta[i]) - centre_eta + pad #+ shift # images look better without shift
                    rel_total_neutral_iphi = np.array(gen_total_neutral_iphi[i]) - centre_phi + pad
                    rel_neutral_ieta = np.array(gen_neutral_ieta[i]) - centre_eta + pad #+ shift
                    rel_neutral_iphi = np.array(gen_neutral_iphi[i]) - centre_phi + pad
                    for n in range(len(rel_neutral_iphi)):
                        if rel_neutral_iphi[n]<0:
                            rel_neutral_iphi[n] += 360
                    plt.plot(rel_neutral_iphi, rel_neutral_ieta, linestyle="", marker = "*", label = "Gen $\pi^0$", markersize=12)
                    if np.array(self.rhTree.tau_dm)[i] == 1:
                        HPS_eta = np.array(self.rhTree.HPSpi0_releta)[i]/0.0174 + 16.5
                        HPS_phi = np.array(self.rhTree.HPSpi0_relphi)[i]/0.0174 + 16.5
                        plt.plot(HPS_phi, HPS_eta, linestyle="", marker = "o", label = "HPS pi0", color='grey')
                    else:    
                        print("Warning, no reco pi0 as HPS DM: ", np.array(self.rhTree.tau_dm)[i])
                plt.savefig(f"/vols/cms/lcr119/Plots/PFECAL/Event_{event}_{i}_PFECAL.pdf", bbox_inches="tight")
                plt.show()
                ##################################################################################
                plt.figure(figsize=(5,4.5))
                plt.pcolormesh( PFHCAL_crop, cmap=newcmp, norm = colors.LogNorm(vmin=1e-5, vmax=np.max(ECAL)))
                plt.xlabel("i$\phi$")
                plt.ylabel("i$\eta$")
                plt.text(1, 30, "PF HCAL Energy", fontsize=14, fontweight="bold")
                sumstr = "$\Sigma$E: " + str(round(np.sum(PFHCAL_crop),2)) + " GeV"
                plt.text(1, 1, sumstr, fontsize=18)
                plt.xlim(0, 33)
                plt.ylim(0, 33)
                pad = 16 # need to add pad to index
                if truthDM[i]!=-1:
                    rel_charged_ieta = np.array(gen_charged_ieta[i]) - centre_eta + pad #+ shift
                    rel_charged_iphi = np.array(gen_charged_iphi[i]) - centre_phi + pad
                    for n in range(len(rel_charged_iphi)):
                        if rel_charged_iphi[n]<0:
                            rel_charged_iphi[n] += 360           
                    plt.plot(rel_charged_iphi, rel_charged_ieta, linestyle="", marker = "*", label = "Gen $\pi^\pm$", markersize=12)
                if truthDM[i]==1 or truthDM[i]==2 or truthDM[i]==11:
                    print("Info: Truth DM is: ", truthDM[i], " HPS reconstructed tau as DM: ", np.array(self.rhTree.tau_dm)[i])
                    rel_total_neutral_ieta = np.array(gen_total_neutral_ieta[i]) - centre_eta + pad #+ shift # images look better without shift
                    rel_total_neutral_iphi = np.array(gen_total_neutral_iphi[i]) - centre_phi + pad
                    rel_neutral_ieta = np.array(gen_neutral_ieta[i]) - centre_eta + pad #+ shift
                    rel_neutral_iphi = np.array(gen_neutral_iphi[i]) - centre_phi + pad
                    for n in range(len(rel_neutral_iphi)):
                        if rel_neutral_iphi[n]<0:
                            rel_neutral_iphi[n] += 360
                    plt.plot(rel_neutral_iphi, rel_neutral_ieta, linestyle="", marker = "*", label = "Gen $\pi^0$", markersize=12)
                    if np.array(self.rhTree.tau_dm)[i] == 1:
                        HPS_eta = np.array(self.rhTree.HPSpi0_releta)[i]/0.0174 + 16.5
                        HPS_phi = np.array(self.rhTree.HPSpi0_relphi)[i]/0.0174 + 16.5
                        plt.plot(HPS_phi, HPS_eta, linestyle="", marker = "o", label = "HPS pi0", color='grey')
                    else:    
                        print("Warning, no reco pi0 as HPS DM: ", np.array(self.rhTree.tau_dm)[i])
                plt.savefig(f"/vols/cms/lcr119/Plots/PFHCAL/Event_{event}_{i}_PFHCAL.pdf", bbox_inches="tight")
                plt.show()
                ########################################################################
                plt.figure(figsize=(5,4.5))
                plt.pcolormesh( FailedTracks_crop, cmap=newcmp, norm = colors.LogNorm(vmin=1e-5, vmax=np.max(ECAL)))
                plt.xlabel("i$\phi$")
                plt.ylabel("i$\eta$")
                plt.text(1, 30, "Additional Tracks @ECAL", fontsize=14, fontweight="bold")
                sumstr = "$\Sigma$E: " + str(round(np.sum(FailedTracks_crop),2)) + " GeV"
                plt.text(1, 1, sumstr, fontsize=18)
                plt.xlim(0, 33)
                plt.ylim(0, 33)
                pad = 16 # need to add pad to index
                if truthDM[i]!=-1:
                    rel_charged_ieta = np.array(gen_charged_ieta[i]) - centre_eta + pad #+ shift
                    rel_charged_iphi = np.array(gen_charged_iphi[i]) - centre_phi + pad
                    for n in range(len(rel_charged_iphi)):
                        if rel_charged_iphi[n]<0:
                            rel_charged_iphi[n] += 360           
                    plt.plot(rel_charged_iphi, rel_charged_ieta, linestyle="", marker = "*", label = "Gen $\pi^\pm$", markersize=12)
                if truthDM[i]==1 or truthDM[i]==2 or truthDM[i]==11:
                    print("Info: Truth DM is: ", truthDM[i], " HPS reconstructed tau as DM: ", np.array(self.rhTree.tau_dm)[i])
                    rel_total_neutral_ieta = np.array(gen_total_neutral_ieta[i]) - centre_eta + pad #+ shift # images look better without shift
                    rel_total_neutral_iphi = np.array(gen_total_neutral_iphi[i]) - centre_phi + pad
                    rel_neutral_ieta = np.array(gen_neutral_ieta[i]) - centre_eta + pad #+ shift
                    rel_neutral_iphi = np.array(gen_neutral_iphi[i]) - centre_phi + pad
                    for n in range(len(rel_neutral_iphi)):
                        if rel_neutral_iphi[n]<0:
                            rel_neutral_iphi[n] += 360
                    plt.plot(rel_neutral_iphi, rel_neutral_ieta, linestyle="", marker = "*", label = "Gen $\pi^0$", markersize=12)
                    if np.array(self.rhTree.tau_dm)[i] == 1:
                        HPS_eta = np.array(self.rhTree.HPSpi0_releta)[i]/0.0174 + 16.5
                        HPS_phi = np.array(self.rhTree.HPSpi0_relphi)[i]/0.0174 + 16.5
                        plt.plot(HPS_phi, HPS_eta, linestyle="", marker = "o", label = "HPS pi0", color='grey')
                    else:    
                        print("Warning, no reco pi0 as HPS DM: ", np.array(self.rhTree.tau_dm)[i])
                plt.savefig(f"/vols/cms/lcr119/Plots/addTracks/Event_{event}_{i}_addTracks.pdf", bbox_inches="tight")
                plt.show()


                # ax[0][2].axis("off")
                # ax[1][2].axis("off")
                # ax[2][1].axis("off")
                # ax[2][2].axis("off")
                
                # ax[0][0].set_xlim(0, 33)
                # ax[0][0].set_ylim(0, 33)
                # ax[0][1].set_xlim(0, 33)
                # ax[0][1].set_ylim(0, 33)
                # ax[1][1].set_xlim(0, 33)
                # ax[1][1].set_ylim(0, 33)
                # ax[1][0].set_xlim(0, 33)
                # ax[1][0].set_ylim(0, 33)
                # ax[2][0].set_xlim(0, 33)
                # ax[2][0].set_ylim(0, 33)
         
                # print gen info below:
                
                fig, ax = plt.subplots(figsize=(5,4.5))

                plt.colorbar(cbar, ax = [ax],location='left')
                
                l = 0.05
                Truthstr = "Gen Truth: " + str(truthLabel[i])
                ax.text(l, 0.86+0.05, Truthstr, fontsize=16, transform=ax.transAxes)
                DMstr = "Decay Mode: " + str(truthDM[i])
                ax.text(l, 0.79+0.05, DMstr, fontsize=16, transform=ax.transAxes)
                ECALstr = "PF $\gamma$ ECAL: " + str(round(neutral_ECAL[i], 2)) + " GeV"
                ax.text(l, 0.65+0.05, ECALstr, fontsize=16, transform=ax.transAxes)
                ax.axis("off")
                if len(ptcharged_elem)==1:
                    if ptcharged_elem != -1:
                        ptctxt = "Charged prongs: " 
                        ax.text(l, 0.58+0.05, ptctxt, fontsize=16, transform=ax.transAxes)
                        ptctxt = "$E$ :"  + str(round(ptcharged_elem[0], 2)) + " GeV" 
                        ax.text(l, 0.51+0.05, ptctxt, fontsize=16, transform=ax.transAxes)
                elif len(ptcharged_elem)==3:
                    ptctxt = "Charged prongs: " 
                    ax.text(l, 0.58+0.05, ptctxt, fontsize=16, transform=ax.transAxes)
                    ptctxt = "$E$ :"  + str(round(ptcharged_elem[0], 2)) + " GeV " 
                    ax.text(l, 0.51+0.05, ptctxt, fontsize=16, transform=ax.transAxes)
                    ptctxt = "$E$ :"  + str(round(ptcharged_elem[1], 2)) + " GeV " 
                    ax.text(l, 0.44+0.05, ptctxt, fontsize=16, transform=ax.transAxes)
                    ptctxt = "$E$ :"  + str(round(ptcharged_elem[2], 2)) + " GeV  "
                    ax.text(l, 0.37+0.05, ptctxt, fontsize=16, transform=ax.transAxes)
                if ptneutral_elem[0]!= -1:
                    if len(ptneutral_elem)==1:
                        ptctxt = "Neutral pions: " 
                        ax.text(l, 0.3+0.05, ptctxt, fontsize=16, transform=ax.transAxes)
                        ptctxt = "$E$: "  + str(round(ptneutral_elem[0], 2)) + " GeV  " 
                        ax.text(l, 0.23+0.05, ptctxt, fontsize=16, transform=ax.transAxes)
                    elif len(ptneutral_elem)==2:
                        ptctxt = "Neutral pions: " 
                        ax.text(l, 0.3+0.05, ptctxt, fontsize=16, transform=ax.transAxes)
                        ptctxt = "$E$ :"  + str(round(ptneutral_elem[0], 2)) + " GeV  "
                        ax.text(l, 0.23+0.05, ptctxt, fontsize=16, transform=ax.transAxes)
                        ptctxt = "$E$ :"  + str(round(ptneutral_elem[1], 2)) + " GeV " 
                        ax.text(l, 0.16+0.05, ptctxt, fontsize=16, transform=ax.transAxes)
                if mass_total_neutral_elem>0.:
                    ptctxt = "Total neutral:" 
                    ax.text(l, 0.09+0.05, ptctxt, fontsize=16, transform=ax.transAxes)
                    ptctxt = "$E$: "  + str(round(p_total_neutral_elem, 2)) + " GeV  " 
                    ax.text(l, 0.02+0.05, ptctxt, fontsize=16, transform=ax.transAxes)
                    massctxt = "$Mass$: "  + str(round(mass_total_neutral_elem, 2)) + " GeV  " 
                    ax.text(l, -0.05+0.05, massctxt, fontsize=16, transform=ax.transAxes)
                        
                # ax[0][0].legend() 
                plt.savefig(f"/vols/cms/lcr119/Plots/Info/Event_{event}_{i}_info.pdf", bbox_inches="tight")
                
                plt.show()