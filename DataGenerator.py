import pyarrow.parquet as pq
import pyarrow 
import pandas as pd
import numpy as np
import ROOT as R

file = "/afs/cern.ch/user/l/lrussell/HTauTau/Images/test.parquet.0"

def reshape(data):
    # Reshape data stored in parquet to a 2D array
    data_reshape = np.zeros((np.shape(data)[0],np.shape(data)[0]))
    for i in range(np.shape(data)[0]):
        data_reshape[i, :] = data[i]
    return data_reshape

def crop(X):
    # Crop ECAL image to 33x33 centred around most energetic central track
    tracks = reshape(X[0])
    ECAL = reshape(X[1])
    # indices of central track
    cy, cx = np.where(tracks == np.max(tracks[40:85, 40:85]))
    lx = int(cx-16)
    ux = int(cx+17)
    ly = int(cy-16)
    uy = int(cy+17)
    # crop ECAL
    crop_ECAL = ECAL[lx:ux, ly:uy]
    return crop_ECAL

# import parquet file containing data
table = pq.read_table(file).to_pandas()
print("Loaded table")

data = table["X_jet"]
DM = table["DM"]

# Arrays to store data in:
X = np.zeros((len(data), 33, 33)) # Cropped ECAL images
Y = np.zeros(len(data)) # Truth (number of pi0s)
df = pd.DataFrame()

# Crop ECAL image to 33x33 centred around most energetic central track
print("Cropping images")
iEvent = 0
for tau, tauDM in zip(data, DM):
    iEvent += 1
    cropped_ECAL = crop(tau)
    X[iEvent, :, :] = cropped_ECAL
    if tauDM == 0 or tauDM == 10:
        Y[iEvent] = 0
    elif tauDM == 1 or tauDM == 11:
        Y[iEvent] = 1
    elif tauDM ==2:
        Y[iEvent] == 2
    else:
        Y[iEvent] == -1
    if iEvent%50==0:
        print("Event ", iEvent)
    if iEvent>50:
        break

df["X"] = X
df["Y"] = Y
# df.to_csv("/afs/cern.ch/user/l/lrussell/HTauTau/Data/cropped_images.csv")
# need to figure out format to save as, maybe direct generator?
    


