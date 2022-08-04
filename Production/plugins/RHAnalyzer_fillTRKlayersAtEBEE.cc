#include "DeepPi/Production/interface/RecHitAnalyzer.h"

// Fill TRK rec hits ////////////////////////////////
// by layer at EBEE 

//TH2F *hTOB_EE[nEE][nTOB];
TH2F *hTOB_EE[nTOB][nEE];
TH2F *hTOB_EB[nTOB];
TH1F *hTOB_layers;
//std::vector<float> vTOB_EE_[nEE][nTOB];
std::vector<float> vTOB_EE_[nTOB][nEE];
std::vector<float> vTOB_EB_[nTOB];

TH2F *hTEC_EE[nTEC][nEE];
TH2F *hTEC_EB[nTEC];
TH1F *hTEC_layers;
std::vector<float> vTEC_EE_[nTEC][nEE];
std::vector<float> vTEC_EB_[nTEC];

TH2F *hTIB_EE[nTIB][nEE];
TH2F *hTIB_EB[nTIB];
TH1F *hTIB_layers;
std::vector<float> vTIB_EE_[nTIB][nEE];
std::vector<float> vTIB_EB_[nTIB];

TH2F *hTID_EE[nTID][nEE];
TH2F *hTID_EB[nTID];
TH1F *hTID_layers;
std::vector<float> vTID_EE_[nTID][nEE];
std::vector<float> vTID_EB_[nTID];

TH2F *hBPIX_EE[nBPIX][nEE];
TH2F *hBPIX_EB[nBPIX];
TH1F *hBPIX_layers;
std::vector<float> vBPIX_EE_[nBPIX][nEE];
std::vector<float> vBPIX_EB_[nBPIX];

TH2F *hFPIX_EE[nFPIX][nEE];
TH2F *hFPIX_EB[nFPIX];
TH1F *hFPIX_layers;
std::vector<float> vFPIX_EE_[nFPIX][nEE];
std::vector<float> vFPIX_EB_[nFPIX];

// Initialize branches ____________________________________________________________//
void RecHitAnalyzer::branchesTRKlayersAtEBEE ( TTree* tree, edm::Service<TFileService> &fs ) {

  // Branches for images

  // Histograms for monitoring
  hTOB_layers = fs->make<TH1F>("TOB_layers", "N(layer);layer",
      nTOB+2, 0., nTOB+2. );
  hTEC_layers = fs->make<TH1F>("TEC_layers", "N(layer);layer",
      nTEC+2, 0., nTEC+2. );
  hTIB_layers = fs->make<TH1F>("TIB_layers", "N(layer);layer",
      nTIB+2, 0., nTIB+2. );
  hTID_layers = fs->make<TH1F>("TID_layers", "N(layer);layer",
      nTID+2, 0., nTID+2. );
  hBPIX_layers = fs->make<TH1F>("BPIX_layers", "N(layer);layer",
      nBPIX+2, 0., nBPIX+2. );
  hFPIX_layers = fs->make<TH1F>("FPIX_layers", "N(layer);layer",
      nFPIX+2, 0., nFPIX+2. );

  int layer;
  char hname[50], htitle[50];
  for ( int iL(0); iL < nTOB; iL++ ) {
    // Branches for images
    layer = iL + 1;
    sprintf(hname, "TOB_layer%d_EB",layer);
    tree->Branch(hname,        &vTOB_EB_[iL]);

    // Histograms for monitoring
    sprintf(htitle,"N(i#phi,i#eta);i#phi;i#eta");
    hTOB_EB[iL] = fs->make<TH2F>(hname, htitle,
        EB_IPHI_MAX  , EB_IPHI_MIN-1, EB_IPHI_MAX,
        2*EB_IETA_MAX,-EB_IETA_MAX,   EB_IETA_MAX );
    for ( int iz(0); iz < nEE; iz++ ) {
      const char *zside = (iz > 0) ? "p" : "m";
      sprintf(hname, "TOB_layer%d_EE%s",layer,zside);
      //tree->Branch(hname,        &vTOB_EE_[iz][iL]);
      tree->Branch(hname,        &vTOB_EE_[iL][iz]);

      // Histograms for monitoring
      sprintf(htitle,"N(ix,iy);ix;iy");
      //hTOB_EE[iz][iL] = fs->make<TH2F>(hname, htitle,
      hTOB_EE[iL][iz] = fs->make<TH2F>(hname, htitle,
          EE_MAX_IX, EE_MIN_IX-1, EE_MAX_IX,
          EE_MAX_IY, EE_MIN_IY-1, EE_MAX_IY );
    } // iz
  } // iL
  for ( int iL(0); iL < nTEC; iL++ ) {
    // Branches for images
    layer = iL + 1;
    sprintf(hname, "TEC_layer%d_EB",layer);
    tree->Branch(hname,        &vTEC_EB_[iL]);

    // Histograms for monitoring
    sprintf(htitle,"N(i#phi,i#eta);i#phi;i#eta");
    hTEC_EB[iL] = fs->make<TH2F>(hname, htitle,
        EB_IPHI_MAX  , EB_IPHI_MIN-1, EB_IPHI_MAX,
        2*EB_IETA_MAX,-EB_IETA_MAX,   EB_IETA_MAX );
    for ( int iz(0); iz < nEE; iz++ ) {
      const char *zside = (iz > 0) ? "p" : "m";
      sprintf(hname, "TEC_layer%d_EE%s",layer,zside);
      //tree->Branch(hname,        &vTEC_EE_[iz][iL]);
      tree->Branch(hname,        &vTEC_EE_[iL][iz]);

      // Histograms for monitoring
      sprintf(htitle,"N(ix,iy);ix;iy");
      //hTEC_EE[iz][iL] = fs->make<TH2F>(hname, htitle,
      hTEC_EE[iL][iz] = fs->make<TH2F>(hname, htitle,
          EE_MAX_IX, EE_MIN_IX-1, EE_MAX_IX,
          EE_MAX_IY, EE_MIN_IY-1, EE_MAX_IY );
    } // iz
  } // iL
  for ( int iL(0); iL < nTIB; iL++ ) {
    // Branches for images
    layer = iL + 1;
    sprintf(hname, "TIB_layer%d_EB",layer);
    tree->Branch(hname,        &vTIB_EB_[iL]);

    // Histograms for monitoring
    sprintf(htitle,"N(i#phi,i#eta);i#phi;i#eta");
    hTIB_EB[iL] = fs->make<TH2F>(hname, htitle,
        EB_IPHI_MAX  , EB_IPHI_MIN-1, EB_IPHI_MAX,
        2*EB_IETA_MAX,-EB_IETA_MAX,   EB_IETA_MAX );
    for ( int iz(0); iz < nEE; iz++ ) {
      const char *zside = (iz > 0) ? "p" : "m";
      sprintf(hname, "TIB_layer%d_EE%s",layer,zside);
      //tree->Branch(hname,        &vTIB_EE_[iz][iL]);
      tree->Branch(hname,        &vTIB_EE_[iL][iz]);

      // Histograms for monitoring
      sprintf(htitle,"N(ix,iy);ix;iy");
      //hTIB_EE[iz][iL] = fs->make<TH2F>(hname, htitle,
      hTIB_EE[iL][iz] = fs->make<TH2F>(hname, htitle,
          EE_MAX_IX, EE_MIN_IX-1, EE_MAX_IX,
          EE_MAX_IY, EE_MIN_IY-1, EE_MAX_IY );
    } // iz
  } // iL
  for ( int iL(0); iL < nTID; iL++ ) {
    // Branches for images
    layer = iL + 1;
    sprintf(hname, "TID_layer%d_EB",layer);
    tree->Branch(hname,        &vTID_EB_[iL]);

    // Histograms for monitoring
    sprintf(htitle,"N(i#phi,i#eta);i#phi;i#eta");
    hTID_EB[iL] = fs->make<TH2F>(hname, htitle,
        EB_IPHI_MAX  , EB_IPHI_MIN-1, EB_IPHI_MAX,
        2*EB_IETA_MAX,-EB_IETA_MAX,   EB_IETA_MAX );
    for ( int iz(0); iz < nEE; iz++ ) {
      const char *zside = (iz > 0) ? "p" : "m";
      sprintf(hname, "TID_layer%d_EE%s",layer,zside);
      //tree->Branch(hname,        &vTID_EE_[iz][iL]);
      tree->Branch(hname,        &vTID_EE_[iL][iz]);

      // Histograms for monitoring
      sprintf(htitle,"N(ix,iy);ix;iy");
      //hTID_EE[iz][iL] = fs->make<TH2F>(hname, htitle,
      hTID_EE[iL][iz] = fs->make<TH2F>(hname, htitle,
          EE_MAX_IX, EE_MIN_IX-1, EE_MAX_IX,
          EE_MAX_IY, EE_MIN_IY-1, EE_MAX_IY );
    } // iz
  } // iL
  for ( int iL(0); iL < nBPIX; iL++ ) {
    // Branches for images
    layer = iL + 1;
    sprintf(hname, "BPIX_layer%d_EB",layer);
    tree->Branch(hname,        &vBPIX_EB_[iL]);

    // Histograms for monitoring
    sprintf(htitle,"N(i#phi,i#eta);i#phi;i#eta");
    hBPIX_EB[iL] = fs->make<TH2F>(hname, htitle,
        EB_IPHI_MAX  , EB_IPHI_MIN-1, EB_IPHI_MAX,
        2*EB_IETA_MAX,-EB_IETA_MAX,   EB_IETA_MAX );
    for ( int iz(0); iz < nEE; iz++ ) {
      const char *zside = (iz > 0) ? "p" : "m";
      sprintf(hname, "BPIX_layer%d_EE%s",layer,zside);
      //tree->Branch(hname,        &vBPIX_EE_[iz][iL]);
      tree->Branch(hname,        &vBPIX_EE_[iL][iz]);

      // Histograms for monitoring
      sprintf(htitle,"N(ix,iy);ix;iy");
      //hBPIX_EE[iz][iL] = fs->make<TH2F>(hname, htitle,
      hBPIX_EE[iL][iz] = fs->make<TH2F>(hname, htitle,
          EE_MAX_IX, EE_MIN_IX-1, EE_MAX_IX,
          EE_MAX_IY, EE_MIN_IY-1, EE_MAX_IY );
    } // iz
  } // iL
  for ( int iL(0); iL < nFPIX; iL++ ) {
    // Branches for images
    layer = iL + 1;
    sprintf(hname, "FPIX_layer%d_EB",layer);
    tree->Branch(hname,        &vFPIX_EB_[iL]);

    // Histograms for monitoring
    sprintf(htitle,"N(i#phi,i#eta);i#phi;i#eta");
    hFPIX_EB[iL] = fs->make<TH2F>(hname, htitle,
        EB_IPHI_MAX  , EB_IPHI_MIN-1, EB_IPHI_MAX,
        2*EB_IETA_MAX,-EB_IETA_MAX,   EB_IETA_MAX );
    for ( int iz(0); iz < nEE; iz++ ) {
      const char *zside = (iz > 0) ? "p" : "m";
      sprintf(hname, "FPIX_layer%d_EE%s",layer,zside);
      //tree->Branch(hname,        &vFPIX_EE_[iz][iL]);
      tree->Branch(hname,        &vFPIX_EE_[iL][iz]);

      // Histograms for monitoring
      sprintf(htitle,"N(ix,iy);ix;iy");
      //hFPIX_EE[iz][iL] = fs->make<TH2F>(hname, htitle,
      hFPIX_EE[iL][iz] = fs->make<TH2F>(hname, htitle,
          EE_MAX_IX, EE_MIN_IX-1, EE_MAX_IX,
          EE_MAX_IY, EE_MIN_IY-1, EE_MAX_IY );
    } // iz
  } // iL

} // branchesEB()

void fillTRKatEB ( EBDetId ebId, int iL, TH2F *hTRK_EB[], std::vector<float> vTRK_EB_[] ) {
  int iphi_, ieta_, idx_;
  iphi_ = ebId.iphi() - 1;
  ieta_ = ebId.ieta() > 0 ? ebId.ieta()-1 : ebId.ieta();
  // Fill histograms for monitoring
  hTRK_EB[iL]->Fill( iphi_, ieta_ );
  idx_ = ebId.hashedIndex(); // (ieta_+EB_IETA_MAX)*EB_IPHI_MAX + iphi_
  // Fill vectors for images
  vTRK_EB_[iL][idx_] += 1.;
}

//template <std::size_t N, std::size_t M> void fillTRKatEE ( EEDetId eeId, int iL, TH2F (*hTRK_EE)[N][M], std::vector<float> vTRK_layers_EE[][nEE] ) {
void fillTRKatEE ( EEDetId eeId, int iL, TH2F *hTRK_EE[][nEE], std::vector<float> vTRK_EE_[][nEE] ) {
  int ix_, iy_, iz_, idx_;
  ix_ = eeId.ix() - 1;
  iy_ = eeId.iy() - 1;
  iz_ = (eeId.zside() > 0) ? 1 : 0;
  // Fill histograms for monitoring
  hTRK_EE[iL][iz_]->Fill( ix_, iy_ );
  // Create hashed Index: maps from [iy][ix] -> [idx_]
  idx_ = iy_*EE_MAX_IX + ix_;
  // Fill vectors for images
  vTRK_EE_[iL][iz_][idx_] += 1.;
}

// Fill TRK rechits at EB/EE ______________________________________________________________//
void RecHitAnalyzer::fillTRKlayersAtEBEE ( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

  //int ix_, iy_, iz_;
  //int iphi_, ieta_, idx_; // rows:ieta, cols:iphi
  int layer;
  float eta, phi;//, rho;
  GlobalPoint pos;

  for ( int iL(0); iL < nTOB; iL++ ) {
    vTOB_EB_[iL].assign( EBDetId::kSizeForDenseIndexing, 0. );
    for ( int iz(0); iz < nEE; iz++ ) {
      //vTOB_EE_[iz][iL].assign( EE_NC_PER_ZSIDE, 0. );
      vTOB_EE_[iL][iz].assign( EE_NC_PER_ZSIDE, 0. );
    }
  }
  for ( int iL(0); iL < nTEC; iL++ ) {
    vTEC_EB_[iL].assign( EBDetId::kSizeForDenseIndexing, 0. );
    for ( int iz(0); iz < nEE; iz++ ) {
      //vTEC_EE_[iz][iL].assign( EE_NC_PER_ZSIDE, 0. );
      vTEC_EE_[iL][iz].assign( EE_NC_PER_ZSIDE, 0. );
    }
  }
  for ( int iL(0); iL < nTIB; iL++ ) {
    vTIB_EB_[iL].assign( EBDetId::kSizeForDenseIndexing, 0. );
    for ( int iz(0); iz < nEE; iz++ ) {
      //vTIB_EE_[iz][iL].assign( EE_NC_PER_ZSIDE, 0. );
      vTIB_EE_[iL][iz].assign( EE_NC_PER_ZSIDE, 0. );
    }
  }
  for ( int iL(0); iL < nTID; iL++ ) {
    vTID_EB_[iL].assign( EBDetId::kSizeForDenseIndexing, 0. );
    for ( int iz(0); iz < nEE; iz++ ) {
      //vTID_EE_[iz][iL].assign( EE_NC_PER_ZSIDE, 0. );
      vTID_EE_[iL][iz].assign( EE_NC_PER_ZSIDE, 0. );
    }
  }
  for ( int iL(0); iL < nBPIX; iL++ ) {
    vBPIX_EB_[iL].assign( EBDetId::kSizeForDenseIndexing, 0. );
    for ( int iz(0); iz < nEE; iz++ ) {
      //vBPIX_EE_[iz][iL].assign( EE_NC_PER_ZSIDE, 0. );
      vBPIX_EE_[iL][iz].assign( EE_NC_PER_ZSIDE, 0. );
    }
  }
  for ( int iL(0); iL < nFPIX; iL++ ) {
    vFPIX_EB_[iL].assign( EBDetId::kSizeForDenseIndexing, 0. );
    for ( int iz(0); iz < nEE; iz++ ) {
      //vFPIX_EE_[iz][iL].assign( EE_NC_PER_ZSIDE, 0. );
      vFPIX_EE_[iL][iz].assign( EE_NC_PER_ZSIDE, 0. );
    }
  }

  edm::Handle<TrackingRecHitCollection> TRKRecHitsH_;
  iEvent.getByToken( TRKRecHitCollectionT_, TRKRecHitsH_ );
  // Provides access to global cell position
  edm::ESHandle<CaloGeometry> caloGeomH_;
  iSetup.get<CaloGeometryRecord>().get( caloGeomH_ );
  const CaloGeometry* caloGeom = caloGeomH_.product();

  edm::ESHandle<TrackerGeometry> tkGeomH_;
  iSetup.get<TrackerDigiGeometryRecord>().get( tkGeomH_ );
  const TrackerGeometry* tkGeom = tkGeomH_.product();

  //float maxEta = 0.;
  for ( TrackingRecHitCollection::const_iterator iRHit = TRKRecHitsH_->begin();
        iRHit != TRKRecHitsH_->end(); ++iRHit ) {

    if ( !iRHit->isValid() ) continue;
    DetId tkId( iRHit->geographicalId() );
    if ( tkId.det() != DetId::Tracker ) continue;
    pos = tkGeom->idToDet( tkId )->surface().toGlobal( iRHit->localPosition() );
    phi = pos.phi();
    eta = pos.eta();
    //rho = pos.perp();
    if ( std::abs(eta) > 3. ) continue;
    DetId ecalId( spr::findDetIdECAL( caloGeom, eta, phi, false ) );
    //if (std::abs(eta) > 1.479) std::cout << "eta:" << eta << std::endl;

    if ( tkId.subdetId() == StripSubdetector::TOB ) {

      //layer = TOBDetId( tkId ).layer();
      //hTOB_layers->Fill(layer);

      if ( ecalId.subdetId() == EcalBarrel )
        fillTRKatEB( EBDetId(ecalId), layer-1, hTOB_EB, vTOB_EB_ );
      else if ( ecalId.subdetId() == EcalEndcap )
        fillTRKatEE( EEDetId(ecalId), layer-1, hTOB_EE, vTOB_EE_ );

    } else if ( tkId.subdetId() == StripSubdetector::TEC ) {
    
      //layer = TECDetId( tkId ).wheel();
      //hTEC_layers->Fill(layer);
      if ( ecalId.subdetId() == EcalBarrel )
        fillTRKatEB( EBDetId(ecalId), layer-1, hTEC_EB, vTEC_EB_ );
      else if ( ecalId.subdetId() == EcalEndcap )
        fillTRKatEE( EEDetId(ecalId), layer-1, hTEC_EE, vTEC_EE_ );

    } else if ( tkId.subdetId() == StripSubdetector::TIB ) {
    
      //layer = TIBDetId( tkId ).layer();
      //hTIB_layers->Fill(layer);
      if ( ecalId.subdetId() == EcalBarrel )
        fillTRKatEB( EBDetId(ecalId), layer-1, hTIB_EB, vTIB_EB_ );
      else if ( ecalId.subdetId() == EcalEndcap )
        fillTRKatEE( EEDetId(ecalId), layer-1, hTIB_EE, vTIB_EE_ );

    } else if ( tkId.subdetId() == StripSubdetector::TID ) {
    
      //layer = TIDDetId( tkId ).wheel();
      //hTID_layers->Fill(layer);
      if ( ecalId.subdetId() == EcalBarrel )
        fillTRKatEB( EBDetId(ecalId), layer-1, hTID_EB, vTID_EB_ );
      else if ( ecalId.subdetId() == EcalEndcap )
        fillTRKatEE( EEDetId(ecalId), layer-1, hTID_EE, vTID_EE_ );

    } else if ( tkId.subdetId() == PixelSubdetector::PixelBarrel ) {
    
      layer = PXBDetId( tkId ).layer();
      hBPIX_layers->Fill(layer);
      if ( ecalId.subdetId() == EcalBarrel )
        fillTRKatEB( EBDetId(ecalId), layer-1, hBPIX_EB, vBPIX_EB_ );
      else if ( ecalId.subdetId() == EcalEndcap )
        fillTRKatEE( EEDetId(ecalId), layer-1, hBPIX_EE, vBPIX_EE_ );

    } else if ( tkId.subdetId() == PixelSubdetector::PixelEndcap ) {
    
      layer = PXFDetId( tkId ).disk();
      hFPIX_layers->Fill(layer);
      if ( ecalId.subdetId() == EcalBarrel )
        fillTRKatEB( EBDetId(ecalId), layer-1, hFPIX_EB, vFPIX_EB_ );
      else if ( ecalId.subdetId() == EcalEndcap )
        fillTRKatEE( EEDetId(ecalId), layer-1, hFPIX_EE, vFPIX_EE_ );

    }

  } // rechits
  //std::cout << maxEta << std::endl;

} // fillEB()
