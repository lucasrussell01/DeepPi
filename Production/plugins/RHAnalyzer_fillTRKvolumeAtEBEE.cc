#include "DeepPi/Production/interface/RecHitAnalyzer.h"

// Fill TRK rec hits ////////////////////////////////
// by volume at EBEE

TH3F *hTRK_EE[nEE];
TH3F *hTRK_EB;
TH1F *hTRK_EB_eta;
TH1F *hTRK_EB_phi;
TH1F *hTRK_EB_rho;
TH1F *hTRK_EE_z;
std::vector<float> vTRK_EE_[nEE];
std::vector<float> vTRK_EB_;

// Initialize branches ____________________________________________________________//
void RecHitAnalyzer::branchesTRKvolumeAtEBEE ( TTree* tree, edm::Service<TFileService> &fs ) {

  // Branches for images
  tree->Branch("TRK_volume_EB",   &vTRK_EB_);

  // Histograms for monitoring
  /*
  hTRK_EB = fs->make<TH3F>("TRK_volume_EB", "N(#phi,#eta,#rho);#phi;#eta;#rho",
      EB_IPHI_MAX  ,-TMath::Pi(),         TMath::Pi(),
      2*EB_IETA_MAX,-1.479,               1.479,
      12,                  0.,                  110. );
  */
  hTRK_EB = fs->make<TH3F>("TRK_volume_EB", "N(iphi,ieta,#rho);iphi;ieta;#rho",
      EB_IPHI_MAX  , EB_IPHI_MIN-1, EB_IPHI_MAX,
      2*EB_IETA_MAX      ,-EB_IETA_MAX  , EB_IETA_MAX, 
      12                 ,0.            , 110. );
  hTRK_EB_eta = fs->make<TH1F>("TRK_eta_EB", "N(eta);eta",
      2*EB_IETA_MAX,-1.479,               1.479 );
  hTRK_EB_phi = fs->make<TH1F>("TRK_phi_EB", "N(phi);phi",
      EB_IPHI_MAX  ,-TMath::Pi(),         TMath::Pi() );
  hTRK_EB_rho = fs->make<TH1F>("TRK_rho_EB", "N(rho);rho",
      120,                 0.,                  110. );

  char hname[50], htitle[50];
  float zMin, zMax;
  for ( int iz(0); iz < nEE; iz++ ) {
    // Branches for images
    const char *zside = (iz > 0) ? "p" : "m";
    sprintf(hname, "TRK_volume_EE%s",zside);
    tree->Branch(hname,        &vTRK_EE_[iz]);

    // Histograms for monitoring
    zMax = (iz > 0) ? 270. :    0.;
    zMin = (iz > 0) ?   0. : -270.;
    sprintf(hname, "TRK_volume_EE%s",zside);
    sprintf(htitle,"N(x,y,z);x;y,z");
    hTRK_EE[iz] = fs->make<TH3F>(hname, htitle,
        EE_MAX_IX, -100., 100.,
        EE_MAX_IY, -100., 100.,
        28,               zMin, zMax );
  } // iz
  hTRK_EE_z = fs->make<TH1F>("TRK_EE_z", "N(z);z",
      28*2, -270., 270. );

} // branchesEB()

// Fill TRK rechits at EB/EE ______________________________________________________________//
void RecHitAnalyzer::fillTRKvolumeAtEBEE ( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

  int ix_, iy_, iz_;
  int iphi_, ieta_, idx_; // rows:ieta, cols:iphi
  float eta, phi, rho, x, y, z;
  GlobalPoint pos;

  vTRK_EB_.assign( EBDetId::kSizeForDenseIndexing, 0. );
  for ( int iz(0); iz < nEE; iz++ ) {
    vTRK_EE_[iz].assign( EE_NC_PER_ZSIDE, 0. );
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

  for ( TrackingRecHitCollection::const_iterator iRHit = TRKRecHitsH_->begin();
        iRHit != TRKRecHitsH_->end(); ++iRHit ) {

    if ( !iRHit->isValid() ) continue;
    DetId tkId( iRHit->geographicalId() );
    if ( tkId.det() != DetId::Tracker ) continue;

    pos = tkGeom->idToDet( tkId )->surface().toGlobal( iRHit->localPosition() );
    phi = pos.phi();
    eta = pos.eta();
    rho = pos.perp();
    //x = pos.x();
    //y = pos.y();
    z = pos.z();

    if ( std::abs(eta) <= 1.479 ) { 

      //hTRK_EB->Fill( phi, eta, rho );
      hTRK_EB_phi->Fill( phi );
      hTRK_EB_eta->Fill( eta );
      hTRK_EB_rho->Fill( rho );

    } else if ( eta >  1.479 ) {
    } else if ( eta < -1.479 ) {
    }

    DetId id( spr::findDetIdECAL( caloGeom, eta, phi, false ) );
    if ( id.subdetId() == EcalBarrel ) {
      EBDetId ebId( id );
      iphi_ = ebId.iphi() - 1;
      ieta_ = ebId.ieta() > 0 ? ebId.ieta()-1 : ebId.ieta();
      // Fill histograms for monitoring
      hTRK_EB->Fill( iphi_, ieta_, rho );
    } else if ( id.subdetId() == EcalEndcap ) {
      EEDetId eeId( id );
      ix_ = eeId.ix() - 1;
      iy_ = eeId.iy() - 1;
      iz_ = (eeId.zside() > 0) ? 1 : 0;
      // Fill histograms for monitoring
      hTRK_EE[iz_]->Fill( ix_, iy_, z );
      hTRK_EE_z->Fill( z );
    }

    /*
    DetId id( spr::findDetIdECAL( caloGeom, eta, phi, false ) );
    if ( id.subdetId() == EcalBarrel ) {
      EBDetId ebId( id );
      iphi_ = ebId.iphi() - 1;
      ieta_ = ebId.ieta() > 0 ? ebId.ieta()-1 : ebId.ieta();
      // Fill histograms for monitoring
      hTRK_EB->Fill( iphi_, ieta_ );
      idx_ = ebId.hashedIndex(); // (ieta_+EB_IETA_MAX)*EB_IPHI_MAX + iphi_
      // Fill vectors for images
      vTRK_EB_[idx_] += 1.;
    } else if ( id.subdetId() == EcalEndcap ) {
      EEDetId eeId( id );
      ix_ = eeId.ix() - 1;
      iy_ = eeId.iy() - 1;
      iz_ = (eeId.zside() > 0) ? 1 : 0;
      // Fill histograms for monitoring
      hTRK_EE[iz_]->Fill( ix_, iy_ );
      // Create hashed Index: maps from [iy][ix] -> [idx_]
      idx_ = iy_*EE_MAX_IX + ix_;
      // Fill vectors for images
      vTRK_EE_[iz_][idx_] += 1.;
    } 
    */

  } // rechits

} // fillEB()
