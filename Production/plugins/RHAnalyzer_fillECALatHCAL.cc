#include "DeepPi/Production/interface/RecHitAnalyzer.h"

// Fill ECAL rechits at HCAL granularity /////////////////
// Project ECAL event rechits into a vector of length
// equal to number of *towers* in HBHE (iphi:72, ieta:56).
//
// NOTE: We do not decrease the iphi granularity as
// happens in the real HE. The findDetIdCalo() enforces
// the even iphi numbering in the coarse region so must
// resort to intermediate helper histograms. Since helper
// histograms are binned by eta,phi some approx. involved.

TH2F *hEvt_HBHE_EMenergy;
TProfile2D *hHBHE_EMenergy;
std::vector<float> vHBHE_EMenergy_;

// Initialize branches _____________________________________________________________//
void RecHitAnalyzer::branchesECALatHCAL ( TTree* tree, edm::Service<TFileService> &fs ) {

  // Branches for images
  tree->Branch("HBHE_EMenergy",    &vHBHE_EMenergy_);
  // Intermediate helper histogram (single event only)
  hEvt_HBHE_EMenergy = new TH2F("evt_HBHE_EMenergy", "E(#phi,#eta);#phi;#eta",
      HBHE_IPHI_NUM,         -TMath::Pi(),     TMath::Pi(),
      2*(HBHE_IETA_MAX_HE-1), eta_bins_HBHE );

  // Histograms for monitoring
  hHBHE_EMenergy = fs->make<TProfile2D>("HBHE_EMenergy", "E(i#phi,i#eta);i#phi;i#eta",
      HBHE_IPHI_NUM,           HBHE_IPHI_MIN-1,    HBHE_IPHI_MAX,
      2*(HBHE_IETA_MAX_HE-1),-(HBHE_IETA_MAX_HE-1),HBHE_IETA_MAX_HE-1 );

} // branchesECALatHCAL

// Fill ECAL rechits at HBHE granularity ___________________________________________________//
void RecHitAnalyzer::fillECALatHCAL ( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

  int ieta_, iphi_, idx_;
  float eta,  phi, energy_;
  GlobalPoint pos;

  vHBHE_EMenergy_.assign( 2*HBHE_IPHI_NUM*(HBHE_IETA_MAX_HE-1),0. );
  hEvt_HBHE_EMenergy->Reset();

  edm::Handle<EcalRecHitCollection> EBRecHitsH_;
  iEvent.getByToken( EBRecHitCollectionT_, EBRecHitsH_ );
  edm::Handle<EcalRecHitCollection> EERecHitsH_;
  iEvent.getByToken( EERecHitCollectionT_, EERecHitsH_ );
  edm::ESHandle<CaloGeometry> caloGeomH_;
  iSetup.get<CaloGeometryRecord>().get( caloGeomH_ );
  const CaloGeometry* caloGeom = caloGeomH_.product();

  // Fill EB rechits
  for ( EcalRecHitCollection::const_iterator iRHit = EBRecHitsH_->begin();
        iRHit != EBRecHitsH_->end(); ++iRHit ) {

    energy_ = iRHit->energy();
    if ( energy_ <= zs ) continue;
    // Get position of cell centers
    pos = caloGeom->getPosition( iRHit->id() );
    eta = pos.eta();
    phi = pos.phi();
    // Fill intermediate helper histogram by eta,phi
    hEvt_HBHE_EMenergy->Fill( phi, eta, energy_ );

    //HcalDetId hId( spr::findDetIdHCAL( caloGeom, eta, phi, false ) );

  } // EB rechits

  // Fill EE rechits
  for ( EcalRecHitCollection::const_iterator iRHit = EERecHitsH_->begin();
        iRHit != EERecHitsH_->end(); ++iRHit ) {

    energy_ = iRHit->energy();
    if ( energy_ <= zs ) continue;
    // Get position of cell centers
    pos = caloGeom->getPosition( iRHit->id() );
    eta = pos.eta();
    phi = pos.phi();
    // Fill intermediate helper histogram by eta,phi
    hEvt_HBHE_EMenergy->Fill( phi, eta, energy_ );

    //HcalDetId hId( spr::findDetIdHCAL( caloGeom, eta, phi, false ) );

  } // EE rechits

  // Fill vector for full ECAL@HCAL image using helper histograms
  for (int ieta = 1; ieta < hEvt_HBHE_EMenergy->GetNbinsY()+1; ieta++) {
    ieta_ = ieta - 1;
    for (int iphi = 1; iphi < hEvt_HBHE_EMenergy->GetNbinsX()+1; iphi++) {

      energy_ = hEvt_HBHE_EMenergy->GetBinContent( iphi, ieta );
      if ( energy_ <= zs ) continue;
      // NOTE: EB iphi = 1 does not correspond to physical phi = -pi so need to shift!
      iphi_ = iphi  + 38; // shift
      iphi_ = iphi_ > HBHE_IPHI_MAX ? iphi_-HBHE_IPHI_MAX : iphi_; // wrap-around
      iphi_ = iphi_ - 1;
      idx_  = ieta_*HBHE_IPHI_NUM + iphi_;
      // Fill vector for image
      vHBHE_EMenergy_[idx_] = energy_;
      // Fill histogram for monitoring
      hHBHE_EMenergy->Fill( iphi_, ieta_-(HBHE_IETA_MAX_HE-1), energy_ );

    } // iphi
  } // ieta

} // fillECALatHCAL()
