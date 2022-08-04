#include "DeepPi/Production/interface/RecHitAnalyzer.h"

// Fill EB rec hits ////////////////////////////////
// Store event rechits in a vector of length equal
// to number of crystals in EB (ieta:170 x iphi:360)

TProfile2D *hPFEB_energy;
TProfile2D *hPFEB_time;
std::vector<float> vPFEB_energy_;
std::vector<float> vPFEB_time_;

// Initialize branches _____________________________________________________//
void RecHitAnalyzer::branchesPFEB ( TTree* tree, edm::Service<TFileService> &fs ) {

  // Branches for images
  tree->Branch("PFEB_energy", &vPFEB_energy_);
  tree->Branch("PFEB_time",   &vPFEB_time_);

  // Histograms for monitoring
  hPFEB_energy = fs->make<TProfile2D>("PFEB_energy", "E(i#phi,i#eta);i#phi;i#eta",
      EB_IPHI_MAX  , EB_IPHI_MIN-1, EB_IPHI_MAX,
      2*EB_IETA_MAX,-EB_IETA_MAX,   EB_IETA_MAX );
  hPFEB_time = fs->make<TProfile2D>("PFEB_time", "t(i#phi,i#eta);i#phi;i#eta",
      EB_IPHI_MAX  , EB_IPHI_MIN-1, EB_IPHI_MAX,
      2*EB_IETA_MAX,-EB_IETA_MAX,   EB_IETA_MAX );

} // branchesPFEB()

// Fill EB rechits _________________________________________________________________//
void RecHitAnalyzer::fillPFEB ( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

  int iphi_, ieta_, idx_; // rows:ieta, cols:iphi
  float energy_;

  vPFEB_energy_.assign( EBDetId::kSizeForDenseIndexing, 0. );
  vPFEB_time_.assign( EBDetId::kSizeForDenseIndexing, 0. );

  edm::Handle<std::vector<reco::PFRecHit>> EBRecHitsH_;
  iEvent.getByToken( PFEBRecHitCollectionT_, EBRecHitsH_);


//  // Fill EB rechits 
  for ( std::vector<reco::PFRecHit>::const_iterator iRHit = EBRecHitsH_->begin();
        iRHit != EBRecHitsH_->end(); ++iRHit ) {


    energy_ = iRHit->energy();
    if ( energy_ <= zs ) continue;
    // Get detector id and convert to histogram-friendly coordinates
    EBDetId ebId( iRHit->detId() );
    iphi_ = ebId.iphi() - 1;
    ieta_ = ebId.ieta() > 0 ? ebId.ieta()-1 : ebId.ieta();
    // Fill histograms for monitoring
    hPFEB_energy->Fill( iphi_,ieta_,energy_ );
    hPFEB_time->Fill( iphi_,ieta_,iRHit->time() );
    // Get Hashed Index: provides convenient 
    // index mapping from [ieta][iphi] -> [idx]
    idx_ = ebId.hashedIndex(); // (ieta_+EB_IETA_MAX)*EB_IPHI_MAX + iphi_
    // Fill vectors for images
    vPFEB_energy_[idx_] = energy_;
    vPFEB_time_[idx_] = iRHit->time();

  } // EB rechits

} // fillPFEB()
