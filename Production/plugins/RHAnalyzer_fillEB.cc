#include "DeepPi/Production/interface/RecHitAnalyzer.h"

// Fill EB rec hits ////////////////////////////////
// Store event rechits in a vector of length equal
// to number of crystals in EB (ieta:170 x iphi:360)

TProfile2D *hEB_energy;
TProfile2D *hEB_time;
std::vector<float> vEB_energy_;
std::vector<float> vEB_time_;

// Initialize branches _____________________________________________________//
void RecHitAnalyzer::branchesEB ( TTree* tree, edm::Service<TFileService> &fs ) {

  // Branches for images
  tree->Branch("EB_energy", &vEB_energy_);
  tree->Branch("EB_time",   &vEB_time_);

  // Histograms for monitoring
  hEB_energy = fs->make<TProfile2D>("EB_energy", "E(i#phi,i#eta);i#phi;i#eta",
      EB_IPHI_MAX  , EB_IPHI_MIN-1, EB_IPHI_MAX,
      2*EB_IETA_MAX,-EB_IETA_MAX,   EB_IETA_MAX );
  hEB_time = fs->make<TProfile2D>("EB_time", "t(i#phi,i#eta);i#phi;i#eta",
      EB_IPHI_MAX  , EB_IPHI_MIN-1, EB_IPHI_MAX,
      2*EB_IETA_MAX,-EB_IETA_MAX,   EB_IETA_MAX );

} // branchesEB()

// Fill EB rechits _________________________________________________________________//
void RecHitAnalyzer::fillEB ( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

  int iphi_, ieta_, idx_; // rows:ieta, cols:iphi
  float energy_;

  vEB_energy_.assign( EBDetId::kSizeForDenseIndexing, 0. );
  vEB_time_.assign( EBDetId::kSizeForDenseIndexing, 0. );

  edm::Handle<EcalRecHitCollection> EBRecHitsH_;
  iEvent.getByToken( EBRecHitCollectionT_, EBRecHitsH_);

  // Fill EB rechits 
  for ( EcalRecHitCollection::const_iterator iRHit = EBRecHitsH_->begin();
        iRHit != EBRecHitsH_->end(); ++iRHit ) {

    energy_ = iRHit->energy();
    if ( energy_ <= zs ) continue;
    // Get detector id and convert to histogram-friendly coordinates
    EBDetId ebId( iRHit->id() );
    iphi_ = ebId.iphi() - 1;
    ieta_ = ebId.ieta() > 0 ? ebId.ieta()-1 : ebId.ieta();
    // Fill histograms for monitoring 
    hEB_energy->Fill( iphi_,ieta_,energy_ );
    hEB_time->Fill( iphi_,ieta_,iRHit->time() );
    // Get Hashed Index: provides convenient 
    // index mapping from [ieta][iphi] -> [idx]
    idx_ = ebId.hashedIndex(); // (ieta_+EB_IETA_MAX)*EB_IPHI_MAX + iphi_
    // Fill vectors for images
    vEB_energy_[idx_] = energy_;
    vEB_time_[idx_] = iRHit->time();

  } // EB rechits

} // fillEB()

/*
// EB digis //
// This contains the raw EB digi collection:
// Each digi is unpacked as a data frame containing info
// from 10 time samples [iS] of the pulse shape
// iS=[0-2]: Presample noise
// iS=[3-9]: Nominal pulse shape
// NOTE: This is the raw collection and includes
// selective-readout and bad channel effects!

//____ Fill EB digis _____//
void RecHitAnalyzer::fillEBdigis ( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

  int iphi_, ieta_, idx;

  // Initialize data collection pointers
  edm::Handle<EBDigiCollection> EBDigisH;
  iEvent.getByToken(EBDigiCollectionT_, EBDigisH);

  // Provides access to global cell position and coordinates below
  edm::ESHandle<CaloGeometry> caloGeomH;
  iSetup.get<CaloGeometryRecord>().get(caloGeomH);
  caloGeom = caloGeomH.product();

  // Initialize arrays
  for(int iS(0); iS < EcalDataFrame::MAXSAMPLES; ++iS)
    vEB_adc_[iS].assign(EBDetId::kSizeForDenseIndexing,0);

  // Record signal-full entries
  for(EBDigiCollection::const_iterator iDigi = EBDigisH->begin();
      iDigi != EBDigisH->end();
      ++iDigi) {

    // Get detector id and convert to histogram-friendly coordinates
    EBDetId ebId( iDigi->id() );
    //DetId id( iDigi->id() );
    iphi_ = ebId.iphi()-1;
    ieta_ = ebId.ieta() > 0 ? ebId.ieta()-1 : ebId.ieta();

    // Get Hashed Index & Cell Geometry
    // Hashed index provides a convenient index mapping
    // from [ieta][iphi] -> [idx]
    idx = ebId.hashedIndex(); // (ieta_+EB_IETA_MAX)*EB_IPHI_MAX + iphi_
    // Cell geometry provides access to (rho,eta,phi) coordinates of cell center
    //cell  = caloGeom->getGeometry(ebId);

    // Unpack the digi into a dataframe
    EcalDataFrame df(*iDigi);
    for(int iS(0); iS < EcalDataFrame::MAXSAMPLES; ++iS) {

      // Get the iS-th sample
      EcalMGPASample digiSample( df.sample(iS) );

      // Fill some histograms to monitor distributions
      // These will contain *cumulative* statistics and as such
      // should be used for monitoring purposes only
      hEB_adc[iS]->Fill( iphi_, ieta_, digiSample.adc() );

      // Fill event arrays
      // These are the actual inputs to the detector images
      vEB_adc_[iS][idx] += digiSample.adc();
      //vEB_adc_[iS][idx] += digiSample.adc()/TMath::CosH(cell->etaPos()); // pick out only transverse component

    } // sample

  } // EB digi

} // fillEBdigis()
*/
