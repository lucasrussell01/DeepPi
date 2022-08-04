#include "DeepPi/Production/interface/RecHitAnalyzer.h"

// Fill ES rec hits /////////////////////////////////////////
// For each endcap, store event rechits in a vector of length 
// equal to number of crystals per endcap (ix:100 x iy:100)

TProfile2D *hES_energy[nES];
TProfile2D *hES_time[nES];
std::vector<float> vES_energy_[nES];
std::vector<float> vES_time_[nES];

// Initialize branches _____________________________________________________//
void RecHitAnalyzer::branchesES ( TTree* tree, edm::Service<TFileService> &fs ) {

  char hname[50], htitle[50];
  for ( int iz(0); iz < nES; iz++ ) {
    // Branches for images
    const char *zside = (iz > 0) ? "p" : "m";
    sprintf(hname, "ES%s_energy",zside);
    tree->Branch(hname,        &vES_energy_[iz]);
    sprintf(hname, "ES%s_time",  zside);
    tree->Branch(hname,        &vES_time_[iz]);

    // Histograms for monitoring
    sprintf(hname, "ES%s_energy",zside);
    sprintf(htitle,"E(ix,iy);ix;iy");
    hES_energy[iz] = fs->make<TProfile2D>(hname, htitle,
        ES_MAX_IX, ES_MIN_IX-1, ES_MAX_IX,
        ES_MAX_IY, ES_MIN_IY-1, ES_MAX_IY );
    sprintf(hname, "ES%s_time",zside);
    sprintf(htitle,"t(ix,iy);ix;iy");
    hES_time[iz] = fs->make<TProfile2D>(hname, htitle,
        ES_MAX_IX, ES_MIN_IX-1, ES_MAX_IX,
        ES_MAX_IY, ES_MIN_IY-1, ES_MAX_IY );
  } // iz

} // branchesES()

// Fill ES rechits _________________________________________________________________//
void RecHitAnalyzer::fillES ( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

  int ix_, iy_, iz_, idx_; // rows:iy, columns:ix
  float energy_;

  for ( int iz(0); iz < nES; iz++ ) {
    vES_energy_[iz].assign( ES_NC_PER_ZSIDE, 0. );
    vES_time_[iz].assign( ES_NC_PER_ZSIDE, 0. );
  }

  edm::Handle<EcalRecHitCollection> ESRecHitsH_;
  iEvent.getByToken( ESRecHitCollectionT_, ESRecHitsH_ );

  // Fill ES rechits
  for ( EcalRecHitCollection::const_iterator iRHit = ESRecHitsH_->begin();
        iRHit != ESRecHitsH_->end(); ++iRHit ) {

    energy_ = iRHit->energy();
    if ( energy_ <= zs ) continue;
    // Get detector id and convert to histogram-friendly coordinates
    ESDetId esId( iRHit->id() );
    ix_ = esId.six() - 1;
    iy_ = esId.siy() - 1;
    iz_ = (esId.zside() > 0) ? 1 : 0;

    // Fill histograms for monitoring
    hES_energy[iz_]->Fill( ix_, iy_, energy_ );
    hES_time[iz_]->Fill( ix_, iy_, iRHit->time() );
    // Create hashed Index: maps from [iy][ix] -> [idx_]
    idx_ = iy_*ES_MAX_IX + ix_;
    // Fill vectors for images
    vES_energy_[iz_][idx_] = energy_;
    vES_time_[iz_][idx_] = iRHit->time();

  } // ES rechits

} // fillES()
