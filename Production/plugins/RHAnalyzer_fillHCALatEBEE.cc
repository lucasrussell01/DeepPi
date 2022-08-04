#include "DeepPi/Production/interface/RecHitAnalyzer.h"

// Fill EE rec hits /////////////////////////////////////////
// For each endcap, store event rechits in a vector of length 
// equal to number of crystals per endcap (ix:100 x iy:100)

TProfile2D *hHBHE_energy_EE_[nEE];
std::vector<float> vHBHE_energy_EE_[nEE];

// Initialize branches _____________________________________________________________//
void RecHitAnalyzer::branchesHCALatEBEE ( TTree* tree, edm::Service<TFileService> &fs ) {

  char hname[50], htitle[50];
  for ( int iz(0); iz < nEE; iz++ ) {
    // Branches for images
    const char *zside = (iz > 0) ? "p" : "m";
    sprintf(hname, "HBHE_energy_EE%s",zside);
    tree->Branch(hname,        &vHBHE_energy_EE_[iz]);

    // Histograms for monitoring
    sprintf(htitle,"E(ix,iy);ix;iy");
    hHBHE_energy_EE_[iz] = fs->make<TProfile2D>(hname, htitle,
        EE_MAX_IX, EE_MIN_IX-1, EE_MAX_IX,
        EE_MAX_IY, EE_MIN_IY-1, EE_MAX_IY );
  } // iz

} // branchesHCALatEBEE()

// Fill HCAL rechits at EB/EE ______________________________________________________________//
void RecHitAnalyzer::fillHCALatEBEE ( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

  int ix_, iy_, iz_, idx_; // rows:iy, columns:ix
  int nEExtals_filled; 
  float energy_;
  float eta, phi, minEta_, minPhi_, maxEta_, maxPhi_;
  GlobalPoint pos;
  bool isBoundary;
  std::vector<int> vEExtals_in_HBHEtower; 

  for ( int iz(0); iz < nEE; iz++ ) {
    vHBHE_energy_EE_[iz].assign( EE_NC_PER_ZSIDE, 0. );
  }

  edm::Handle<HBHERecHitCollection> HBHERecHitsH_;
  iEvent.getByToken( HBHERecHitCollectionT_, HBHERecHitsH_ );
  // Provides access to global cell position
  edm::ESHandle<CaloGeometry> caloGeomH_;
  iSetup.get<CaloGeometryRecord>().get( caloGeomH_ );
  const CaloGeometry* caloGeom;
  caloGeom = caloGeomH_.product();
  //const CaloCellGeometry::RepCorners& repCorners;

  for ( HBHERecHitCollection::const_iterator iRHit = HBHERecHitsH_->begin();
        iRHit != HBHERecHitsH_->end(); ++iRHit ) {
  //energy_ = 10;
  //for ( int ieta = 18; ieta < 29+1; ieta++) {
  //  for ( int iphi = 1; iphi < 72+1; iphi++) {
  //for ( int ieta = 19; ieta < 23; ieta++) {
  //  for ( int iphi = 1; iphi < 6; iphi++) {
      //if ( ieta > 20 && iphi%2==0 ) continue;  
      //HcalDetId hId(HcalEndcap, ieta, iphi, 1); //ieta,iphi,d
      //if ( iphi > 30 || iphi < 40 ) std::cout << "ieta"<<hId.ieta()<<",iphi"<<hId.iphi()<< std::endl;

      //energy_ = (ieta*iphi);
      //energy_ += 10.;
      energy_ = iRHit->energy();
      if ( energy_ <= zs ) continue;
      HcalDetId hId( iRHit->id() );

      if ( hId.ietaAbs() <= HBHE_IETA_MAX_EB ) continue;

      // Get REP corners of cell Id
      const auto repCorners = caloGeom->getGeometry(hId)->getCornersREP();
      // Get min,max phi,eta at plane closest to IP
      // See illustration at bottom
      isBoundary = false;
      minEta_ = repCorners[2].eta();
      maxEta_ = repCorners[0].eta();
      minPhi_ = repCorners[2].phi();
      maxPhi_ = repCorners[0].phi();
      if ( minPhi_ > maxPhi_ ) isBoundary = true;
      //if ( iphi > 30 || iphi < 40 ) std::cout << "(minEta:maxEta) "<<minEta_ <<":"<<maxEta_ << ", minPhi:maxPhi " << minPhi_<<":"<<maxPhi_<<std::endl;
      //std::cout << "(minEta:maxEta) "<<minEta_ <<":"<<maxEta_ << ", minPhi:maxPhi " << minPhi_<<":"<<maxPhi_ << " isBoundary:" << isBoundary <<std::endl;

      // Loop over all EE crystals
      vEExtals_in_HBHEtower.clear();
      for ( int iC = 0; iC < EEDetId::kSizeForDenseIndexing; iC++ ) {
        // Store EE xtals with centers within corners of HCAL tower
        EEDetId eeId( EEDetId::unhashIndex(iC) );
        pos = caloGeom->getPosition( eeId );
        eta = pos.eta();
        phi = pos.phi();
        if ( eta < minEta_ || eta > maxEta_ ) continue; 
        if ( !isBoundary ) {
          if ( phi < minPhi_ || phi > maxPhi_ ) continue; 
        } else {
          if ( phi < minPhi_ && phi > maxPhi_ ) continue; 
        }
        vEExtals_in_HBHEtower.push_back( iC );
        //std::cout << "xtal eta,phi: " << eta << " " << phi << std::endl;
        //if ( iphi > 30 || iphi < 40 ) std::cout << "xtal eta,phi: " << eta << " " << phi << std::endl;
      } // EE
      nEExtals_filled = vEExtals_in_HBHEtower.size();
      // Loop over selected EE xtals
      for ( int iC = 0; iC < nEExtals_filled; iC++ ) {
        // Split HCAL tower energy evenly among xtals
        EEDetId eeId( EEDetId::unhashIndex(vEExtals_in_HBHEtower[iC]) );
        ix_ = eeId.ix() - 1;
        iy_ = eeId.iy() - 1;
        iz_ = (eeId.zside() > 0) ? 1 : 0;
        // Create hashed Index: maps from [iy][ix] -> [idx_]
        idx_ = iy_*EE_MAX_IX + ix_;
        // Fill vector for images
        vHBHE_energy_EE_[iz_][idx_] += ( energy_/float(nEExtals_filled) );
        // Fill histogram for monitoring
        hHBHE_energy_EE_[iz_]->Fill( ix_, iy_, energy_ );
      } // EE, selected

    //} // HBHE rechits
  }

  /*
  for ( int ieta = 19; ieta < 24; ieta++) {
    for ( int iphi = 1; iphi < 7; iphi++) {
      //HcalDetId hId(HcalEndcap, ieta, iphi, 1); //ieta,iphi,d
      HcalDetId hId( iRHit->id() );
      //if ( hId.ieta() < 20 || hId.ieta() > 22 ) continue;

      pos = caloGeom->getGeometry(hId)->getPosition();
      const CaloCellGeometry::RepCorners& repCorners = caloGeom->getGeometry(hId)->getCornersREP();
      for ( unsigned c = 0; c < repCorners.size(); c++ ) {
        std::cout << c << ":rho,eta,phi: " << repCorners[c].rho() << " " << repCorners[c].eta() << " " << repCorners[c].phi() << std::endl;
      }
      //pos = caloGeom->getPosition( hId );
      eta = pos.eta();
      phi = pos.phi();
      //std::cout << "ieta"<<ieta<<hId.ieta()<<",iphi"<<iphi<<hId.iphi()<<": " << eta << " , " << phi << std::endl;
      std::cout << "ieta"<<hId.ieta()<<",iphi"<<hId.iphi()<<": " << eta << " , " << phi << std::endl;
    }
    std::cout << "" <<std::endl;
  }

  // Fill HBHE rechits
  for ( HBHERecHitCollection::const_iterator iRHit = HBHERecHitsH_->begin();
      iRHit != HBHERecHitsH_->end(); ++iRHit ) {

    energy_ = iRHit->energy();
    if ( energy_ <= zs ) continue;
    DetId id( spr::findDetIdECAL( caloGeom, eta, phi, false ) );

    if ( id.subdetId() == EcalBarrel ) continue; 

    // Get detector id and convert to histogram-friendly coordinates
    EEDetId eeId( id );
    ix_ = eeId.ix() - 1;
    iy_ = eeId.iy() - 1;
    iz_ = (eeId.zside() > 0) ? 1 : 0;
    // Fill histograms for monitoring
    hHBHE_energy_EE_[iz_]->Fill( ix_,iy_,energy_ );
    // Create hashed Index: maps from [iy][ix] -> [idx_]
    idx_ = iy_*EE_MAX_IX + ix_;
    // Fill vectors for images
    vHBHE_energy_EE_[iz_][idx_] += energy_;

  //if ( hId.eta() > 20 ) fillEEhit( HcalDetId hId(ieta_,iphi_+1,0) )

  } // HBHE rechits
  */

} // fillHCALatEBEE()

// HCAL REPcorners index illustration:
/*
         6_____7   eta
         /|   /|    |
        / |_ /_|   \|/ 
       / 5  / / 4   
      /    / /
     /    / /
    /    / /
  2/___3/ /
   |   | /
   |_ _|/   phi__\
  1     0        /
 
   / IP
 |/_ 

*/
