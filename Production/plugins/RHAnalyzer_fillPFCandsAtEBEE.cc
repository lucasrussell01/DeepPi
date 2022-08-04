
#include "DeepPi/Production/interface/RecHitAnalyzer.h"

// Fill PFCands in EB+EE ////////////////////////////////
// Store PFCands in EB+EE projection

std::vector<float> vEndTracksPt_EE_[nEE];
std::vector<float> vEndTracksQPt_EE_[nEE];
std::vector<float> vEndTracksPt_EE_PV_[nEE];
std::vector<float> vEndTracksQPt_EE_PV_[nEE];
std::vector<float> vEndTracksPt_EE_nPV_[nEE];
std::vector<float> vEndTracksQPt_EE_nPV_[nEE];

std::vector<float> vMuonsPt_EE_[nEE];
std::vector<float> vMuonsQPt_EE_[nEE];
std::vector<float> vMuonsPt_EE_PV_[nEE];
std::vector<float> vMuonsQPt_EE_PV_[nEE];

std::vector<float> vEndTracksPt_EB_;
std::vector<float> vEndTracksQPt_EB_;
std::vector<float> vEndTracksPt_EB_PV_;
std::vector<float> vEndTracksQPt_EB_PV_;
std::vector<float> vEndTracksPt_EB_nPV_;
std::vector<float> vEndTracksQPt_EB_nPV_;
std::vector<float> vMuonsPt_EB_;
std::vector<float> vMuonsQPt_EB_;
std::vector<float> vMuonsPt_EB_PV_;
std::vector<float> vMuonsQPt_EB_PV_;

// Initialize branches ____________________________________________________________//
void RecHitAnalyzer::branchesPFCandsAtEBEE ( TTree* tree, edm::Service<TFileService> &fs ) {

  // Branches for images
  tree->Branch("EndTracksPt_EB",  &vEndTracksPt_EB_);
  tree->Branch("EndTracksQPt_EB", &vEndTracksQPt_EB_);
  tree->Branch("EndTracksPt_EB_PV",  &vEndTracksPt_EB_PV_);
  tree->Branch("EndTracksQPt_EB_PV", &vEndTracksQPt_EB_PV_);
  tree->Branch("EndTracksPt_EB_nPV",  &vEndTracksPt_EB_nPV_);
  tree->Branch("EndTracksQPt_EB_npPV", &vEndTracksQPt_EB_nPV_);
  tree->Branch("MuonsPt_EB",  &vMuonsPt_EB_);
  tree->Branch("MuonsQPt_EB", &vMuonsQPt_EB_);
  tree->Branch("MuonsPt_EB_PV",  &vMuonsPt_EB_PV_);
  tree->Branch("MuonsQPt_EB_PV", &vMuonsQPt_EB_PV_);

  char hname[50];
  for ( int iz(0); iz < nEE; iz++ ) {
    // Branches for images
    const char *zside = (iz > 0) ? "p" : "m";
    sprintf(hname, "EndTracksPt_EE%s",zside);      tree->Branch(hname,        &vEndTracksPt_EE_[iz]);
    sprintf(hname, "EndTracksQPt_EE%s",zside);     tree->Branch(hname,        &vEndTracksQPt_EE_[iz]);
    sprintf(hname, "EndTracksPt_PV_EE%s",zside);   tree->Branch(hname,        &vEndTracksPt_EE_PV_[iz]);
    sprintf(hname, "EndTracksQPt_PV_EE%s",zside);  tree->Branch(hname,        &vEndTracksQPt_EE_PV_[iz]);
    sprintf(hname, "EndTracksPt_nPV_EE%s",zside);  tree->Branch(hname,        &vEndTracksPt_EE_nPV_[iz]);
    sprintf(hname, "EndTracksQPt_nPV_EE%s",zside); tree->Branch(hname,        &vEndTracksQPt_EE_nPV_[iz]);

    sprintf(hname, "MuonsPt_EE%s",zside);          tree->Branch(hname,        &vMuonsPt_EE_[iz]);
    sprintf(hname, "MuonsQPt_EE%s",zside);         tree->Branch(hname,        &vMuonsQPt_EE_[iz]);
    sprintf(hname, "MuonsPt_PV_EE%s",zside);       tree->Branch(hname,        &vMuonsPt_EE_PV_[iz]);
    sprintf(hname, "MuonsQPt_PV_EE%s",zside);      tree->Branch(hname,        &vMuonsQPt_EE_PV_[iz]);
    
  } // iz

} // branchesEB()

// Fill TRK rechits at EB/EE ______________________________________________________________//
void RecHitAnalyzer::fillPFCandsAtEBEE ( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

  int ix_, iy_, iz_;
  int idx_; // rows:ieta, cols:iphi
  float eta, phi;
  GlobalPoint pos;

  vEndTracksPt_EB_.assign( EBDetId::kSizeForDenseIndexing, 0. );
  vEndTracksQPt_EB_.assign( EBDetId::kSizeForDenseIndexing, 0. );
  vEndTracksPt_EB_PV_.assign( EBDetId::kSizeForDenseIndexing, 0. );
  vEndTracksQPt_EB_PV_.assign( EBDetId::kSizeForDenseIndexing, 0. );
  vEndTracksPt_EB_nPV_.assign( EBDetId::kSizeForDenseIndexing, 0. );
  vEndTracksQPt_EB_nPV_.assign( EBDetId::kSizeForDenseIndexing, 0. );
  vMuonsPt_EB_.assign( EBDetId::kSizeForDenseIndexing, 0. );
  vMuonsQPt_EB_.assign( EBDetId::kSizeForDenseIndexing, 0. );
  vMuonsPt_EB_PV_.assign( EBDetId::kSizeForDenseIndexing, 0. );
  vMuonsQPt_EB_PV_.assign( EBDetId::kSizeForDenseIndexing, 0. );
  for ( int iz(0); iz < nEE; iz++ ) {
    vEndTracksPt_EE_[iz].assign( EE_NC_PER_ZSIDE, 0. );
    vEndTracksQPt_EE_[iz].assign( EE_NC_PER_ZSIDE, 0. );
    vEndTracksPt_EE_PV_[iz].assign( EE_NC_PER_ZSIDE, 0. );
    vEndTracksQPt_EE_PV_[iz].assign( EE_NC_PER_ZSIDE, 0. );
    vEndTracksPt_EE_nPV_[iz].assign( EE_NC_PER_ZSIDE, 0. );
    vEndTracksQPt_EE_nPV_[iz].assign( EE_NC_PER_ZSIDE, 0. );
    vMuonsPt_EE_[iz].assign( EE_NC_PER_ZSIDE, 0. );
    vMuonsQPt_EE_[iz].assign( EE_NC_PER_ZSIDE, 0. );
    vMuonsPt_EE_PV_[iz].assign( EE_NC_PER_ZSIDE, 0. );
    vMuonsQPt_EE_PV_[iz].assign( EE_NC_PER_ZSIDE, 0. );
  }

  edm::Handle<PFCollection> pfCandsH_;
  iEvent.getByToken( pfCollectionT_, pfCandsH_ );

  // Provides access to global cell position
  edm::ESHandle<CaloGeometry> caloGeomH_;
  iSetup.get<CaloGeometryRecord>().get( caloGeomH_ );
  const CaloGeometry* caloGeom = caloGeomH_.product();

  edm::Handle<reco::VertexCollection> vertexInfo;
  iEvent.getByToken(vertexCollectionT_, vertexInfo);
  const reco::VertexCollection& vtxs = *vertexInfo;


  for ( PFCollection::const_iterator iPFC = pfCandsH_->begin();
        iPFC != pfCandsH_->end(); ++iPFC ) {
    const reco::Track* thisTrk = iPFC->bestTrack();
    if(!thisTrk) continue;

    const math::XYZPointF& ecalPos = iPFC->positionAtECALEntrance();
    eta = ecalPos.eta();
    phi = ecalPos.phi();
    float z0    =  ( !vtxs.empty() ? thisTrk->dz(vtxs[0].position())  : thisTrk->dz() );
    
    if ( std::abs(eta) > 3. ) continue;
    DetId id( spr::findDetIdECAL( caloGeom, eta, phi, false ) );

    float thisTrkPt = thisTrk->pt();
    float thisTrkQPt = (thisTrk->pt()*thisTrk->charge());
    
    if ( id.subdetId() == EcalBarrel ) {
      EBDetId ebId( id );

      idx_ = ebId.hashedIndex(); // (ieta_+EB_IETA_MAX)*EB_IPHI_MAX + iphi_
      // Fill vectors for images
      
      vEndTracksPt_EB_[idx_] += thisTrkPt;
      vEndTracksQPt_EB_[idx_] += thisTrkQPt;
      if(iPFC->particleId() == 3){
	vMuonsPt_EB_[idx_] +=thisTrkPt;
	vMuonsPt_EB_[idx_] += thisTrkQPt;
      }

      if(fabs(z0) < z0PVCut_){
	vEndTracksPt_EB_PV_[idx_] += thisTrkPt;
	vEndTracksQPt_EB_PV_[idx_] += thisTrkQPt;

	if(iPFC->particleId() == 3){
	  vMuonsPt_EB_PV_[idx_] += thisTrkPt;
	  vMuonsPt_EB_PV_[idx_] += thisTrkQPt;
	}
	
      }else{
	vEndTracksPt_EB_nPV_[idx_] += thisTrkPt;
	vEndTracksQPt_EB_nPV_[idx_] += thisTrkQPt;
      }



    } else if ( id.subdetId() == EcalEndcap ) {
      EEDetId eeId( id );
      ix_ = eeId.ix() - 1;
      iy_ = eeId.iy() - 1;
      iz_ = (eeId.zside() > 0) ? 1 : 0;
        
      // Create hashed Index: maps from [iy][ix] -> [idx_]
      idx_ = iy_*EE_MAX_IX + ix_;
      // Fill vectors for images
      vEndTracksPt_EE_[iz_][idx_] += thisTrkPt;
      vEndTracksQPt_EE_[iz_][idx_] += thisTrkQPt;
      if(iPFC->particleId() == 3){
	vMuonsPt_EE_[iz_][idx_] += thisTrkPt;
	vMuonsQPt_EE_[iz_][idx_] += thisTrkQPt;
      }


      if(fabs(z0) < z0PVCut_){
	vEndTracksPt_EE_PV_[iz_][idx_] += thisTrkPt;
	vEndTracksQPt_EE_PV_[iz_][idx_] += thisTrkQPt;
	if(iPFC->particleId() == 3){
	  vMuonsPt_EE_PV_[iz_][idx_] += thisTrkPt;
	  vMuonsQPt_EE_PV_[iz_][idx_] += thisTrkQPt;
	}

      }else{
	vEndTracksPt_EE_nPV_[iz_][idx_] += thisTrkPt;
	vEndTracksQPt_EE_nPV_[iz_][idx_] += thisTrkQPt;
      }

    } 
  }//PF Candidates

} // fillEB()
