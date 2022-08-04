
#include "DeepPi/Production/interface/RecHitAnalyzer.h"

// Fill Tracks into stitched EEm_EB_EEp image //////////////////////
// Store all Track positions into a stitched EEm_EB_EEp image 

TH2F *hEvt_EE_EndtracksPt[nEE];
TH2F *hEvt_EE_EndtracksQPt[nEE];
TH2F *hEvt_EE_EndtracksPt_PV[nEE];
TH2F *hEvt_EE_EndtracksQPt_PV[nEE];
TH2F *hEvt_EE_EndtracksPt_nPV[nEE];
TH2F *hEvt_EE_EndtracksQPt_nPV[nEE];
TH2F *hEvt_EE_muonsPt[nEE];
TH2F *hEvt_EE_muonsQPt[nEE];
TH2F *hEvt_EE_muonsPt_PV[nEE];
TH2F *hEvt_EE_muonsQPt_PV[nEE];

TProfile2D *hECAL_EndtracksPt;
TProfile2D *hECAL_muonsPt;
std::vector<float> vECAL_EndtracksPt_;
std::vector<float> vECAL_EndtracksQPt_;
std::vector<float> vECAL_EndtracksPt_PV_;
std::vector<float> vECAL_EndtracksQPt_PV_;
std::vector<float> vECAL_EndtracksPt_nPV_;
std::vector<float> vECAL_EndtracksQPt_nPV_;
std::vector<float> vECAL_muonsPt_;
std::vector<float> vECAL_muonsQPt_;
std::vector<float> vECAL_muonsPt_PV_;
std::vector<float> vECAL_muonsQPt_PV_;

// Initialize branches _______________________________________________________________//
void RecHitAnalyzer::branchesPFCandsAtECALstitched ( TTree* tree, edm::Service<TFileService> &fs ) {

  // Branches for images
  tree->Branch("ECAL_EndtracksPt",    &vECAL_EndtracksPt_);
  tree->Branch("ECAL_EndtracksQPt",   &vECAL_EndtracksQPt_);

  tree->Branch("ECAL_EndtracksPt_PV",    &vECAL_EndtracksPt_PV_);
  tree->Branch("ECAL_EndtracksQPt_PV",   &vECAL_EndtracksQPt_PV_);

  tree->Branch("ECAL_EndtracksPt_nPV",    &vECAL_EndtracksPt_nPV_);
  tree->Branch("ECAL_EndtracksQPt_nPV",   &vECAL_EndtracksQPt_nPV_);

  tree->Branch("ECAL_muonsPt",        &vECAL_muonsPt_);
  tree->Branch("ECAL_muonsQPt",       &vECAL_muonsQPt_);

  tree->Branch("ECAL_muonsPt_PV",        &vECAL_muonsPt_PV_);
  tree->Branch("ECAL_muonsQPt_PV",       &vECAL_muonsQPt_PV_);

  static const std::string strIndex[2] = {"m","p"};
  static const double* binIndex[2] = {eta_bins_EEm, eta_bins_EEp};


  // Intermediate helper histogram (single event only)
  for(unsigned int idx = 0; idx < 2; ++idx){
    hEvt_EE_EndtracksPt[idx] = new TH2F(("evt_EE"+strIndex[idx]+"_EndtracksPt").c_str(), "E(i#phi,i#eta);i#phi;i#eta",
					EB_IPHI_MAX, -TMath::Pi(), TMath::Pi(),
					5*(HBHE_IETA_MAX_HE-1-HBHE_IETA_MAX_EB), binIndex[idx] );

    hEvt_EE_EndtracksQPt[idx] = new TH2F(("evt_EE"+strIndex[idx]+"_EndtracksQPt").c_str(), "QPt(i#phi,i#eta);i#phi;i#eta",
					 EB_IPHI_MAX, -TMath::Pi(), TMath::Pi(),
					 5*(HBHE_IETA_MAX_HE-1-HBHE_IETA_MAX_EB), binIndex[idx] );


    hEvt_EE_EndtracksPt_PV[idx] = new TH2F(("evt_EE"+strIndex[idx]+"_EndtracksPt_PV").c_str(), "E(i#phi,i#eta);i#phi;i#eta",
					   EB_IPHI_MAX, -TMath::Pi(), TMath::Pi(),
					   5*(HBHE_IETA_MAX_HE-1-HBHE_IETA_MAX_EB), binIndex[idx] );

    hEvt_EE_EndtracksQPt_PV[idx] = new TH2F(("evt_EE"+strIndex[idx]+"_EndtracksQPt_PV").c_str(), "QPt(i#phi,i#eta);i#phi;i#eta",
					    EB_IPHI_MAX, -TMath::Pi(), TMath::Pi(),
					    5*(HBHE_IETA_MAX_HE-1-HBHE_IETA_MAX_EB), binIndex[idx] );

    hEvt_EE_EndtracksPt_nPV[idx] = new TH2F(("evt_EE"+strIndex[idx]+"_EndtracksPt_nPV").c_str(), "E(i#phi,i#eta);i#phi;i#eta",
					    EB_IPHI_MAX, -TMath::Pi(), TMath::Pi(),
					    5*(HBHE_IETA_MAX_HE-1-HBHE_IETA_MAX_EB), binIndex[idx] );

    hEvt_EE_EndtracksQPt_nPV[idx] = new TH2F(("evt_EE"+strIndex[idx]+"_EndtracksQPt_nPV").c_str(), "QPt(i#phi,i#eta);i#phi;i#eta",
					     EB_IPHI_MAX, -TMath::Pi(), TMath::Pi(),
					     5*(HBHE_IETA_MAX_HE-1-HBHE_IETA_MAX_EB), binIndex[idx] );



    hEvt_EE_muonsPt[idx] = new TH2F(("evt_EE"+strIndex[idx]+"_muonsPt").c_str(), "E(i#phi,i#eta);i#phi;i#eta",
				    EB_IPHI_MAX, -TMath::Pi(), TMath::Pi(),
				    5*(HBHE_IETA_MAX_HE-1-HBHE_IETA_MAX_EB), binIndex[idx] );

    hEvt_EE_muonsQPt[idx] = new TH2F(("evt_EE"+strIndex[idx]+"_muonsQPt").c_str(), "QPt(i#phi,i#eta);i#phi;i#eta",
				     EB_IPHI_MAX, -TMath::Pi(), TMath::Pi(),
				     5*(HBHE_IETA_MAX_HE-1-HBHE_IETA_MAX_EB), binIndex[idx] );

    hEvt_EE_muonsPt_PV[idx] = new TH2F(("evt_EE"+strIndex[idx]+"_muonsPt_PV").c_str(), "E(i#phi,i#eta);i#phi;i#eta",
				       EB_IPHI_MAX, -TMath::Pi(), TMath::Pi(),
				       5*(HBHE_IETA_MAX_HE-1-HBHE_IETA_MAX_EB), binIndex[idx] );

    hEvt_EE_muonsQPt_PV[idx] = new TH2F(("evt_EE"+strIndex[idx]+"_muonsQPt_PV").c_str(), "QPt(i#phi,i#eta);i#phi;i#eta",
					EB_IPHI_MAX, -TMath::Pi(), TMath::Pi(),
					5*(HBHE_IETA_MAX_HE-1-HBHE_IETA_MAX_EB), binIndex[idx] );

  }


  // Histograms for monitoring
  hECAL_EndtracksPt = fs->make<TProfile2D>("ECAL_EndtracksPt", "E(i#phi,i#eta);i#phi;i#eta",
      EB_IPHI_MAX,    EB_IPHI_MIN-1, EB_IPHI_MAX,
      2*ECAL_IETA_MAX_EXT, -ECAL_IETA_MAX_EXT,   ECAL_IETA_MAX_EXT );

  hECAL_muonsPt = fs->make<TProfile2D>("ECAL_muonsPt", "E(i#phi,i#eta);i#phi;i#eta",
      EB_IPHI_MAX,    EB_IPHI_MIN-1, EB_IPHI_MAX,
      2*ECAL_IETA_MAX_EXT, -ECAL_IETA_MAX_EXT,   ECAL_IETA_MAX_EXT );

} // branchesTracksAtECALstitched()

// Function to map EE(phi,eta) histograms to ECAL(iphi,ieta) vector _______________________________//
void fillPFCandsAtECAL_with_EEproj ( TH2F *hEvt_EE_EndtracksPt_, TH2F *hEvt_EE_EndtracksQPt_, 
				     TH2F *hEvt_EE_EndtracksPt_PV_, TH2F *hEvt_EE_EndtracksQPt_PV_, 
				     TH2F *hEvt_EE_EndtracksPt_nPV_, TH2F *hEvt_EE_EndtracksQPt_nPV_, 
				     TH2F *hEvt_EE_muonsPt_, TH2F *hEvt_EE_muonsQPt_, 
				     TH2F *hEvt_EE_muonsPt_PV, TH2F *hEvt_EE_muonsQPt_PV, 
				     int ieta_global_offset, int ieta_signed_offset ) {

  int ieta_global_, ieta_signed_;
  int ieta_, iphi_, idx_;
  float EndtrackPt_, EndtrackQPt_, EndtrackPt_PV_, EndtrackQPt_PV_, EndtrackPt_nPV_, EndtrackQPt_nPV_;
  float muonPt_, muonQPt_,muonPt_PV_, muonQPt_PV_;

  for (int ieta = 1; ieta < hEvt_EE_EndtracksPt_->GetNbinsY()+1; ieta++) {
    ieta_ = ieta - 1;
    ieta_global_ = ieta_ + ieta_global_offset;
    ieta_signed_ = ieta_ + ieta_signed_offset;
    for (int iphi = 1; iphi < hEvt_EE_EndtracksPt_->GetNbinsX()+1; iphi++) {

      EndtrackPt_  = hEvt_EE_EndtracksPt_->GetBinContent( iphi, ieta );
      EndtrackQPt_ = hEvt_EE_EndtracksQPt_->GetBinContent( iphi, ieta );
      EndtrackPt_PV_  = hEvt_EE_EndtracksPt_PV_->GetBinContent( iphi, ieta );
      EndtrackQPt_PV_ = hEvt_EE_EndtracksQPt_PV_->GetBinContent( iphi, ieta );
      EndtrackPt_nPV_  = hEvt_EE_EndtracksPt_nPV_->GetBinContent( iphi, ieta );
      EndtrackQPt_nPV_ = hEvt_EE_EndtracksQPt_nPV_->GetBinContent( iphi, ieta );
      muonPt_ = hEvt_EE_muonsPt_->GetBinContent( iphi, ieta );
      muonQPt_ = hEvt_EE_muonsQPt_->GetBinContent( iphi, ieta );
      muonPt_PV_ = hEvt_EE_muonsPt_PV->GetBinContent( iphi, ieta );
      muonQPt_PV_ = hEvt_EE_muonsQPt_PV->GetBinContent( iphi, ieta );
      if ( (EndtrackPt_ <= zs)  && (muonPt_ <= zs)) continue;
      // NOTE: EB iphi = 1 does not correspond to physical phi = -pi so need to shift!
      iphi_ = iphi  + 5*38; // shift
      iphi_ = iphi_ > EB_IPHI_MAX ? iphi_-EB_IPHI_MAX : iphi_; // wrap-around
      iphi_ = iphi_ - 1;
      idx_  = ieta_global_*EB_IPHI_MAX + iphi_;
      // Fill vector for image
      vECAL_EndtracksPt_[idx_]  = EndtrackPt_;
      vECAL_EndtracksQPt_[idx_] = EndtrackQPt_;
      vECAL_EndtracksPt_PV_[idx_]  = EndtrackPt_PV_;
      vECAL_EndtracksQPt_PV_[idx_] = EndtrackQPt_PV_;
      vECAL_EndtracksPt_nPV_[idx_]  = EndtrackPt_nPV_;
      vECAL_EndtracksQPt_nPV_[idx_] = EndtrackQPt_nPV_;
      vECAL_muonsPt_[idx_] = muonPt_;
      vECAL_muonsQPt_[idx_] = muonQPt_;
      vECAL_muonsPt_PV_[idx_] = muonPt_PV_;
      vECAL_muonsQPt_PV_[idx_] = muonQPt_PV_;
      // Fill histogram for monitoring
      hECAL_EndtracksPt->Fill( iphi_, ieta_signed_, EndtrackPt_ );
      hECAL_muonsPt->Fill( iphi_, ieta_signed_, muonPt_ );

    } // iphi_
  } // ieta_

} // fillPFCandsAtECAL_with_EEproj

// Fill stitched EE-, EB, EE+ rechits ________________________________________________________//
void RecHitAnalyzer::fillPFCandsAtECALstitched ( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

  int iphi_, ieta_, iz_, idx_;
  int ieta_global, ieta_signed;
  int ieta_global_offset, ieta_signed_offset;
  float eta, phi, trackPt_, trackQ_;
  GlobalPoint pos;

  vECAL_EndtracksPt_.assign( 2*ECAL_IETA_MAX_EXT*EB_IPHI_MAX, 0. );
  vECAL_EndtracksQPt_.assign( 2*ECAL_IETA_MAX_EXT*EB_IPHI_MAX, 0. );
  vECAL_EndtracksPt_PV_.assign( 2*ECAL_IETA_MAX_EXT*EB_IPHI_MAX, 0. );
  vECAL_EndtracksQPt_PV_.assign( 2*ECAL_IETA_MAX_EXT*EB_IPHI_MAX, 0. );
  vECAL_EndtracksPt_nPV_.assign( 2*ECAL_IETA_MAX_EXT*EB_IPHI_MAX, 0. );
  vECAL_EndtracksQPt_nPV_.assign( 2*ECAL_IETA_MAX_EXT*EB_IPHI_MAX, 0. );
  vECAL_muonsPt_.assign( 2*ECAL_IETA_MAX_EXT*EB_IPHI_MAX, 0. );
  vECAL_muonsQPt_.assign( 2*ECAL_IETA_MAX_EXT*EB_IPHI_MAX, 0. );
  vECAL_muonsPt_PV_.assign( 2*ECAL_IETA_MAX_EXT*EB_IPHI_MAX, 0. );
  vECAL_muonsQPt_PV_.assign( 2*ECAL_IETA_MAX_EXT*EB_IPHI_MAX, 0. );
  for ( int iz(0); iz < nEE; ++iz ) {
    hEvt_EE_EndtracksPt[iz]->Reset();
    hEvt_EE_EndtracksQPt[iz]->Reset();
    hEvt_EE_EndtracksPt_PV[iz]->Reset();
    hEvt_EE_EndtracksQPt_PV[iz]->Reset();
    hEvt_EE_EndtracksPt_nPV[iz]->Reset();
    hEvt_EE_EndtracksQPt_nPV[iz]->Reset();
    hEvt_EE_muonsPt[iz]->Reset();
    hEvt_EE_muonsQPt[iz]->Reset();
  }

  edm::Handle<EcalRecHitCollection> EBRecHitsH_;
  iEvent.getByToken( EBRecHitCollectionT_, EBRecHitsH_ );
  edm::Handle<EcalRecHitCollection> EERecHitsH_;
  iEvent.getByToken( EERecHitCollectionT_, EERecHitsH_ );
  edm::ESHandle<CaloGeometry> caloGeomH_;
  iSetup.get<CaloGeometryRecord>().get( caloGeomH_ );
  const CaloGeometry* caloGeom = caloGeomH_.product();

  edm::Handle<PFCollection> pfCandsH_;
  iEvent.getByToken( pfCollectionT_, pfCandsH_ );

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

    if ( std::abs(eta) > 3. ) continue;
    DetId id( spr::findDetIdECAL( caloGeom, eta, phi, false ) );
    if ( id.subdetId() == EcalBarrel ) continue;
    if ( id.subdetId() == EcalEndcap ) {
      iz_ = (eta > 0.) ? 1 : 0;

      // Fill intermediate helper histogram by eta,phi
      hEvt_EE_EndtracksPt[iz_]->Fill( phi, eta, thisTrk->pt() );
      hEvt_EE_EndtracksQPt[iz_]->Fill( phi, eta, thisTrk->charge() * thisTrk->pt() );

      if(iPFC->particleId() == 3){
	hEvt_EE_muonsPt[iz_]->Fill( phi, eta, thisTrk->pt() );
	hEvt_EE_muonsQPt[iz_]->Fill( phi, eta, thisTrk->charge()*thisTrk->pt() );
      }

      const double z0 = ( !vtxs.empty() ? thisTrk->dz(vtxs[0].position()) : thisTrk->dz() );

      // if is PV
      if(fabs(z0) < z0PVCut_){

	hEvt_EE_EndtracksPt_PV[iz_]->Fill( phi, eta, thisTrk->pt() );
	hEvt_EE_EndtracksQPt_PV[iz_]->Fill( phi, eta, thisTrk->charge() * thisTrk->pt() );

	if(iPFC->particleId() == 3){
	  hEvt_EE_muonsPt_PV[iz_]->Fill( phi, eta, thisTrk->pt() );
	  hEvt_EE_muonsQPt_PV[iz_]->Fill( phi, eta, thisTrk->charge()*thisTrk->pt() );
	}

      }else{	// if is not PV
	
	hEvt_EE_EndtracksPt_nPV[iz_]->Fill( phi, eta, thisTrk->pt() );
	hEvt_EE_EndtracksQPt_nPV[iz_]->Fill( phi, eta, thisTrk->charge() * thisTrk->pt() );
      }


    }
  } // pfCands



  // Map EE-(phi,eta) to bottom part of ECAL(iphi,ieta)
  ieta_global_offset = 0;
  ieta_signed_offset = -ECAL_IETA_MAX_EXT;
  fillPFCandsAtECAL_with_EEproj( hEvt_EE_EndtracksPt[0], hEvt_EE_EndtracksQPt[0],
				 hEvt_EE_EndtracksPt_PV[0], hEvt_EE_EndtracksQPt_PV[0],
				 hEvt_EE_EndtracksPt_nPV[0], hEvt_EE_EndtracksQPt_nPV[0],
				 hEvt_EE_muonsPt[0], hEvt_EE_muonsQPt[0], 
				 hEvt_EE_muonsPt_PV[0], hEvt_EE_muonsQPt_PV[0], 
				 ieta_global_offset, ieta_signed_offset );

  // Fill middle part of ECAL(iphi,ieta) with the EB rechits.
  ieta_global_offset = 55;


  for ( PFCollection::const_iterator iPFC = pfCandsH_->begin();
        iPFC != pfCandsH_->end(); ++iPFC ) {
    const reco::Track* thisTrk = iPFC->bestTrack();
    if(!thisTrk) continue;

    const math::XYZPointF& ecalPos = iPFC->positionAtECALEntrance();
    eta = ecalPos.eta();
    phi = ecalPos.phi();

    trackPt_ = thisTrk->pt();
    trackQ_  = thisTrk->charge();
    float trackz0_ =  ( !vtxs.empty() ? thisTrk->dz(vtxs[0].position()) : thisTrk->dz() );

    if ( std::abs(eta) > 3. ) continue;
    DetId id( spr::findDetIdECAL( caloGeom, eta, phi, false ) );
    if ( id.subdetId() == EcalEndcap ) continue;
    if ( id.subdetId() == EcalBarrel ) { 
      EBDetId ebId( id );
      iphi_ = ebId.iphi() - 1;
      ieta_ = ebId.ieta() > 0 ? ebId.ieta()-1 : ebId.ieta();
      if ( trackPt_ <= zs ) continue;
      // Fill vector for image
      ieta_signed = ieta_;
      ieta_global = ieta_ + EB_IETA_MAX + ieta_global_offset;
      idx_ = ieta_global*EB_IPHI_MAX + iphi_; 
      vECAL_EndtracksPt_[idx_] += trackPt_;
      vECAL_EndtracksQPt_[idx_] += (trackPt_*trackQ_);

      // Fill histogram for monitoring
      hECAL_EndtracksPt->Fill( iphi_, ieta_signed, trackPt_ );
      if(iPFC->particleId() == 3){
	vECAL_muonsPt_[idx_] += trackPt_;
	vECAL_muonsQPt_[idx_] += (trackPt_*trackQ_);
	hECAL_muonsPt->Fill( iphi_, ieta_signed, trackPt_ );
      }

      if(fabs(trackz0_) < z0PVCut_){
	vECAL_EndtracksPt_PV_[idx_] += trackPt_;
	vECAL_EndtracksQPt_PV_[idx_] += (trackPt_*trackQ_);

	if(iPFC->particleId() == 3){
	  vECAL_muonsPt_PV_[idx_] += trackPt_;
	  vECAL_muonsQPt_PV_[idx_] += (trackPt_*trackQ_);
	}

      }else{
	vECAL_EndtracksPt_nPV_[idx_] += trackPt_;
	vECAL_EndtracksQPt_nPV_[idx_] += (trackPt_*trackQ_);
      }


    }

  } // EB PFCands


  // Map EE+(phi,eta) to upper part of ECAL(iphi,ieta)
  ieta_global_offset = ECAL_IETA_MAX_EXT + EB_IETA_MAX;
  ieta_signed_offset = EB_IETA_MAX;
  fillPFCandsAtECAL_with_EEproj( hEvt_EE_EndtracksPt[1], hEvt_EE_EndtracksQPt[1], 
				 hEvt_EE_EndtracksPt_PV[1], hEvt_EE_EndtracksQPt_PV[1], 
				 hEvt_EE_EndtracksPt_nPV[1], hEvt_EE_EndtracksQPt_nPV[1], 
				 hEvt_EE_muonsPt[1], hEvt_EE_muonsQPt[1], 
				 hEvt_EE_muonsPt_PV[1], hEvt_EE_muonsQPt_PV[1], 
				 ieta_global_offset, ieta_signed_offset );

} // fillPFCandsAtECALstitched()
