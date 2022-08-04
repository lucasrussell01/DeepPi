#include "DeepPi/Production/interface/RecHitAnalyzer.h"

// Fill JetInfo into stitched EEm_EB_EEp image //////////////////////

TH2F *hEvt_EE_tracksIP2D[nEE];
TH2F *hEvt_EE_tracksIP3D[nEE];
TH2F *hEvt_EE_tracksIP2Dsig[nEE];
TH2F *hEvt_EE_tracksIP3Dsig[nEE];

std::vector<float> vECAL_tracksIP2D_;
std::vector<float> vECAL_tracksIP3D_;
std::vector<float> vECAL_tracksIP2Dsig_;
std::vector<float> vECAL_tracksIP3Dsig_;

// Initialize branches _______________________________________________________________//
void RecHitAnalyzer::branchesJetInfoAtECALstitched ( TTree* tree, edm::Service<TFileService> &fs ) {

  // Branches for images
  tree->Branch("ECAL_tracksIP2D",       &vECAL_tracksIP2D_);
  tree->Branch("ECAL_tracksIP3D",       &vECAL_tracksIP3D_);
  tree->Branch("ECAL_tracksIP2Dsig",    &vECAL_tracksIP2Dsig_);
  tree->Branch("ECAL_tracksIP3Dsig",    &vECAL_tracksIP3Dsig_);

  static const std::string strIndex[2] = {"m","p"};
  static const double* binIndex[2] = {eta_bins_EEm, eta_bins_EEp};

  // Intermediate helper histogram (single event only)
  for(unsigned int idx = 0; idx < 2; ++idx){
    hEvt_EE_tracksIP2D[idx] = new TH2F(("evt_EE"+strIndex[idx]+"_tracksIP2D").c_str(), "IP2D(i#phi,i#eta);i#phi;i#eta",
					    EB_IPHI_MAX, -TMath::Pi(), TMath::Pi(),
					    5*(HBHE_IETA_MAX_HE-1-HBHE_IETA_MAX_EB), binIndex[idx] );

    hEvt_EE_tracksIP3D[idx] = new TH2F(("evt_EE"+strIndex[idx]+"_tracksIP3D").c_str(), "IP3D(i#phi,i#eta);i#phi;i#eta",
					    EB_IPHI_MAX, -TMath::Pi(), TMath::Pi(),
					    5*(HBHE_IETA_MAX_HE-1-HBHE_IETA_MAX_EB), binIndex[idx] );

    hEvt_EE_tracksIP2Dsig[idx] = new TH2F(("evt_EE"+strIndex[idx]+"_tracksIP2Dsig").c_str(), "IP2DSig(i#phi,i#eta);i#phi;i#eta",
					       EB_IPHI_MAX, -TMath::Pi(), TMath::Pi(),
					       5*(HBHE_IETA_MAX_HE-1-HBHE_IETA_MAX_EB), binIndex[idx] );

    hEvt_EE_tracksIP3Dsig[idx] = new TH2F(("evt_EE"+strIndex[idx]+"_tracksIP3Dsig").c_str(), "IP3DSig(i#phi,i#eta);i#phi;i#eta",
					       EB_IPHI_MAX, -TMath::Pi(), TMath::Pi(),
					       5*(HBHE_IETA_MAX_HE-1-HBHE_IETA_MAX_EB), binIndex[idx] );

  }


} // branchesTracksAtECALstitched()

// Function to map EE(phi,eta) histograms to ECAL(iphi,ieta) vector _______________________________//

void fillJetInfoAtECAL_with_EEproj ( TH2F *hEvt_EE_tracksIP2D_, TH2F *hEvt_EE_tracksIP3D_, TH2F *hEvt_EE_tracksIP2Dsig_, TH2F *hEvt_EE_tracksIP3Dsig_,
				    int ieta_global_offset, int ieta_signed_offset ) {

  int ieta_global_;//, ieta_signed_;
  int ieta_, iphi_, idx_;
  float trackIP2D, trackIP3D, trackIP2Dsig, trackIP3Dsig;

  for (int ieta = 1; ieta < hEvt_EE_tracksIP2D_->GetNbinsY()+1; ieta++) {
    ieta_ = ieta - 1;
    ieta_global_ = ieta_ + ieta_global_offset;
    //ieta_signed_ = ieta_ + ieta_signed_offset;
    for (int iphi = 1; iphi < hEvt_EE_tracksIP2D_->GetNbinsX()+1; iphi++) {

      trackIP2D    = hEvt_EE_tracksIP2D_   ->GetBinContent( iphi, ieta );
      trackIP3D    = hEvt_EE_tracksIP3D_   ->GetBinContent( iphi, ieta );
      trackIP2Dsig = hEvt_EE_tracksIP2Dsig_->GetBinContent( iphi, ieta );
      trackIP3Dsig = hEvt_EE_tracksIP3Dsig_->GetBinContent( iphi, ieta );

      // NOTE: EB iphi = 1 does not correspond to physical phi = -pi so need to shift!
      iphi_ = iphi  + 5*38; // shift
      iphi_ = iphi_ > EB_IPHI_MAX ? iphi_-EB_IPHI_MAX : iphi_; // wrap-around
      iphi_ = iphi_ - 1;
      idx_  = ieta_global_*EB_IPHI_MAX + iphi_;

      // Fill vector for image
      vECAL_tracksIP2D_[idx_]    = trackIP2D;
      vECAL_tracksIP3D_[idx_]    = trackIP3D;
      vECAL_tracksIP2Dsig_[idx_] = trackIP2Dsig;
      vECAL_tracksIP3Dsig_[idx_] = trackIP3Dsig;


    } // iphi_
  } // ieta_

} // fillTracksAtECAL_with_EEproj

// Fill stitched EE-, EB, EE+ rechits ________________________________________________________//
void RecHitAnalyzer::fillJetInfoAtECALstitched ( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

  int iphi_, ieta_, iz_, idx_;
  int ieta_global;
  int ieta_global_offset, ieta_signed_offset;
  float eta, phi, trackIP2D_, trackIP3D_, trackIP2Dsig_, trackIP3Dsig_;
  GlobalPoint pos;

  vECAL_tracksIP2D_.assign( 2*ECAL_IETA_MAX_EXT*EB_IPHI_MAX, 0. );
  vECAL_tracksIP3D_.assign( 2*ECAL_IETA_MAX_EXT*EB_IPHI_MAX, 0. );
  vECAL_tracksIP2Dsig_.assign( 2*ECAL_IETA_MAX_EXT*EB_IPHI_MAX, 0. );
  vECAL_tracksIP3Dsig_.assign( 2*ECAL_IETA_MAX_EXT*EB_IPHI_MAX, 0. );
  for ( int iz(0); iz < nEE; ++iz ){
    hEvt_EE_tracksIP2D[iz]->Reset();
    hEvt_EE_tracksIP3D[iz]->Reset();
    hEvt_EE_tracksIP2Dsig[iz]->Reset();
    hEvt_EE_tracksIP3Dsig[iz]->Reset();
  }

  edm::ESHandle<CaloGeometry> caloGeomH_;
  iSetup.get<CaloGeometryRecord>().get( caloGeomH_ );
  const CaloGeometry* caloGeom = caloGeomH_.product();

  edm::Handle<edm::View<reco::Jet> > recoJetCollection;
  iEvent.getByToken(recoJetsT_, recoJetCollection);

  edm::Handle<reco::PFJetCollection> jets;
  iEvent.getByToken(jetCollectionT_, jets);

  edm::Handle<std::vector<reco::CandIPTagInfo> > ipTagInfo;
  if (!ipTagInfoCollectionT_.isUninitialized()) {
    if(!iEvent.getByToken(ipTagInfoCollectionT_, ipTagInfo)) {
      return;
    }
  }

  for(int thisJetIdx : vJetIdxs){
    reco::PFJetRef thisJet( jets, thisJetIdx );

    // loop over jets
    for( edm::View<reco::Jet>::const_iterator jetToMatch = recoJetCollection->begin(); jetToMatch != recoJetCollection->end(); ++jetToMatch ){
      
      reco::Jet matchCand = *jetToMatch;
      float dR = reco::deltaR( thisJet->eta(),thisJet->phi(), matchCand.eta(),matchCand.phi() );
      if(dR > 0.1) continue;

      size_t idx = (jetToMatch - recoJetCollection->begin());
      edm::RefToBase<reco::Jet> jetRef = recoJetCollection->refAt(idx);

      for( std::vector<reco::CandIPTagInfo>::const_iterator itTI = ipTagInfo->begin(); itTI != ipTagInfo->end(); ++itTI ){
	if( itTI->jet() != jetRef ) continue;

	if(debug){
	  std::cout << "Have Match !!! " << std::endl;
	  std::cout << "\thasTracks: " << itTI->hasTracks() << std::endl;
	}

	const std::vector<reco::btag::TrackIPData> &ipData = itTI->impactParameterData();
	const auto &tracks = itTI->selectedTracks();

	if(debug){
	  std::cout << "\\tSize IPData: " << ipData.size() << std::endl;
	  std::cout << "\\tSize TrackSize: " << tracks.size() << std::endl;
	}
	    
	unsigned int nTracks = ipData.size();
	for(unsigned int idTrk = 0; idTrk < nTracks; ++idTrk){
	  //const auto &track = tracks[idTrk];
	  
	  //math::XYZVector trackMom = track->momentum();
	  const reco::Track * theTrack = itTI->selectedTrack(idTrk);
	  float eta = theTrack->eta();
	  float phi = theTrack->phi();
	  float IP2D = ipData[idTrk].ip2d.value();
	  float IP3D = ipData[idTrk].ip3d.value();
	  float IP2Dsig = ipData[idTrk].ip2d.significance();
	  float IP3Dsig = ipData[idTrk].ip3d.significance();

	  if ( std::abs(eta) > 3. ) continue;
	  DetId id( spr::findDetIdECAL( caloGeom, eta, phi, false ) );
	  if ( id.subdetId() == EcalBarrel ) {
	    // Fill middle part of ECAL(iphi,ieta) with the EB rechits.
	    ieta_global_offset = 55;

	    EBDetId ebId( id );
	    iphi_ = ebId.iphi() - 1;
	    ieta_ = ebId.ieta() > 0 ? ebId.ieta()-1 : ebId.ieta();

	    // Fill vector for image
	    ieta_global = ieta_ + EB_IETA_MAX + ieta_global_offset;
	    idx_ = ieta_global*EB_IPHI_MAX + iphi_; 
	    vECAL_tracksIP2D_[idx_] += IP2D;
	    vECAL_tracksIP3D_[idx_] += IP3D;
	    vECAL_tracksIP2Dsig_[idx_] += IP2Dsig;
	    vECAL_tracksIP3Dsig_[idx_] += IP3Dsig;

	  }


	  if ( id.subdetId() == EcalEndcap ) {
	    iz_ = (eta > 0.) ? 1 : 0;
	    // Fill intermediate helper histogram by eta,phi
	    hEvt_EE_tracksIP2D[iz_]->Fill( phi, eta, IP2D );
	    hEvt_EE_tracksIP3D[iz_]->Fill( phi, eta, IP3D );

	    hEvt_EE_tracksIP2Dsig[iz_]->Fill( phi, eta, IP2Dsig );
	    hEvt_EE_tracksIP3Dsig[iz_]->Fill( phi, eta, IP3Dsig );

	  }



	}//trks

      }//CandIP

    }// recoJetCollection


  }//vJetIdxs


  // Map EE-(phi,eta) to bottom part of ECAL(iphi,ieta)
  ieta_global_offset = 0;
  ieta_signed_offset = -ECAL_IETA_MAX_EXT;
  fillJetInfoAtECAL_with_EEproj( hEvt_EE_tracksIP2D[0], hEvt_EE_tracksIP3D[0],  hEvt_EE_tracksIP2Dsig[0], hEvt_EE_tracksIP3Dsig[0], ieta_global_offset, ieta_signed_offset );
  

  // Map EE+(phi,eta) to upper part of ECAL(iphi,ieta)
  ieta_global_offset = ECAL_IETA_MAX_EXT + EB_IETA_MAX;
  ieta_signed_offset = EB_IETA_MAX;
  fillJetInfoAtECAL_with_EEproj( hEvt_EE_tracksIP2D[1], hEvt_EE_tracksIP3D[1],  hEvt_EE_tracksIP2Dsig[1], hEvt_EE_tracksIP3Dsig[1], ieta_global_offset, ieta_signed_offset );


} // fillJetInfoAtECALstitched()
