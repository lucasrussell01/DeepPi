
#include "DeepPi/Production/interface/RecHitAnalyzer.h"
#include "CommonTools/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "CommonTools/BaseParticlePropagator/interface/RawParticle.h"

// Fill Tracks in EB+EE ////////////////////////////////
// Store tracks in EB+EE projection

TH2F *hTracks_EE[nEE];
TH2F *hTracks_EB;
TH2F *hTracksPt_EE[nEE];
TH2F *hTracksPt_EB;
std::vector<float> vTracksPt_EE_[nEE];
std::vector<float> vTracksQPt_EE_[nEE];
std::vector<float> vTracks_EE_[nEE];


std::vector<float> vTracksPt_EB_;
std::vector<float> vTracksE_EB_;
std::vector<float> vTracksQPt_EB_;
std::vector<float> vTracks_EB_;


// HCAL PF info:
std::vector<float> vPF_HCAL_EB_;
std::vector<float> vPF_HCAL_EB_raw_;
// ECAL PF info
std::vector<float> vPF_ECAL_EB_;
std::vector<float> vPF_ECAL_EB_raw_;

// Initialize branches ____________________________________________________________//
void RecHitAnalyzer::branchesTracksAtEBEE ( TTree* tree, edm::Service<TFileService> &fs ) {

  // Branches for images
  tree->Branch("Tracks_EB",    &vTracks_EB_);
  tree->Branch("TracksPt_EB",  &vTracksPt_EB_);
  tree->Branch("TracksE_EB",  &vTracksE_EB_);
  tree->Branch("TracksQPt_EB", &vTracksQPt_EB_);

  tree->Branch("PF_HCAL_EB",     &vPF_HCAL_EB_);
  tree->Branch("PF_HCAL_EB_raw", &vPF_HCAL_EB_raw_);
  tree->Branch("PF_ECAL_EB",     &vPF_ECAL_EB_);
  tree->Branch("PF_ECAL_EB_raw", &vPF_ECAL_EB_raw_);

  // Histograms for monitoring
  hTracks_EB = fs->make<TH2F>("Tracks_EB", "N(i#phi,i#eta);i#phi;i#eta",
      EB_IPHI_MAX  , EB_IPHI_MIN-1, EB_IPHI_MAX,
      2*EB_IETA_MAX,-EB_IETA_MAX,   EB_IETA_MAX );
  hTracksPt_EB = fs->make<TH2F>("TracksPt_EB", "pT(i#phi,i#eta);i#phi;i#eta",
      EB_IPHI_MAX  , EB_IPHI_MIN-1, EB_IPHI_MAX,
      2*EB_IETA_MAX,-EB_IETA_MAX,   EB_IETA_MAX );


  char hname[50], htitle[50];
  for ( int iz(0); iz < nEE; iz++ ) {
    // Branches for images
    const char *zside = (iz > 0) ? "p" : "m";
    sprintf(hname, "Tracks_EE%s",zside);
    tree->Branch(hname,        &vTracks_EE_[iz]);
    sprintf(hname, "TracksPt_EE%s",zside);
    tree->Branch(hname,        &vTracksPt_EE_[iz]);
    sprintf(hname, "TracksQPt_EE%s",zside);
    tree->Branch(hname,        &vTracksQPt_EE_[iz]);

    // Histograms for monitoring
    sprintf(hname, "Tracks_EE%s",zside);
    sprintf(htitle,"N(ix,iy);ix;iy");
    hTracks_EE[iz] = fs->make<TH2F>(hname, htitle,
        EE_MAX_IX, EE_MIN_IX-1, EE_MAX_IX,
        EE_MAX_IY, EE_MIN_IY-1, EE_MAX_IY );
    sprintf(hname, "TracksPt_EE%s",zside);
    sprintf(htitle,"pT(ix,iy);ix;iy");
    hTracksPt_EE[iz] = fs->make<TH2F>(hname, htitle,
        EE_MAX_IX, EE_MIN_IX-1, EE_MAX_IX,
        EE_MAX_IY, EE_MIN_IY-1, EE_MAX_IY );
  } // iz

} // branchesEB()

// Fill TRK rechits at EB/EE ______________________________________________________________//
void RecHitAnalyzer::fillTracksAtEBEE ( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

  int ix_, iy_, iz_;
  int iphi_, ieta_, idx_; // rows:ieta, cols:iphi
  float eta, phi, pt, qpt, dxy, dz, energy;
  math::XYZVector position; //vector for propagating pfC
  GlobalPoint pos;

  vTracks_EB_.assign( EBDetId::kSizeForDenseIndexing, 0. );
  vTracksPt_EB_.assign( EBDetId::kSizeForDenseIndexing, 0. );
  vTracksE_EB_.assign( EBDetId::kSizeForDenseIndexing, 0. );
  vTracksQPt_EB_.assign( EBDetId::kSizeForDenseIndexing, 0. );


  // energy deposits on same grid resolution
  vPF_HCAL_EB_.assign( EBDetId::kSizeForDenseIndexing, 0. );
  vPF_HCAL_EB_raw_.assign( EBDetId::kSizeForDenseIndexing, 0. );
  vPF_ECAL_EB_.assign( EBDetId::kSizeForDenseIndexing, 0. );
  vPF_ECAL_EB_raw_.assign( EBDetId::kSizeForDenseIndexing, 0. );

  for ( int iz(0); iz < nEE; iz++ ) {
    vTracks_EE_[iz].assign( EE_NC_PER_ZSIDE, 0. );
    vTracksPt_EE_[iz].assign( EE_NC_PER_ZSIDE, 0. );
    vTracksQPt_EE_[iz].assign( EE_NC_PER_ZSIDE, 0. );

  }

  edm::Handle<reco::TrackCollection> tracksH_;
  iEvent.getByToken( trackCollectionT_, tracksH_ );



  // Provides access to global cell position
  edm::ESHandle<CaloGeometry> caloGeomH_;
  iSetup.get<CaloGeometryRecord>().get( caloGeomH_ );
  const CaloGeometry* caloGeom = caloGeomH_.product();

  edm::Handle<reco::VertexCollection> vertexInfo;
  iEvent.getByToken(vertexCollectionT_, vertexInfo);
  const reco::VertexCollection& vtxs = *vertexInfo;

  // std::cout << "PV: " << vtxs[0].position() << std::endl;

  reco::Track::TrackQuality tkQt_ = reco::Track::qualityByName("highPurity");

  edm::ESHandle<MagneticField> magfield;
  iSetup.get<IdealMagneticFieldRecord>().get(magfield);

  // load jets to loop through
  edm::Handle<reco::PFJetCollection> jets;
  iEvent.getByToken(jetCollectionT_, jets);

  for ( reco::TrackCollection::const_iterator iTk = tracksH_->begin();
        iTk != tracksH_->end(); ++iTk ) {
    
    energy = iTk->p(); // energy is roughly the same as p()
    pt    = iTk->pt();
    dxy = ( !vtxs.empty() ? iTk->dxy(vtxs[0].position()) : iTk->dxy() ); // 1st entry=PV
    dz = ( !vtxs.empty() ? iTk->dz(vtxs[0].position())  : iTk->dz() );
    qpt   = (iTk->charge()*pt);

    if (pt<0.5 || iTk->numberOfValidHits()<3 || std::abs(dz)>0.4 || std::abs(dxy)>0.1){ // tau reco like selection
      continue;
    }
    // get B field
    double magneticField = (magfield.product() ? magfield.product()->inTesla(GlobalPoint(0., 0., 0.)).z() : 0.0);
    math::XYZTLorentzVector  track_p4(iTk->px(),iTk->py(),iTk->pz(),sqrt(pow(iTk->p(),2)+0.14*0.14)); //setup 4-vector assuming mass is mass of charged pion
    BaseParticlePropagator propagator = BaseParticlePropagator(
        RawParticle(track_p4, math::XYZTLorentzVector(iTk->vx(), iTk->vy(), iTk->vz(), 0.),  iTk->charge()),
        0., 0., magneticField);
    propagator.propagateToEcalEntrance(false); // propogate to ECAL entrance
    auto track_position = propagator.particle().vertex().Vect();

    if ( std::abs(track_position.eta()) > 3. ) continue;

    DetId id( spr::findDetIdECAL( caloGeom, track_position.eta(), track_position.phi(), false ) );
      if ( id.subdetId() == EcalBarrel ) {
        EBDetId ebId( id );
        iphi_ = ebId.iphi() - 1;
        ieta_ = ebId.ieta() > 0 ? ebId.ieta()-1 : ebId.ieta();

        // Fill histograms for monitoring
        hTracks_EB->Fill( iphi_, ieta_ );
        hTracksPt_EB->Fill( iphi_, ieta_, pt );
        idx_ = ebId.hashedIndex(); // (ieta_+EB_IETA_MAX)*EB_IPHI_MAX + iphi_
        // Fill vectors for images
        vTracks_EB_[idx_] += 1.;
        vTracksPt_EB_[idx_] += pt;
        vTracksE_EB_[idx_] += energy;
        vTracksQPt_EB_[idx_] += qpt;
      } else if ( id.subdetId() == EcalEndcap ) {
        EEDetId eeId( id );
        ix_ = eeId.ix() - 1;
        iy_ = eeId.iy() - 1;
        iz_ = (eeId.zside() > 0) ? 1 : 0;
        // Fill histograms for monitoring
        hTracks_EE[iz_]->Fill( ix_, iy_ );
        hTracksPt_EE[iz_]->Fill( ix_, iy_, pt );
        // Create hashed Index: maps from [iy][ix] -> [idx_]
        idx_ = iy_*EE_MAX_IX + ix_;
        // Fill vectors for images
        vTracks_EE_   [iz_][idx_] += 1.;
        vTracksPt_EE_ [iz_][idx_] += pt;
        vTracksQPt_EE_[iz_][idx_] += qpt;
      }
      
    }

  for(int thisJetIdx : vJetIdxs){
    reco::PFJetRef thisJet( jets, thisJetIdx );

    std::vector<reco::PFCandidatePtr> pfCands = thisJet->getPFConstituents();

    for (const auto &pfC : pfCands){

    // check if charged/avoid null reference
    if (pfC->charge() == 0 || !pfC->trackRef().isNonnull()) { 
      // if not track propagate the pfC p4
      energy = pfC->p();
      pt = pfC->p4().pt();
      qpt = 0; // =0 since charge zero

      // get B field
      double magneticField = (magfield.product() ? magfield.product()->inTesla(GlobalPoint(0., 0., 0.)).z() : 0.0);
      math::XYZTLorentzVector  track_p4(pfC->p4().px(),pfC->p4().py(),pfC->p4().pz(),sqrt(pow(pfC->p(),2)+pfC->mass()*pfC->mass())); //setup 4-vector 
      BaseParticlePropagator propagator = BaseParticlePropagator(
          RawParticle(track_p4, math::XYZTLorentzVector(pfC->vx(), pfC->vy(), pfC->vz(), 0.),  pfC->charge()),
          0., 0., magneticField);
      propagator.propagateToEcalEntrance(false); // propogate to ECAL entrance
      position = propagator.particle().vertex().Vect();

    } else{
      // if track propagate the track ref p4
      reco::TrackRef iTk = pfC->trackRef();

      energy = iTk->p(); // energy is roughly the same as p()
      pt    = iTk->pt();
      qpt   = (iTk->charge()*pt);
      dxy = ( !vtxs.empty() ? iTk->dxy(vtxs[0].position()) : iTk->dxy() ); // 1st entry=PV
      dz = ( !vtxs.empty() ? iTk->dz(vtxs[0].position())  : iTk->dz() );

      if (pt<0.5 || iTk->numberOfValidHits()<3 || std::abs(dz)>0.4 || std::abs(dxy)>0.1){ // tau reco like selection
      continue;
      }
      // get B field
      double magneticField = (magfield.product() ? magfield.product()->inTesla(GlobalPoint(0., 0., 0.)).z() : 0.0);
      math::XYZTLorentzVector  track_p4(iTk->px(),iTk->py(),iTk->pz(),sqrt(pow(iTk->p(),2)+pfC->mass()*pfC->mass())); //setup 4-vector assuming mass is mass of charged pion
      BaseParticlePropagator propagator = BaseParticlePropagator(
          RawParticle(track_p4, math::XYZTLorentzVector(iTk->vx(), iTk->vy(), iTk->vz(), 0.),  iTk->charge()),
          0., 0., magneticField);
      propagator.propagateToEcalEntrance(false); // propogate to ECAL entrance
      position = propagator.particle().vertex().Vect();
    }
    
      if ( std::abs(position.eta()) > 3. ) continue;
 
      DetId id( spr::findDetIdECAL( caloGeom, position.eta(), position.phi(), false ) );
      if ( id.subdetId() == EcalBarrel ) {
        EBDetId ebId( id );
        iphi_ = ebId.iphi() - 1;
        ieta_ = ebId.ieta() > 0 ? ebId.ieta()-1 : ebId.ieta();

        idx_ = ebId.hashedIndex();
        // store HCAL energies
        vPF_HCAL_EB_[idx_] += pfC->hcalEnergy();
        vPF_HCAL_EB_raw_[idx_] += pfC->rawHcalEnergy();
        // store ECAL energies
        vPF_ECAL_EB_[idx_] += pfC->ecalEnergy();
        vPF_ECAL_EB_raw_[idx_] += pfC->rawEcalEnergy();

      } 
    }
  } // tracks


} // fillEB()
