
#include "DeepPi/Production/interface/RecHitAnalyzer.h"
#include "CommonTools/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "CommonTools/BaseParticlePropagator/interface/RawParticle.h"

// Fill GsfTracks in EB+EE ////////////////////////////////
// Store tracks in EB+EE projection

TH2F *hGsfTracks_EE[nEE];
TH2F *hGsfTracks_EB;
TH2F *hGsfTracksPt_EE[nEE];
TH2F *hGsfTracksPt_EB;
std::vector<float> vGsfTracksPt_EE_[nEE];
std::vector<float> vGsfTracksQPt_EE_[nEE];
std::vector<float> vGsfTracks_EE_[nEE];


std::vector<float> vGsfTracksPt_EB_;
std::vector<float> vGsfTracksE_EB_;
std::vector<float> vGsfTracksQPt_EB_;
std::vector<float> vGsfTracks_EB_;

// Initialize branches ____________________________________________________________//
void RecHitAnalyzer::branchesGsfTracksAtEBEE ( TTree* tree, edm::Service<TFileService> &fs ) {

  // Branches for images
  tree->Branch("GsfTracks_EB",    &vGsfTracks_EB_);
  tree->Branch("GsfTracksPt_EB",  &vGsfTracksPt_EB_);
  tree->Branch("GsfTracksE_EB",  &vGsfTracksE_EB_);
  tree->Branch("GsfTracksQPt_EB", &vGsfTracksQPt_EB_);


  // Histograms for monitoring
  hGsfTracks_EB = fs->make<TH2F>("GsfTracks_EB", "N(i#phi,i#eta);i#phi;i#eta",
      EB_IPHI_MAX  , EB_IPHI_MIN-1, EB_IPHI_MAX,
      2*EB_IETA_MAX,-EB_IETA_MAX,   EB_IETA_MAX );
  hGsfTracksPt_EB = fs->make<TH2F>("GsfTracksPt_EB", "pT(i#phi,i#eta);i#phi;i#eta",
      EB_IPHI_MAX  , EB_IPHI_MIN-1, EB_IPHI_MAX,
      2*EB_IETA_MAX,-EB_IETA_MAX,   EB_IETA_MAX );


  char hname[50], htitle[50];
  for ( int iz(0); iz < nEE; iz++ ) {
    // Branches for images
    const char *zside = (iz > 0) ? "p" : "m";
    sprintf(hname, "GsfTracks_EE%s",zside);
    tree->Branch(hname,        &vGsfTracks_EE_[iz]);
    sprintf(hname, "GsfTracksPt_EE%s",zside);
    tree->Branch(hname,        &vGsfTracksPt_EE_[iz]);
    sprintf(hname, "GsfTracksQPt_EE%s",zside);
    tree->Branch(hname,        &vGsfTracksQPt_EE_[iz]);

    // Histograms for monitoring
    sprintf(hname, "GsfTracks_EE%s",zside);
    sprintf(htitle,"N(ix,iy);ix;iy");
    hGsfTracks_EE[iz] = fs->make<TH2F>(hname, htitle,
        EE_MAX_IX, EE_MIN_IX-1, EE_MAX_IX,
        EE_MAX_IY, EE_MIN_IY-1, EE_MAX_IY );
    sprintf(hname, "GsfTracksPt_EE%s",zside);
    sprintf(htitle,"pT(ix,iy);ix;iy");
    hGsfTracksPt_EE[iz] = fs->make<TH2F>(hname, htitle,
        EE_MAX_IX, EE_MIN_IX-1, EE_MAX_IX,
        EE_MAX_IY, EE_MIN_IY-1, EE_MAX_IY );
  } // iz

} // branchesGsfTracksAtEBEE()

// Fill TRK rechits at EB/EE ______________________________________________________________//
void RecHitAnalyzer::fillGsfTracksAtEBEE ( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

  int ix_, iy_, iz_;
  int iphi_, ieta_, idx_; // rows:ieta, cols:iphi
  float eta, phi, pt, qpt, /*dxy, dz,*/ energy;
  math::XYZVector position; //vector for propagating pfC
  GlobalPoint pos;

  vGsfTracks_EB_.assign( EBDetId::kSizeForDenseIndexing, 0. );
  vGsfTracksPt_EB_.assign( EBDetId::kSizeForDenseIndexing, 0. );
  vGsfTracksE_EB_.assign( EBDetId::kSizeForDenseIndexing, 0. );
  vGsfTracksQPt_EB_.assign( EBDetId::kSizeForDenseIndexing, 0. );

  for ( int iz(0); iz < nEE; iz++ ) {
    vGsfTracks_EE_[iz].assign( EE_NC_PER_ZSIDE, 0. );
    vGsfTracksPt_EE_[iz].assign( EE_NC_PER_ZSIDE, 0. );
    vGsfTracksQPt_EE_[iz].assign( EE_NC_PER_ZSIDE, 0. );

  }

  edm::Handle<std::vector<reco::GsfTrack>> tracksH_;
  iEvent.getByToken( gsfTracksCollectionT_, tracksH_ );



  // Provides access to global cell position
  edm::ESHandle<CaloGeometry> caloGeomH_;
  iSetup.get<CaloGeometryRecord>().get( caloGeomH_ );
  const CaloGeometry* caloGeom = caloGeomH_.product();

  edm::Handle<reco::VertexCollection> vertexInfo;
  iEvent.getByToken(vertexCollectionT_, vertexInfo);
  const reco::VertexCollection& vtxs = *vertexInfo;


  edm::ESHandle<MagneticField> magfield;
  iSetup.get<IdealMagneticFieldRecord>().get(magfield);

  for ( std::vector<reco::GsfTrack>::const_iterator iTk = tracksH_->begin();
        iTk != tracksH_->end(); ++iTk ) {
    
    energy = iTk->p(); // energy is roughly the same as p()
    pt    = iTk->pt();
    //dxy = ( !vtxs.empty() ? iTk->dxy(vtxs[0].position()) : iTk->dxy() ); // 1st entry=PV
    //dz = ( !vtxs.empty() ? iTk->dz(vtxs[0].position())  : iTk->dz() );
    qpt   = (iTk->charge()*pt);

    //if (pt<0.5 || iTk->numberOfValidHits()<3 || std::fabs(dz)>0.4 || std::fabs(dxy)>0.1){ // tau reco like selection
    //  continue;
    //}
    // get B field
    double magneticField = (magfield.product() ? magfield.product()->inTesla(GlobalPoint(0., 0., 0.)).z() : 0.0);
    math::XYZTLorentzVector  track_p4(iTk->px(),iTk->py(),iTk->pz(),sqrt(pow(iTk->p(),2)+0.000511*0.000511)); //setup 4-vector assuming mass is mass of charged pion
    BaseParticlePropagator propagator = BaseParticlePropagator(
        RawParticle(track_p4, math::XYZTLorentzVector(iTk->vx(), iTk->vy(), iTk->vz(), 0.),  iTk->charge()),
        0., 0., magneticField);
    propagator.propagateToEcalEntrance(false); // propogate to ECAL entrance
    auto track_position = propagator.particle().vertex().Vect();

    if ( std::fabs(track_position.eta()) > 3. ) continue;

    DetId id( spr::findDetIdECAL( caloGeom, track_position.eta(), track_position.phi(), false ) );
      if ( id.subdetId() == EcalBarrel ) {
        EBDetId ebId( id );
        iphi_ = ebId.iphi() - 1;
        ieta_ = ebId.ieta() > 0 ? ebId.ieta()-1 : ebId.ieta();

        // Fill histograms for monitoring
        hGsfTracks_EB->Fill( iphi_, ieta_ );
        hGsfTracksPt_EB->Fill( iphi_, ieta_, pt );
        idx_ = ebId.hashedIndex(); // (ieta_+EB_IETA_MAX)*EB_IPHI_MAX + iphi_
        // Fill vectors for images
        vGsfTracks_EB_[idx_] += 1.;
        vGsfTracksPt_EB_[idx_] += pt;
        vGsfTracksE_EB_[idx_] += energy;
        vGsfTracksQPt_EB_[idx_] += qpt;
      } else if ( id.subdetId() == EcalEndcap ) {
        EEDetId eeId( id );
        ix_ = eeId.ix() - 1;
        iy_ = eeId.iy() - 1;
        iz_ = (eeId.zside() > 0) ? 1 : 0;
        // Fill histograms for monitoring
        hGsfTracks_EE[iz_]->Fill( ix_, iy_ );
        hGsfTracksPt_EE[iz_]->Fill( ix_, iy_, pt );
        // Create hashed Index: maps from [iy][ix] -> [idx_]
        idx_ = iy_*EE_MAX_IX + ix_;
        // Fill vectors for images
        vGsfTracks_EE_   [iz_][idx_] += 1.;
        vGsfTracksPt_EE_ [iz_][idx_] += pt;
        vGsfTracksQPt_EE_[iz_][idx_] += qpt;
      }
      
    }

} // fillGsfTracksAtEBEE()
