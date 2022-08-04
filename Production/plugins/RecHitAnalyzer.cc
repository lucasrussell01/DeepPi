// -*- C++ -*-
//
// Package:    MLAnalyzer/RecHitAnalyzer
// Class:      RecHitAnalyzer
//
//
// Original Author:  Michael Andrews
//         Created:  Sat, 14 Jan 2017 17:45:54 GMT
//
//

#include "DeepPi/Production/interface/RecHitAnalyzer.h"

//
// constructors and destructor
//
RecHitAnalyzer::RecHitAnalyzer(const edm::ParameterSet& iConfig)
{
  //EBRecHitCollectionT_    = consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("EBRecHitCollection"));
  EBRecHitCollectionT_    = consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("reducedEBRecHitCollection"));
  //EBDigiCollectionT_      = consumes<EBDigiCollection>(iConfig.getParameter<edm::InputTag>("selectedEBDigiCollection"));
  //EBDigiCollectionT_      = consumes<EBDigiCollection>(iConfig.getParameter<edm::InputTag>("EBDigiCollection"));
  EERecHitCollectionT_    = consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("reducedEERecHitCollection"));
  ESRecHitCollectionT_    = consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("reducedESRecHitCollection"));
  //EERecHitCollectionT_    = consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("EERecHitCollection"));
  HBHERecHitCollectionT_  = consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("reducedHBHERecHitCollection"));
  TRKRecHitCollectionT_   = consumes<TrackingRecHitCollection>(iConfig.getParameter<edm::InputTag>("trackRecHitCollection"));

  genParticleCollectionT_ = consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("genParticleCollection"));
  photonCollectionT_      = consumes<reco::PhotonCollection>(iConfig.getParameter<edm::InputTag>("gedPhotonCollection"));
  jetCollectionT_         = consumes<reco::PFJetCollection>(iConfig.getParameter<edm::InputTag>("ak4PFJetCollection"));
  genJetCollectionT_      = consumes<reco::GenJetCollection>(iConfig.getParameter<edm::InputTag>("genJetCollection"));
  trackCollectionT_       = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("trackCollection"));


  pfCollectionT_          = consumes<PFCollection>(iConfig.getParameter<edm::InputTag>("pfCollection"));
  vertexCollectionT_       = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexCollection"));

  recoJetsT_              = consumes<edm::View<reco::Jet> >(iConfig.getParameter<edm::InputTag>("recoJetsForBTagging"));
  jetTagCollectionT_      = consumes<reco::JetTagCollection>(iConfig.getParameter<edm::InputTag>("jetTagCollection"));
  ipTagInfoCollectionT_   = consumes<std::vector<reco::CandIPTagInfo> > (iConfig.getParameter<edm::InputTag>("ipTagInfoCollection"));
  
  PFEBRecHitCollectionT_    = consumes<std::vector<reco::PFRecHit>>(iConfig.getParameter<edm::InputTag>("PFEBRecHitCollection"));
  PFHBHERecHitCollectionT_    = consumes<std::vector<reco::PFRecHit>>(iConfig.getParameter<edm::InputTag>("PFHBHERecHitCollection"));

  //johnda add configuration
  mode_      = iConfig.getParameter<std::string>("mode");
  minJetPt_  = iConfig.getParameter<double>("minJetPt");
  maxJetEta_ = iConfig.getParameter<double>("maxJetEta");
  z0PVCut_   = iConfig.getParameter<double>("z0PVCut");

  std::cout << " >> Mode set to " << mode_ << std::endl;
  if ( mode_ == "JetLevel" ) {
    doJets_ = true;
    nJets_ = iConfig.getParameter<int>("nJets");
    std::cout << "\t>> nJets set to " << nJets_ << std::endl;
  } else if ( mode_ == "EventLevel" ) {
    doJets_ = false;
  } else {
    std::cout << " >> Assuming EventLevel Config. " << std::endl;
    doJets_ = false;
  }



  // Initialize file writer
  // NOTE: initializing dynamic-memory histograms outside of TFileService
  // will cause memory leaks
  usesResource("TFileService");
  edm::Service<TFileService> fs;
  h_sel = fs->make<TH1F>("h_sel", "isSelected;isSelected;Events", 2, 0., 2.);

  //////////// TTree //////////

  // These will be use to create the actual images
  RHTree = fs->make<TTree>("RHTree", "RecHit tree");
  if ( doJets_ ) {
    branchesEvtSel_jet( RHTree, fs );
  } else {
    branchesEvtSel( RHTree, fs );
  }
  branchesEB           ( RHTree, fs );
  branchesEE           ( RHTree, fs );
  branchesES           ( RHTree, fs );
  //branchesESatEE           ( RHTree, fs );
  branchesHBHE         ( RHTree, fs );
  branchesECALatHCAL   ( RHTree, fs );
  branchesECALstitched ( RHTree, fs );
  branchesHCALatEBEE   ( RHTree, fs );
  branchesTracksAtEBEE(RHTree, fs);
  branchesTracksAtECALstitched( RHTree, fs);
  branchesPFCandsAtEBEE(RHTree, fs);
  branchesPFCandsAtECALstitched( RHTree, fs);
  branchesJetInfoAtECALstitched( RHTree, fs);
  branchesPFEB           ( RHTree, fs );


} // constructor
//
RecHitAnalyzer::~RecHitAnalyzer()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}
  

//
// member functions
//
// ------------ method called for each event  ------------
void
RecHitAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  nTotal++;
  using namespace edm;

  // ----- Apply event selection cuts ----- //

  bool passedSelection = false;
  if ( doJets_ ) {
    passedSelection = runEvtSel_jet( iEvent, iSetup );
  } else {
    passedSelection = runEvtSel( iEvent, iSetup );
  }

  if ( !passedSelection ) {
    h_sel->Fill( 0. );;
    return;
  }

  fillEB( iEvent, iSetup );
  fillEE( iEvent, iSetup );
  fillES( iEvent, iSetup );
  //fillESatEE( iEvent, iSetup );
  fillHBHE( iEvent, iSetup );
  fillECALatHCAL( iEvent, iSetup );
  fillECALstitched( iEvent, iSetup );
  fillHCALatEBEE( iEvent, iSetup );
  fillTracksAtEBEE( iEvent, iSetup );
  fillTracksAtECALstitched( iEvent, iSetup );
  fillPFCandsAtEBEE( iEvent, iSetup );
  fillPFCandsAtECALstitched( iEvent, iSetup );
  //fillTRKlayersAtEBEE( iEvent, iSetup );
  //fillTRKlayersAtECAL( iEvent, iSetup );
  //fillTRKvolumeAtEBEE( iEvent, iSetup );
  //fillTRKvolumeAtECAL( iEvent, iSetup );
  fillJetInfoAtECALstitched( iEvent, iSetup );
  fillPFEB( iEvent, iSetup );
  //fillPFHBHE( iEvent, iSetup );

  ////////////// 4-Momenta //////////
  //fillFC( iEvent, iSetup );

  // Fill RHTree
  RHTree->Fill();
  h_sel->Fill( 1. );
  nPassed++;

} // analyze()

// ------------ method called once each job just before starting event loop  ------------
void
RecHitAnalyzer::beginJob(const edm::EventSetup& iSetup)
{
  nTotal = 0;
  nPassed = 0;
}

// ------------ method called once each job just after ending the event loop  ------------
void 
RecHitAnalyzer::endJob() 
{
  std::cout << " selected: " << nPassed << "/" << nTotal << std::endl;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
RecHitAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);

  //Specify that only 'tracks' is allowed
  //To use, remove the default given above and uncomment below
  //ParameterSetDescription desc;
  //desc.addUntracked<edm::InputTag>("tracks","ctfWithMaterialTracks");
  //descriptions.addDefault(desc);
}

const reco::PFCandidate*
RecHitAnalyzer::getPFCand(edm::Handle<PFCollection> pfCands, float eta, float phi, float& minDr, bool debug ) {

  minDr = 10;
  const reco::PFCandidate* minDRCand = nullptr;
  
  for ( PFCollection::const_iterator iPFC = pfCands->begin();
        iPFC != pfCands->end(); ++iPFC ) {

    const reco::Track* thisTrk = iPFC->bestTrack();
    if ( !thisTrk ) continue;

    float thisdR = reco::deltaR( eta, phi, thisTrk->eta(), thisTrk->phi() );
    if (debug) std::cout << "\tthisdR: " << thisdR << " " << thisTrk->pt() << " " << iPFC->particleId() << std::endl;

    const reco::PFCandidate& thisPFCand = (*iPFC);
      
    if ( (thisdR < 0.01) && (thisdR <minDr) ) {
      minDr    = thisdR; 
      minDRCand = &thisPFCand;
    }
  }

  return minDRCand;  
}

const reco::Track*
RecHitAnalyzer::getTrackCand(edm::Handle<reco::TrackCollection> trackCands, float eta, float phi, float& minDr, bool debug ) {

  minDr = 10;
  const reco::Track* minDRCand = nullptr;
  reco::Track::TrackQuality tkQt_ = reco::Track::qualityByName("highPurity");

  for ( reco::TrackCollection::const_iterator iTk = trackCands->begin();
        iTk != trackCands->end(); ++iTk ) {
    if ( !(iTk->quality(tkQt_)) ) continue;  

    float thisdR = reco::deltaR( eta, phi, iTk->eta(),iTk->phi() );
    if (debug) std::cout << "\tthisdR: " << thisdR << " " << iTk->pt() << std::endl;

    const reco::Track& thisTrackCand = (*iTk);
      
    if ( (thisdR < 0.01) && (thisdR <minDr) ) {
      minDr    = thisdR; 
      minDRCand = &thisTrackCand;
    }
  }

  return minDRCand;  
}

int RecHitAnalyzer::getTruthLabel(const reco::PFJetRef& recJet, edm::Handle<reco::GenParticleCollection> genParticles, float dRMatch , bool debug ){
  if ( debug ) {
    std::cout << " Mathcing reco jetPt:" << recJet->pt() << " jetEta:" << recJet->eta() << " jetPhi:" << recJet->phi() << std::endl;
  }

  for (reco::GenParticleCollection::const_iterator iGen = genParticles->begin();
       iGen != genParticles->end();
       ++iGen) {

    // From: (page 7/ Table 1.5.2)
    //https://indico.desy.de/indico/event/7142/session/9/contribution/31/material/slides/6.pdf
    //code range explanation:
    // 11 - 19 beam particles
    // 21 - 29 particles of the hardest subprocess
    // 31 - 39 particles of subsequent subprocesses in multiparton interactions
    // 41 - 49 particles produced by initial-state-showers
    // 51 - 59 particles produced by final-state-showers
    // 61 - 69 particles produced by beam-remnant treatment
    // 71 - 79 partons in preparation of hadronization process
    // 81 - 89 primary hadrons produced by hadronization process
    // 91 - 99 particles produced in decay process, or by Bose-Einstein effects

    // Do not want to match to the final particles in the shower
    if ( iGen->status() > 99 ) continue;
    
    // Only want to match to partons/leptons/bosons
    if ( iGen->pdgId() > 25 ) continue;

    float dR = reco::deltaR( recJet->eta(),recJet->phi(), iGen->eta(),iGen->phi() );

    if ( debug ) std::cout << " \t >> dR " << dR << " id:" << iGen->pdgId() << " status:" << iGen->status() << " nDaught:" << iGen->numberOfDaughters() << " pt:"<< iGen->pt() << " eta:" <<iGen->eta() << " phi:" <<iGen->phi() << " nMoms:" <<iGen->numberOfMothers()<< std::endl;

    if ( dR > dRMatch ) continue; 
    if ( debug ) std::cout << " Matched pdgID " << iGen->pdgId() << std::endl;

    return iGen->pdgId();

  } // gen particles 





  return -99;
}

std::vector<reco::GenTau *> BuildTauJets(edm::Handle<reco::GenParticleCollection> genParticles, double magneticField, bool include_leptonic, bool use_prompt) {
  // Warning: returned tau type works for taus decayed by Pythia8 but might not work for other generators e.g tauola!
  std::vector<reco::GenTau *> taus;
  for (reco::GenParticleCollection::const_iterator iGen = genParticles->begin();
       iGen != genParticles->end();
       ++iGen) {
    bool is_prompt=true; 
    bool is_last_copy=iGen->statusFlags().isLastCopy();
    if(use_prompt){
      is_prompt=iGen->statusFlags().isPrompt();
    }
    if (abs(iGen->pdgId()) == 15 && is_prompt && is_last_copy) {
      bool has_tau_daughter = false;
      bool has_lepton_daughter = false;
      unsigned count_pi0 =  0, count_pi = 0, count_tot = 0, count_k = 0, count_rho = 0;
      math::XYZPoint vtx = math::XYZPoint(0, 0, 0);
      bool foundVertex = false;
      math::XYZTLorentzVector nuvec;
      math::XYZTLorentzVector charge_vec;
      math::XYZTLorentzVector neutral_vec;
      math::XYZTLorentzVector lead_pi0_vec;
      std::vector<std::pair<math::XYZVector,double>> propogated_pis = {};
      std::vector<std::pair<math::XYZVector, double>> propogated_pi0s = {};
      // Add vector of charged
      std::vector<math::XYZTLorentzVector> charge_vec_all;
      std::vector<math::XYZTLorentzVector> neutral_vec_all;
      for (const reco::GenParticleRef& daughter : iGen->daughterRefVector()) {
        if (abs(daughter->pdgId()) == 15) has_tau_daughter = true;
        if (abs(daughter->pdgId()) == 11 || abs(daughter->pdgId()) == 13) has_lepton_daughter = true;
        if (!foundVertex) {
          // take tau vertex from one of the decay products since this will be displaced from the PV
          vtx = daughter->vertex();
          foundVertex = true; 
        }
        unsigned pdgId = abs(daughter->pdgId());
        if(pdgId == 111) {
          count_pi0++;
          neutral_vec+=daughter->p4(); 
          // std::cout << "Gen-Info pi0 energy: " <<  daughter->p4().energy() << "Mother ID " << daughter->mother()->pdgId() << std::endl;
          std::pair<math::XYZVector, double> neutral_prop = std::make_pair(ExtrapolateToECAL(daughter, magneticField), daughter->p());
          neutral_vec_all.push_back(daughter->p4());
          propogated_pi0s.push_back(neutral_prop);
          if(daughter->pt() > lead_pi0_vec.pt()) lead_pi0_vec = daughter->p4();
        } 
        if(pdgId == 211) count_pi++;
        if(pdgId == 213) count_rho++;
        if(pdgId == 321) count_k++;
        if(daughter->charge()!=0) {
          charge_vec+=daughter->p4(); 
          charge_vec_all.push_back(daughter->p4());
          std::pair<math::XYZVector, double> charge_prop = std::make_pair(ExtrapolateToECAL(daughter, magneticField), daughter->p());
          propogated_pis.push_back(charge_prop);
        }
        if(pdgId!=12&&pdgId!=14&&pdgId!=16) {
          count_tot++;
        } else {
          nuvec+=daughter->p4();
        }
      }
      if (has_tau_daughter) continue;
      if (has_lepton_daughter && !include_leptonic) continue;
      int tauFlag = -1;
      if(count_tot==1 && count_pi==1 && count_pi0==0) tauFlag=0;
      if(count_tot==1 && count_pi==0 && count_k==1 && count_pi0==0) tauFlag=20;
      if(count_tot==2 && count_pi==1 && count_pi0==1) tauFlag=1; 
      if(count_tot==2 && count_rho==1) tauFlag=21; 
      if(count_tot==3 && count_pi==1 && count_pi0==2) tauFlag=2;
      if(count_tot==3 && count_pi==3 && count_pi0==0) tauFlag=10;
      if(count_tot==4 && count_pi==3 && count_pi0==1) tauFlag=11;
      reco::GenTau *tau = new reco::GenTau(iGen->charge(), iGen->p4(), vtx, iGen->pdgId(), iGen->status(), true);   
      tau->set_decay_mode(tauFlag);
      tau->set_charge_p4(charge_vec); 
      tau->set_neutral_p4(neutral_vec);
      tau->set_lead_pi0_p4(lead_pi0_vec); 
      tau->set_nu_p4(nuvec);
      tau->set_charge_p4_indv(charge_vec_all);
      tau->set_neutral_p4_indv(neutral_vec_all);
      tau->set_pis_at_ecal(propogated_pis);
      tau->set_pi0s_at_ecal(propogated_pi0s);
      taus.push_back(tau);
    }
  }
  return taus;
}

std::vector<reco::GenTau *> BuildTauLikeJets(edm::Handle<reco::GenJetCollection> genJets, double magneticField) {

  std::vector<reco::GenTau *> taus;
  for (reco::GenJetCollection::const_iterator iJet = genJets->begin();
       iJet != genJets->end();
       ++iJet) {

    std::vector<const reco::GenParticle*> parts = iJet->getGenConstituents();
    math::XYZTLorentzVector charge_vec;
    math::XYZTLorentzVector neutral_vec;
    math::XYZTLorentzVector lead_pi0_vec;
    std::vector<const reco::GenParticle*> pis = {};

    unsigned Npi=0;
    unsigned Ngammas=0;
    std::set<reco::GenParticleRef> mother_refs = {};
    //std::set<int> mother_refs = {};
 
    for(auto x : parts) {
      int pdgId = abs(x->pdgId());
      if(pdgId == 22) {
        Ngammas++;
        auto mothers = x->motherRefVector();
        for (auto m : mothers){
          //if(abs(m->pdgId())==111) mother_refs.insert(m.key());
          if(abs(m->pdgId())==111 && m->pt()>1.) mother_refs.insert(m);
        }
        
        neutral_vec+=x->p4();
        if(x->pt() > lead_pi0_vec.pt()) lead_pi0_vec = x->p4();
      }
      if(pdgId == 211) {
        Npi++;
        if(x->pt()>1.) pis.push_back(x);
      }
    }

    std::vector<math::XYZTLorentzVector> neutral = {};
    std::vector<std::pair<math::XYZVector, double>> propogated_pi0s = {};
    for(auto y : mother_refs) {
      double dR_pi0 = std::fabs(ROOT::Math::VectorUtil::DeltaR(y->p4(),iJet->p4()));
      if(dR_pi0>0.1) continue; //tau-like jets must be narrow
      neutral.push_back(y->p4());
      std::pair<math::XYZVector, double> neutral_prop = std::make_pair(ExtrapolateToECAL(y, magneticField), y->p());
      propogated_pi0s.push_back(neutral_prop);
    }
    math::XYZTLorentzVector tot_neutral(0.,0.,0.,0.);
    for(auto n : neutral) tot_neutral+=n; 
    math::XYZTLorentzVector lead_pi0_p4(0.,0.,0.,0.);
    for(auto n : neutral) {
       if(n.Pt()>lead_pi0_p4.Pt()) lead_pi0_p4=n;
     }

    std::vector<reco::GenTau*> tau_cands = {};

    // try 1-prong combinations
    
    for (auto x : pis) {
      bool realTau = false;
      double dR_pi = std::fabs(ROOT::Math::VectorUtil::DeltaR(x->p4(),iJet->p4()));
      if(dR_pi>0.1) continue; //tau-like jets must be narrow
      reco::GenTau *t = new reco::GenTau();
      std::vector<math::XYZTLorentzVector> charge = {x->p4()};
      if (x->motherRefVector().size()>0 && std::abs(x->motherRefVector()[0]->pdgId())==15) realTau=true;
      t->set_charge_p4_indv(charge);
      t->set_charge_p4(x->p4());
      t->set_neutral_p4_indv(neutral);
      t->set_neutral_p4(tot_neutral);
      t->set_decay_mode(std::min((int)neutral.size(),9));
      t->set_lead_pi0_p4(lead_pi0_p4);
      t->setP4(tot_neutral+x->p4());
      // require isolation-like selection to reject jets that aren't tau-like
      if ((iJet->pt()-t->vis_p4().Pt())/t->vis_p4().Pt() > 0.2) continue;
      if(realTau) continue; // veto real taus
      t->setPdgId(6);
      std::vector<std::pair<math::XYZVector,double>> propogated_pis = {};
      std::pair<math::XYZVector, double> charge_prop = std::make_pair(ExtrapolateToECAL(x, magneticField), x->p());
      propogated_pis.push_back(charge_prop);
      t->set_pis_at_ecal(propogated_pis);
      t->set_pi0s_at_ecal(propogated_pi0s);
      tau_cands.push_back(t);
    } 
    
    // might want to add function for 3-prong taus eventually

    // take highest pT tau candidate
    if(tau_cands.size()==0) continue;
    double lead_pt=0.;
    reco::GenTau *lead_tau;
    for(auto t : tau_cands) {
      if(t->vis_p4().Pt()>lead_pt) {
        lead_tau = t;
        lead_pt = t->vis_p4().Pt();
      }
    }
    taus.push_back(lead_tau); 
  }

  return taus;
}


std::pair<int, reco::GenTau*> RecHitAnalyzer::getTruthLabelForTauJets(const reco::PFJetRef& recJet, edm::Handle<reco::GenParticleCollection> genParticles, edm::Handle<reco::GenJetCollection> genJets, double magneticField, float dRMatch , bool debug ){
  if ( debug ) {
    std::cout << " Matching reco jetPt:" << recJet->pt() << " jetEta:" << recJet->eta() << " jetPhi:" << recJet->phi() << std::endl;
  }
  std::vector<reco::GenTau *> gen_taus = BuildTauJets(genParticles, magneticField, false, true);
  std::vector<reco::GenTau *> gen_taus_like_jets = BuildTauLikeJets(genJets, magneticField);
  std::vector<const reco::GenParticle *> gen_leptons;
  for (reco::GenParticleCollection::const_iterator iGen = genParticles->begin();
       iGen != genParticles->end();
       ++iGen) {
         if (((abs(iGen->pdgId()) == 11 )||(abs(iGen->pdgId()) == 13)) && iGen->pt() > 8. && (iGen->statusFlags().isPrompt() || iGen->statusFlags().isDirectPromptTauDecayProduct())) gen_leptons.push_back(&(*iGen));
  }


  reco::GenTau *gen_tau = new reco::GenTau(); 
  float minDR=-1.;
  int match_pdgId=0;

  // match to light leptons
  for (auto part : gen_leptons) {
 
    float dR = reco::deltaR( recJet->eta(),recJet->phi(), part->eta(),part->phi() );

    if ( debug ) std::cout << " \t >> dR " << dR << " id:" << part->pdgId() << " status:" << part->status() << " nDaught:" << part->numberOfDaughters() << " pt:"<< part->pt() << " eta:" << part->eta() << " phi:" << part->phi() << " nMoms:" << part->numberOfMothers()<< std::endl;

    if ( dR > dRMatch ) continue;
    if(minDR<0 || dR<minDR) {
      minDR = dR;
      match_pdgId = part->pdgId();
      if ( debug ) std::cout << " Matched pdgID " << part->pdgId() << std::endl;
    }
  }

  // match to taus
  for (auto part : gen_taus) {

    float dR = reco::deltaR( recJet->eta(),recJet->phi(), part->vis_p4().eta(),part->vis_p4().phi() );

    if ( debug ) std::cout << " \t >> dR " << dR << " id:" << part->pdgId() << " status:" << part->status() << " nDaught:" << part->numberOfDaughters() << " pt:"<< part->vis_p4().pt() << " eta:" << part->vis_p4().eta() << " phi:" << part->vis_p4().phi() << " nMoms:" << part->numberOfMothers()<< std::endl;

    if ( dR > dRMatch ) continue;
    if(minDR<0 || dR<minDR) {
      minDR = dR;
      match_pdgId = part->pdgId();
      gen_tau = part;
      if ( debug ) std::cout << " Matched pdgID " << part->pdgId() << std::endl;
    }
  }

  if(minDR>=0) return std::make_pair(match_pdgId, gen_tau);
  else {
    // when we have a jet try to match to a tau-like jet
    for (auto part : gen_taus_like_jets) {

      float dR = reco::deltaR( recJet->eta(),recJet->phi(), part->vis_p4().eta(),part->vis_p4().phi() );

      if ( debug ) std::cout << " \t >> dR " << dR << " id:" << part->pdgId() << " status:" << part->status() << " nDaught:" << part->numberOfDaughters() << " pt:"<< part->vis_p4().pt() << " eta:" << part->vis_p4().eta() << " phi:" << part->vis_p4().phi() << " nMoms:" << part->numberOfMothers()<< std::endl;

      if ( dR > dRMatch ) continue;
      if(minDR<0 || dR<minDR) {
        minDR = dR;
        gen_tau = part;
        if ( debug ) std::cout << " Matched pdgID " << part->pdgId() << std::endl;
      }
    }
    return std::make_pair(6, gen_tau);
  }
}


float RecHitAnalyzer::getBTaggingValue(const reco::PFJetRef& recJet, edm::Handle<edm::View<reco::Jet> >& recoJetCollection, edm::Handle<reco::JetTagCollection>& btagCollection, float dRMatch, bool debug ){

  // loop over jets
  for( edm::View<reco::Jet>::const_iterator jetToMatch = recoJetCollection->begin(); jetToMatch != recoJetCollection->end(); ++jetToMatch )
    {
      reco::Jet thisJet = *jetToMatch;
      float dR = reco::deltaR( recJet->eta(),recJet->phi(), thisJet.eta(),thisJet.phi() );
      if(dR > 0.1) continue;

      size_t idx = (jetToMatch - recoJetCollection->begin());
      edm::RefToBase<reco::Jet> jetRef = recoJetCollection->refAt(idx);

      if(debug) std::cout << "btag discriminator value = " << (*btagCollection)[jetRef] << std::endl;
      return (*btagCollection)[jetRef];
  
    }

  if(debug){
    std::cout << "ERROR  No btag match: " << std::endl;
    
    // loop over jets
    for( edm::View<reco::Jet>::const_iterator jetToMatch = recoJetCollection->begin(); jetToMatch != recoJetCollection->end(); ++jetToMatch )
      {
	const reco::Jet thisJet = *jetToMatch;
	std::cout << "\t Match attempt pt: " <<  thisJet.pt() << " vs " <<  recJet->pt()
		  << " eta: " << thisJet.eta() << " vs " << recJet->eta()
		  << "phi: "<< thisJet.phi() << " vs " << recJet->phi()
		  << std::endl;
	float dR = reco::deltaR( recJet->eta(),recJet->phi(), thisJet.eta(),thisJet.phi() );
	std::cout << "dR " << dR << std::endl;
      }
  }    

  return -99;
}

math::XYZVector RecHitAnalyzer::GetPi0Direction(math::XYZPoint vertex, double releta, double relphi, double seedeta, double seedphi) {

    double eta_orig = releta+seedeta;
    double phi_orig = relphi+seedphi;
    phi_orig = TVector2::Phi_mpi_pi(phi_orig);

    BaseParticlePropagator propagator = BaseParticlePropagator(
    RawParticle((math::XYZTLorentzVector)math::PtEtaPhiELorentzVector(1.,eta_orig, phi_orig,1.), math::XYZTLorentzVector(0., 0., 0., 0.),
                0.),0.,0.,0.);

    propagator.propagateToEcalEntrance(false); // propogate to ECAL entrance
    math::XYZVector pos = propagator.particle().vertex().Vect();

    math::XYZVector vertex_vec(vertex.x(), vertex.y(), vertex.z());
    math::XYZVector direction = pos - vertex_vec;
 
    return direction;
}

//define this as a plug-in
DEFINE_FWK_MODULE(RecHitAnalyzer);
