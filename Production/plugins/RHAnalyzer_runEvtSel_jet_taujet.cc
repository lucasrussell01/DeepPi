#include "DeepPi/Production/interface/RecHitAnalyzer.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "CommonTools/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "CommonTools/BaseParticlePropagator/interface/RawParticle.h"
#include "TVector2.h"

using std::vector;
using std::cout;
using std::endl;

TH1D *h_taujet_jet_pT;
TH1D *h_taujet_jet_E;
TH1D *h_taujet_jet_eta;
TH1D *h_taujet_jet_m0;
TH1D *h_taujet_jet_nJet;
vector<float> vTaujet_jet_pT_;
vector<float> vTaujet_jet_m0_;
vector<float> vTaujet_jet_eta_;
vector<float> vTaujet_jet_phi_;
vector<float> vTaujet_jet_truthLabel_;
vector<float> vTaujet_jet_truthDM_;
vector<float> vTaujet_jet_neutral_pT_;
vector<float> vTaujet_jet_neutral_m0_;
vector<float> vTaujet_jet_neutral_eta_;
vector<float> vTaujet_jet_neutral_phi_;
vector<vector<float>> vTaujet_jet_charged_indv_p_;
vector<vector<float>> vTaujet_jet_neutral_indv_p_;
vector<vector<float>> vTaujet_jet_charged_indv_eta_;
vector<vector<float>> vTaujet_jet_neutral_indv_eta_;
vector<vector<float>> vTaujet_jet_charged_indv_phi_;
vector<vector<float>> vTaujet_jet_neutral_indv_phi_;

vector<vector<double>> vTaujet_jet_charged_indv_relp_;
vector<vector<double>> vTaujet_jet_neutral_indv_relp_;
vector<vector<double>> vTaujet_jet_charged_indv_releta_;
vector<vector<double>> vTaujet_jet_neutral_indv_releta_;
vector<vector<double>> vTaujet_jet_charged_indv_relphi_;
vector<vector<double>> vTaujet_jet_neutral_indv_relphi_;

vector<vector<float>> vTaujet_jet_charged_indv_releta_crystal_;
vector<vector<float>> vTaujet_jet_neutral_indv_releta_crystal_;
vector<vector<float>> vTaujet_jet_charged_indv_relphi_crystal_;
vector<vector<float>> vTaujet_jet_neutral_indv_relphi_crystal_;

vector<float> vTaujet_jet_neutral_relphi_crystal_;
vector<float> vTaujet_jet_neutral_releta_crystal_;
vector<float> vTaujet_jet_neutral_relphi_;
vector<float> vTaujet_jet_neutral_releta_;
vector<float> vTaujet_jet_neutral_relp_;
vector<float> vTaujet_jet_neutral_relmass_;

// for centering:
vector<float> vTaujet_jet_leading_eta_;
vector<float> vTaujet_jet_leading_phi_;
vector<float> vTaujet_jet_leading_ieta_;
vector<float> vTaujet_jet_leading_iphi_;
vector<float> vTaujet_jet_leading_energy_;
vector<float> vTaujet_jet_neutralsum_eta_;
vector<float> vTaujet_jet_neutralsum_phi_;
vector<float> vTaujet_jet_neutralsum_ieta_;
vector<float> vTaujet_jet_neutralsum_iphi_;
vector<float> vTaujet_jet_neutralsum_pt_;
vector<float> vTaujet_jet_neutralsum_ECAL_;
// sum all pf
vector<float> vTaujet_jet_centre_ieta_;
vector<float> vTaujet_jet_centre_iphi_;
vector<float> vTaujet_jet_centre1_ieta_;
vector<float> vTaujet_jet_centre1_iphi_;
vector<float> vTaujet_jet_centre2_ieta_;
vector<float> vTaujet_jet_centre2_iphi_;

vector<double> vTaujet_jet_centre2_eta_;
vector<double> vTaujet_jet_centre2_phi_;

// Initialize branches _____________________________________________________//
void RecHitAnalyzer::branchesEvtSel_jet_taujet( TTree* tree, edm::Service<TFileService> &fs ) {

  h_taujet_jet_pT    = fs->make<TH1D>("h_jet_pT"  , "p_{T};p_{T};Particles", 100,  0., 500.);
  h_taujet_jet_E     = fs->make<TH1D>("h_jet_E"   , "E;E;Particles"        , 100,  0., 800.);
  h_taujet_jet_eta   = fs->make<TH1D>("h_jet_eta" , "#eta;#eta;Particles"  , 100, -5., 5.);
  h_taujet_jet_nJet  = fs->make<TH1D>("h_jet_nJet", "nJet;nJet;Events"     ,  10,  0., 10.);
  h_taujet_jet_m0    = fs->make<TH1D>("h_jet_m0"  , "m0;m0;Events"         , 100,  0., 100.);

  tree->Branch("jetPt",          &vTaujet_jet_pT_);
  tree->Branch("jetM",           &vTaujet_jet_m0_);
  tree->Branch("jetEta",         &vTaujet_jet_eta_);
  tree->Branch("jetPhi",         &vTaujet_jet_phi_);
  tree->Branch("jet_truthLabel", &vTaujet_jet_truthLabel_);

  tree->Branch("jet_truthDM", &vTaujet_jet_truthDM_);
  tree->Branch("neutralPt", &vTaujet_jet_neutral_pT_);
  tree->Branch("neutralM", &vTaujet_jet_neutral_m0_);
  tree->Branch("neutralEta", &vTaujet_jet_neutral_eta_);
  tree->Branch("neutralPhi", &vTaujet_jet_neutral_phi_);
  tree->Branch("jet_charged_indv_p", &vTaujet_jet_charged_indv_p_);
  tree->Branch("jet_neutral_indv_p", &vTaujet_jet_neutral_indv_p_);
  tree->Branch("jet_charged_indv_ieta", &vTaujet_jet_charged_indv_eta_);
  tree->Branch("jet_neutral_indv_ieta", &vTaujet_jet_neutral_indv_eta_);
  tree->Branch("jet_charged_indv_iphi", &vTaujet_jet_charged_indv_phi_);
  tree->Branch("jet_neutral_indv_iphi", &vTaujet_jet_neutral_indv_phi_);

  tree->Branch("jet_charged_indv_relp", &vTaujet_jet_charged_indv_relp_);
  tree->Branch("jet_neutral_indv_relp", &vTaujet_jet_neutral_indv_relp_);
  tree->Branch("jet_charged_indv_releta", &vTaujet_jet_charged_indv_releta_);
  tree->Branch("jet_neutral_indv_releta", &vTaujet_jet_neutral_indv_releta_);
  tree->Branch("jet_charged_indv_relphi", &vTaujet_jet_charged_indv_relphi_);
  tree->Branch("jet_neutral_indv_relphi", &vTaujet_jet_neutral_indv_relphi_);

  tree->Branch("jet_charged_indv_releta_crystal", &vTaujet_jet_charged_indv_releta_crystal_);
  tree->Branch("jet_neutral_indv_releta_crystal", &vTaujet_jet_neutral_indv_releta_crystal_);
  tree->Branch("jet_charged_indv_relphi_crystal", &vTaujet_jet_charged_indv_relphi_crystal_);
  tree->Branch("jet_neutral_indv_relphi_crystal", &vTaujet_jet_neutral_indv_relphi_crystal_);

  tree->Branch("jet_neutral_relphi_crystal", &vTaujet_jet_neutral_relphi_crystal_);
  tree->Branch("jet_neutral_releta_crystal", &vTaujet_jet_neutral_releta_crystal_);
  tree->Branch("jet_neutral_relphi", &vTaujet_jet_neutral_relphi_);
  tree->Branch("jet_neutral_releta", &vTaujet_jet_neutral_releta_);
  tree->Branch("jet_neutral_relp", &vTaujet_jet_neutral_relp_);
  tree->Branch("jet_neutral_relmass", &vTaujet_jet_neutral_relmass_);


  tree->Branch("leading_eta", &vTaujet_jet_leading_eta_);
  tree->Branch("leading_phi", &vTaujet_jet_leading_phi_);
  tree->Branch("leading_ieta", &vTaujet_jet_leading_ieta_);
  tree->Branch("leading_iphi", &vTaujet_jet_leading_iphi_);
  tree->Branch("leading_energy", &vTaujet_jet_leading_energy_);
  tree->Branch("neutralsum_eta", &vTaujet_jet_neutralsum_eta_);
  tree->Branch("neutralsum_phi", &vTaujet_jet_neutralsum_phi_);
  tree->Branch("neutralsum_ieta", &vTaujet_jet_neutralsum_ieta_);
  tree->Branch("neutralsum_iphi", &vTaujet_jet_neutralsum_iphi_);
  tree->Branch("neutralsum_pt", &vTaujet_jet_neutralsum_pt_);
  tree->Branch("neutralsum_ECAL", &vTaujet_jet_neutralsum_ECAL_);

  tree->Branch("jet_centre_ieta", &vTaujet_jet_centre_ieta_);
  tree->Branch("jet_centre_iphi", &vTaujet_jet_centre_iphi_);
  tree->Branch("jet_centre1_ieta", &vTaujet_jet_centre1_ieta_);
  tree->Branch("jet_centre1_iphi", &vTaujet_jet_centre1_iphi_);
  tree->Branch("jet_centre2_ieta", &vTaujet_jet_centre2_ieta_);
  tree->Branch("jet_centre2_iphi", &vTaujet_jet_centre2_iphi_);

  tree->Branch("jet_centre2_eta", &vTaujet_jet_centre2_eta_);
  tree->Branch("jet_centre2_phi", &vTaujet_jet_centre2_phi_);

} // branchesEvtSel_jet_taujet()

// Run jet selection _____________________________________________________//
bool RecHitAnalyzer::runEvtSel_jet_taujet( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{

  edm::Handle<reco::PFJetCollection> jets;
  iEvent.getByToken(jetCollectionT_, jets);

  vJetIdxs.clear();
  vTaujet_jet_pT_.clear();
  vTaujet_jet_m0_.clear();
  vTaujet_jet_eta_.clear();
  vTaujet_jet_phi_.clear();
  vTaujet_jet_truthLabel_.clear();
  vTaujet_jet_truthDM_.clear();
  vTaujet_jet_neutral_pT_.clear();
  vTaujet_jet_neutral_m0_.clear();
  vTaujet_jet_neutral_eta_.clear();
  vTaujet_jet_neutral_phi_.clear();
  vTaujet_jet_charged_indv_p_.clear();
  vTaujet_jet_neutral_indv_p_.clear();
  vTaujet_jet_charged_indv_eta_.clear();
  vTaujet_jet_neutral_indv_eta_.clear();
  vTaujet_jet_charged_indv_phi_.clear();
  vTaujet_jet_neutral_indv_phi_.clear();
  vTaujet_jet_leading_phi_.clear();
  vTaujet_jet_leading_eta_.clear();
  vTaujet_jet_leading_iphi_.clear();
  vTaujet_jet_leading_ieta_.clear();
  vTaujet_jet_leading_energy_.clear();
  vTaujet_jet_neutralsum_phi_.clear();
  vTaujet_jet_neutralsum_eta_.clear();
  vTaujet_jet_neutralsum_iphi_.clear();
  vTaujet_jet_neutralsum_ieta_.clear();
  vTaujet_jet_neutralsum_pt_.clear();
  vTaujet_jet_neutralsum_ECAL_.clear();
  vTaujet_jet_centre_ieta_.clear();
  vTaujet_jet_centre_iphi_.clear();
  vTaujet_jet_centre1_ieta_.clear();
  vTaujet_jet_centre1_iphi_.clear();
  vTaujet_jet_centre2_ieta_.clear();
  vTaujet_jet_centre2_iphi_.clear();
  vTaujet_jet_centre2_eta_.clear();
  vTaujet_jet_centre2_phi_.clear();
  vTaujet_jet_charged_indv_relp_.clear();
  vTaujet_jet_neutral_indv_relp_.clear();
  vTaujet_jet_charged_indv_releta_.clear();
  vTaujet_jet_neutral_indv_releta_.clear();
  vTaujet_jet_charged_indv_relphi_.clear();
  vTaujet_jet_neutral_indv_relphi_.clear();
  vTaujet_jet_charged_indv_releta_crystal_.clear();
  vTaujet_jet_neutral_indv_releta_crystal_.clear();
  vTaujet_jet_charged_indv_relphi_crystal_.clear();
  vTaujet_jet_neutral_indv_relphi_crystal_.clear();
  vTaujet_jet_neutral_relphi_crystal_.clear();
  vTaujet_jet_neutral_releta_crystal_.clear();
  vTaujet_jet_neutral_relphi_.clear();
  vTaujet_jet_neutral_releta_.clear();
  vTaujet_jet_neutral_relp_.clear();
  vTaujet_jet_neutral_relmass_.clear();

  int nJet = 0;
  // Loop over jets
  for ( unsigned iJ(0); iJ != jets->size(); ++iJ ) {

    reco::PFJetRef iJet( jets, iJ );
    if ( std::abs(iJet->pt()) < minJetPt_ ) continue;
    if ( std::abs(iJet->eta()) > maxJetEta_) continue;
    if ( debug ) std::cout << " >> jet[" << iJ << "]Pt:" << iJet->pt() << " jetE:" << iJet->energy() << " jetM:" << iJet->mass() << std::endl;

    vJetIdxs.push_back(iJ);

    nJet++;
    if ( (nJets_ > 0) && (nJet >= nJets_) ) break;

  } // jets


  if ( debug ) {
    for(int thisJetIdx : vJetIdxs)
      std::cout << " >> vJetIdxs:" << thisJetIdx << std::endl;
  }

  if ( (nJets_ > 0) && (nJet != nJets_) ){
    if ( debug ) std::cout << " Fail jet multiplicity:  " << nJet << " != " << nJets_ << std::endl;
    return false;
  }

  if ( vJetIdxs.size() == 0){
    if ( debug ) std::cout << " No passing jets...  " << std::endl;
    return false;
  }

  if ( debug ) std::cout << " >> has_jet_taujet: passed" << std::endl;
  return true;

} // runEvtSel_jet_taujet() 

// Fill branches and histograms _____________________________________________________//
void RecHitAnalyzer::fillEvtSel_jet_taujet( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {

  edm::Handle<reco::PFJetCollection> jets;
  iEvent.getByToken(jetCollectionT_, jets);

  edm::Handle<reco::GenParticleCollection> genParticles;
  iEvent.getByToken( genParticleCollectionT_, genParticles );
  
  edm::Handle<reco::GenJetCollection> genJets;
  iEvent.getByToken( genJetCollectionT_, genJets );

  edm::Handle<edm::View<reco::Jet> > recoJetCollection;
  iEvent.getByToken(recoJetsT_, recoJetCollection);

  edm::Handle<reco::JetTagCollection> btagDiscriminators;
  iEvent.getByToken(jetTagCollectionT_, btagDiscriminators);

  edm::Handle<std::vector<reco::CandIPTagInfo> > ipTagInfo;
  iEvent.getByToken(ipTagInfoCollectionT_, ipTagInfo);

  edm::Handle<reco::VertexCollection> vertexInfo;
  iEvent.getByToken(vertexCollectionT_, vertexInfo);
  const reco::VertexCollection& vtxs = *vertexInfo;
	      
  edm::ESHandle<MagneticField> magfield;
  iSetup.get<IdealMagneticFieldRecord>().get(magfield);

  // Provides access to global cell position
  edm::ESHandle<CaloGeometry> caloGeomH_;
  iSetup.get<CaloGeometryRecord>().get( caloGeomH_ );
  const CaloGeometry* caloGeom = caloGeomH_.product();

  h_taujet_jet_nJet->Fill( vJetIdxs.size() );
  // Fill branches and histograms 
  for(int thisJetIdx : vJetIdxs){
    reco::PFJetRef thisJet( jets, thisJetIdx );
    if ( debug ) std::cout << " >> Jet[" << thisJetIdx << "] Pt:" << thisJet->pt() << std::endl;

    if (debug) std::cout << " Passed Jet " << " Pt:" << thisJet->pt()  << " Eta:" << thisJet->eta()  << " Phi:" << thisJet->phi() 
			 << " jetE:" << thisJet->energy() << " jetM:" << thisJet->mass() 
			 << " photonE:" << thisJet->photonEnergy()  
			 << " chargedHadronEnergy:" << thisJet->chargedHadronEnergy()  
			 << " neutralHadronEnergy :" << thisJet->neutralHadronEnergy()
			 << " electronEnergy	 :" << thisJet->electronEnergy	()
			 << " muonEnergy		 :" << thisJet->muonEnergy		()
			 << " HFHadronEnergy	 :" << thisJet->HFHadronEnergy	()
			 << " HFEMEnergy		 :" << thisJet->HFEMEnergy		()
			 << " chargedEmEnergy	 :" << thisJet->chargedEmEnergy	()
			 << " chargedMuEnergy	 :" << thisJet->chargedMuEnergy	()
			 << " neutralEmEnergy	 :" << thisJet->neutralEmEnergy	()
			 << std::endl;

    h_taujet_jet_pT->Fill( std::abs(thisJet->pt()) );
    h_taujet_jet_E->Fill( thisJet->energy() );
    h_taujet_jet_m0->Fill( thisJet->mass() );
    h_taujet_jet_eta->Fill( thisJet->eta() );
    vTaujet_jet_pT_.push_back( std::abs(thisJet->pt()) );
    vTaujet_jet_m0_.push_back( thisJet->mass() );
    vTaujet_jet_eta_.push_back( thisJet->eta() );
    vTaujet_jet_phi_.push_back( thisJet->phi() );

    std::vector<reco::PFCandidatePtr> pfCands = thisJet->getPFConstituents();

    math::XYZTLorentzVector neutral_PF;

    math::XYZTLorentzVector p4_leading = math::XYZTLorentzVector(0,0,0,0); 
    reco::PFCandidatePtr leading_pfC;


    float neutral_ECAL = 0; // store ECAL energy deposits

    double total_energy = 0; // total energy of event
    double eta_sum = 0; //sum of E_i*eta_i for all i
    double phi_sum = 0;
    double total_energy1 = 0; // total energy of event
    double eta_sum1 = 0; //sum of E_i*eta_i for all i
    double phi_sum1 = 0;
    double total_energy2 = 0; // total energy of event
    double eta_sum2 = 0; //sum of E_i*eta_i for all i
    double phi_sum2 = 0;

    for (const auto &pfC : pfCands){
      
      
      // Loop over all PF candidates and make energy weighted average position
      
      // propagate all particles 
      double magneticField = (magfield.product() ? magfield.product()->inTesla(GlobalPoint(0., 0., 0.)).z() : 0.0);
      math::XYZTLorentzVector  prop_p4(pfC->p4().px(),pfC->p4().py(),pfC->p4().pz(),sqrt(pow(pfC->p(),2)+pfC->mass()*pfC->mass())); //setup 4-vector 
      BaseParticlePropagator propagator = BaseParticlePropagator(
          RawParticle(prop_p4, math::XYZTLorentzVector(pfC->vx(), pfC->vy(), pfC->vz(), 0.),
                      pfC->charge()),0.,0.,magneticField);
      propagator.propagateToEcalEntrance(false); // propogate to ECAL entrance
      auto pfC_position = propagator.particle().vertex().Vect();
      
      eta_sum += pfC_position.eta()*pfC->energy();
      phi_sum += pfC_position.phi()*pfC->energy();
      total_energy += pfC->energy();

      if (pfC->particleId() == 1 || pfC->particleId() ==4){
        // Store gamma and hPM for position
        eta_sum1 += pfC_position.eta()*pfC->energy();
        phi_sum1 += pfC_position.phi()*pfC->energy();
        total_energy1 += pfC->energy();
      }
      if (pfC->particleId() == 4 || pfC->particleId()==2){
        // Store gamma and e for position
        eta_sum2 += pfC_position.eta()*pfC->energy();
        phi_sum2 += pfC_position.phi()*pfC->energy();
        total_energy2 += pfC->energy();
      }

      if (pfC->particleId()==4){
        
        auto n_p4_vec = pfC->p4();
        neutral_PF += n_p4_vec;
        neutral_ECAL += pfC->ecalEnergy(); // going with corrected energy for now

      } else if (pfC->particleId()==1){
        
        if (pfC->p4().pt() > p4_leading.pt()){
          // store leading hadron propagated
          math::XYZTLorentzVector propagated_p4(pfC->p4().pt(), pfC_position.eta(), pfC_position.phi(), pfC->p4().mass());
          p4_leading = propagated_p4;
          leading_pfC = pfC;
        }
      } else {

        }
        } 

    
    double magneticField = (magfield.product() ? magfield.product()->inTesla(GlobalPoint(0., 0., 0.)).z() : 0.0);

    // find indices for leading prong
    DetId id_leading( spr::findDetIdECAL( caloGeom, p4_leading.eta(), p4_leading.phi(), false ) );
    EBDetId ebId( id_leading );
    int leading_iphi_ = ebId.iphi() - 1;
    int leading_ieta_ = ebId.ieta() > 0 ? ebId.ieta()-1 : ebId.ieta();
    
    //find indices for neutral component of jet
    DetId id_neutral( spr::findDetIdECAL( caloGeom, neutral_PF.eta(), neutral_PF.phi(), false ) );
    EBDetId ebId_neutral( id_neutral );
    int neutral_iphi_ = ebId_neutral.iphi() - 1;
    int neutral_ieta_ = ebId_neutral.ieta() > 0 ? ebId_neutral.ieta()-1 : ebId_neutral.ieta();

    // find indices for centering on all PFc
    double eta_avg = eta_sum/total_energy;
    double phi_avg = phi_sum/total_energy;
    DetId id_jet( spr::findDetIdECAL( caloGeom, eta_avg, phi_avg, false ) );
    EBDetId ebId_jet( id_jet );
    int jet_sum_iphi_ = ebId_jet.iphi() - 1;
    int jet_sum_ieta_ = ebId_jet.ieta() > 0 ? ebId_jet.ieta()-1 : ebId_jet.ieta();

    // find indices for centering on gamma and charged hadrons
    double eta_avg1 = eta_sum1/total_energy1;
    double phi_avg1 = phi_sum1/total_energy1;
    DetId id_jet1( spr::findDetIdECAL( caloGeom, eta_avg1, phi_avg1, false ) );
    EBDetId ebId_jet1( id_jet1 );
    int jet_sum_iphi1_ = ebId_jet1.iphi() - 1;
    int jet_sum_ieta1_ = ebId_jet1.ieta() > 0 ? ebId_jet1.ieta()-1 : ebId_jet1.ieta();

    // find indices for centering on egamma
    double eta_avg2 = eta_sum2/total_energy2;
    double phi_avg2 = phi_sum2/total_energy2;
    DetId id_jet2( spr::findDetIdECAL( caloGeom, eta_avg2, phi_avg2, false ) );
    EBDetId ebId_jet2( id_jet2 );
    int jet_sum_iphi2_ = ebId_jet2.iphi() - 1;
    int jet_sum_ieta2_ = ebId_jet2.ieta() > 0 ? ebId_jet2.ieta()-1 : ebId_jet2.ieta();

    const auto centre_pos = caloGeom->getPosition(ebId_jet2);
    double jet_sum_phi2_ = centre_pos.phi();
    double jet_sum_eta2_ = centre_pos.eta();

    std::pair<int, reco::GenTau*> match = getTruthLabelForTauJets(thisJet, genParticles, genJets, magneticField, 0.4, false);

    int truthLabel = match.first;
    vTaujet_jet_truthLabel_      .push_back(truthLabel);

    int truthDM=-1;
    float neutral_pT=0.;
    float neutral_M=0.;
    float neutral_eta=0.;
    float neutral_phi=0.;
    vector<float> charge_p_indv;
    vector<float> neutral_p_indv;
    vector<float> charge_eta_indv;
    vector<float> neutral_eta_indv;
    vector<float> charge_phi_indv;
    vector<float> neutral_phi_indv;

    vector<double> charge_relp_indv;
    vector<double> neutral_relp_indv;
    vector<double> charge_releta_indv;
    vector<double> neutral_releta_indv;
    vector<double> charge_relphi_indv;
    vector<double> neutral_relphi_indv;

    vector<float> charge_releta_crystal_indv;
    vector<float> neutral_releta_crystal_indv;
    vector<float> charge_relphi_crystal_indv;
    vector<float> neutral_relphi_crystal_indv;

    if(match.second->neutral_p4().mass()>0) {
      vTaujet_jet_neutral_relmass_.push_back(match.second->neutral_p4().mass());
      vTaujet_jet_neutral_relp_.push_back(match.second->neutral_p4().P());

      double magneticField = (magfield.product() ? magfield.product()->inTesla(GlobalPoint(0., 0., 0.)).z() : 0.0);
      math::XYZTLorentzVector  prop_p4(match.second->neutral_p4().px(),match.second->neutral_p4().py(),match.second->neutral_p4().pz(),match.second->neutral_p4().energy()); //setup 4-vector 
      BaseParticlePropagator propagator = BaseParticlePropagator(
          RawParticle(prop_p4, math::XYZTLorentzVector(match.second->vx(), match.second->vy(), match.second->vz(), 0.),
                      0.),0.,0.,magneticField);
      propagator.propagateToEcalEntrance(false); // propogate to ECAL entrance
      auto neutral_position = propagator.particle().vertex().Vect();

      double eta = neutral_position.eta();
      double phi = neutral_position.phi();
      double releta = eta-jet_sum_eta2_;
      double relphi = phi-jet_sum_phi2_;
      relphi = TVector2::Phi_mpi_pi(relphi);
      vTaujet_jet_neutral_relphi_.push_back(relphi);
      vTaujet_jet_neutral_releta_.push_back(releta);

      // also store in crystal units:
      DetId id( spr::findDetIdECAL( caloGeom, eta, phi, false ) );
      EBDetId ebId( id );

      // get index of the crystal
      float iphi = ebId.iphi() -1;
      float ieta = ebId.ieta() > 0 ? ebId.ieta()-1 : ebId.ieta();

      float ieta_cont = -10000;
      float iphi_cont = -10000;

      if (nullptr != caloGeom->getGeometry(ebId)) {
        // now work out how far along the crystal the particle overlapped to get a continuous number
        const auto repCorners = caloGeom->getGeometry(ebId)->getCornersREP();
        float minEta_ = repCorners[2].eta();
        float maxEta_ = repCorners[0].eta();
        float minPhi_ = repCorners[2].phi();
        float maxPhi_ = repCorners[0].phi();

        ieta_cont = ieta+(eta-minEta_)/(maxEta_-minEta_);
        iphi_cont = iphi+(phi-minPhi_)/(maxPhi_-minPhi_);
      }
      vTaujet_jet_neutral_releta_crystal_.push_back(ieta_cont);
      vTaujet_jet_neutral_relphi_crystal_.push_back(iphi_cont);

    } else {
      // when there is no pi0's set the 4-vector size to 0 and direction to centre of image
      vTaujet_jet_neutral_relmass_.push_back(0.);
      vTaujet_jet_neutral_relp_.push_back(0.);
      vTaujet_jet_neutral_relphi_.push_back(0.);
      vTaujet_jet_neutral_releta_.push_back(0.);
      vTaujet_jet_neutral_relphi_crystal_.push_back(jet_sum_iphi2_);
      vTaujet_jet_neutral_releta_crystal_.push_back(jet_sum_ieta2_);
    }
    if (abs(truthLabel)==15) {
      truthDM = match.second->decay_mode();
      neutral_pT = match.second->neutral_p4().pt();
      neutral_M = match.second->neutral_p4().mass();
      neutral_eta = match.second->neutral_p4().eta();
      neutral_phi = match.second->neutral_p4().phi();

      // Save charged prongs and index:
      for (const auto &charged : match.second->charge_p4_indv()){
          // Find ieta iphi index
          DetId id_leading( spr::findDetIdECAL( caloGeom, charged.eta(), charged.phi(), false ) );
          EBDetId ebId( id_leading );
          int charged_iphi_ = ebId.iphi() - 1;
          int charged_ieta_ = ebId.ieta() > 0 ? ebId.ieta()-1 : ebId.ieta();
          
          charge_p_indv.push_back(charged.energy());
          charge_eta_indv.push_back(charged_ieta_);
          charge_phi_indv.push_back(charged_iphi_);
      }
      for (auto x : match.second->pis_at_ecal()){
          double p = x.second;
          double eta = x.first.eta();
          double phi = x.first.phi();
          double releta = eta-jet_sum_eta2_;
          double relphi = phi-jet_sum_phi2_;
          relphi = TVector2::Phi_mpi_pi(relphi);
          charge_relp_indv.push_back(p);
          charge_releta_indv.push_back(releta);
          charge_relphi_indv.push_back(relphi);

          // also store in crystal units:
          DetId id( spr::findDetIdECAL( caloGeom, eta, phi, false ) );
          EBDetId ebId( id );

          // get index of the crystal
          float iphi = ebId.iphi() -1;
          float ieta = ebId.ieta() > 0 ? ebId.ieta()-1 : ebId.ieta(); 

          float ieta_cont = -10000;
          float iphi_cont = -10000;

          if (nullptr != caloGeom->getGeometry(ebId)) {     
            // now work out how far along the crystal the particle overlapped to get a continuous number
            const auto repCorners = caloGeom->getGeometry(ebId)->getCornersREP();
            float minEta_ = repCorners[2].eta();
            float maxEta_ = repCorners[0].eta();
            float minPhi_ = repCorners[2].phi();
            float maxPhi_ = repCorners[0].phi();
            ieta_cont = ieta+(eta-minEta_)/(maxEta_-minEta_);
            iphi_cont = iphi+(phi-minPhi_)/(maxPhi_-minPhi_);  
          }
          charge_releta_crystal_indv.push_back(ieta_cont);
          charge_relphi_crystal_indv.push_back(iphi_cont);
        }
      if (match.second->neutral_p4_indv().size()>0){
        for (const auto &neutral : match.second->neutral_p4_indv()){
            DetId id_neutral( spr::findDetIdECAL( caloGeom, neutral.eta(), neutral.phi(), false ) );
            EBDetId ebId_neutral( id_neutral );
            int neutral_iphi_ = ebId_neutral.iphi() - 1;
            int neutral_ieta_ = ebId_neutral.ieta() > 0 ? ebId_neutral.ieta()-1 : ebId_neutral.ieta();


            neutral_p_indv.push_back(neutral.energy());
            neutral_eta_indv.push_back(neutral_ieta_);
            neutral_phi_indv.push_back(neutral_iphi_);
          }
      } else{
        neutral_p_indv.push_back(-1);
        neutral_eta_indv.push_back(-100); 
        neutral_phi_indv.push_back(-100);
        
      }
      if (match.second->pi0s_at_ecal().size()>0){
        for (auto x : match.second->pi0s_at_ecal()){
            double p = x.second;
            double eta = x.first.eta(); 
            double phi = x.first.phi(); 
            double releta = eta-jet_sum_eta2_;
            double relphi = phi-jet_sum_phi2_; 
            relphi = TVector2::Phi_mpi_pi(relphi);
            neutral_relp_indv.push_back(p);
            neutral_releta_indv.push_back(releta);
            neutral_relphi_indv.push_back(relphi);

            // also store in crystal units:
            DetId id( spr::findDetIdECAL( caloGeom, eta, phi, false ) );
            EBDetId ebId( id );

            // get index of the crystal
            float iphi = ebId.iphi() -1;
            float ieta = ebId.ieta() > 0 ? ebId.ieta()-1 : ebId.ieta(); 
      
            float ieta_cont = -10000;
            float iphi_cont = -10000;

            if (nullptr != caloGeom->getGeometry(ebId)) {
              // now work out how far along the crystal the particle overlapped to get a continuous number
              const auto repCorners = caloGeom->getGeometry(ebId)->getCornersREP();
              float minEta_ = repCorners[2].eta();
              float maxEta_ = repCorners[0].eta();
              float minPhi_ = repCorners[2].phi();
              float maxPhi_ = repCorners[0].phi();

              ieta_cont = ieta+(eta-minEta_)/(maxEta_-minEta_);
              iphi_cont = iphi+(phi-minPhi_)/(maxPhi_-minPhi_); 
            }
            neutral_releta_crystal_indv.push_back(ieta_cont);
            neutral_relphi_crystal_indv.push_back(iphi_cont);

            // uncomment below to determine actual direction
            //math::XYZVector direction = GetPi0Direction(match.second->vertex(), releta, relphi, jet_sum_eta2_, jet_sum_phi2_);

          }
      } else{
        neutral_relp_indv.push_back(0.);
        neutral_releta_indv.push_back(0.);
        neutral_relphi_indv.push_back(0.);
      }
    } else{
      charge_p_indv.push_back(-1);
      neutral_p_indv.push_back(-1);
      charge_eta_indv.push_back(-100);
      neutral_eta_indv.push_back(-100);
      charge_phi_indv.push_back(-100);
      neutral_phi_indv.push_back(-100);
    }

    vTaujet_jet_truthDM_.push_back(truthDM);
    vTaujet_jet_neutral_pT_.push_back(neutral_pT);
    vTaujet_jet_neutral_m0_.push_back(neutral_M);
    vTaujet_jet_neutral_eta_.push_back(neutral_eta);
    vTaujet_jet_neutral_phi_.push_back(neutral_phi);

    vTaujet_jet_charged_indv_p_.push_back(charge_p_indv);
    vTaujet_jet_neutral_indv_p_.push_back(neutral_p_indv);
    vTaujet_jet_charged_indv_eta_.push_back(charge_eta_indv);
    vTaujet_jet_neutral_indv_eta_.push_back(neutral_eta_indv);
    vTaujet_jet_charged_indv_phi_.push_back(charge_phi_indv);
    vTaujet_jet_neutral_indv_phi_.push_back(neutral_phi_indv);

    vTaujet_jet_leading_eta_.push_back(p4_leading.eta());
    vTaujet_jet_leading_phi_.push_back(p4_leading.phi());
    vTaujet_jet_leading_ieta_.push_back(leading_ieta_);
    vTaujet_jet_leading_iphi_.push_back(leading_iphi_);
    vTaujet_jet_leading_energy_.push_back(p4_leading.energy());

    vTaujet_jet_neutralsum_eta_.push_back(neutral_PF.eta());
    vTaujet_jet_neutralsum_phi_.push_back(neutral_PF.phi());
    vTaujet_jet_neutralsum_ieta_.push_back(neutral_ieta_);
    vTaujet_jet_neutralsum_iphi_.push_back(neutral_iphi_);
    vTaujet_jet_neutralsum_pt_.push_back(neutral_PF.pt());
    vTaujet_jet_neutralsum_ECAL_.push_back(neutral_ECAL);

    vTaujet_jet_centre_ieta_.push_back(jet_sum_ieta_);
    vTaujet_jet_centre_iphi_.push_back(jet_sum_iphi_);
    vTaujet_jet_centre1_ieta_.push_back(jet_sum_ieta1_);
    vTaujet_jet_centre1_iphi_.push_back(jet_sum_iphi1_);
    vTaujet_jet_centre2_ieta_.push_back(jet_sum_ieta2_);
    vTaujet_jet_centre2_iphi_.push_back(jet_sum_iphi2_);
    vTaujet_jet_centre2_eta_.push_back(jet_sum_eta2_);
    vTaujet_jet_centre2_phi_.push_back(jet_sum_phi2_); 

    vTaujet_jet_charged_indv_relp_.push_back(charge_relp_indv);
    vTaujet_jet_neutral_indv_relp_.push_back(neutral_relp_indv);
    vTaujet_jet_charged_indv_releta_.push_back(charge_releta_indv);
    vTaujet_jet_neutral_indv_releta_.push_back(neutral_releta_indv);
    vTaujet_jet_charged_indv_relphi_.push_back(charge_relphi_indv);
    vTaujet_jet_neutral_indv_relphi_.push_back(neutral_relphi_indv);
   
    vTaujet_jet_charged_indv_releta_crystal_.push_back(charge_releta_crystal_indv);
    vTaujet_jet_neutral_indv_releta_crystal_.push_back(neutral_releta_crystal_indv);
    vTaujet_jet_charged_indv_relphi_crystal_.push_back(charge_relphi_crystal_indv);
    vTaujet_jet_neutral_indv_relphi_crystal_.push_back(neutral_relphi_crystal_indv);
 
  }//vJetIdxs


} // fillEvtSel_jet_taujet()
