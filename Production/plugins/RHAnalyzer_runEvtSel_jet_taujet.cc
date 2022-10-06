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

// eta/phi weighted average of the Egamma components
vector<float> vTaujet_jet_centre2_ieta_;
vector<float> vTaujet_jet_centre2_iphi_;
vector<double> vTaujet_jet_centre2_eta_;
vector<double> vTaujet_jet_centre2_phi_;

// HPS tau direction - assuming no bending in the B field 
// If no HPS tau exists then use the jet direction
vector<double> vTaujet_tau_centre_ieta_;
vector<double> vTaujet_tau_centre_iphi_;
vector<double> vTaujet_tau_centre_eta_;
vector<double> vTaujet_tau_centre_phi_;
// HPS neutral direction - assuming no bending in the B field 
// If no HPS tau exists then use the neutral (E/gamma) jet direction
vector<double> vTaujet_pi0_centre_ieta_;
vector<double> vTaujet_pi0_centre_iphi_;
vector<double> vTaujet_pi0_centre_eta_;
vector<double> vTaujet_pi0_centre_phi_;

// HPS variables 
// common variables
vector<float> vHPSTau_dm_;
vector<float> vHPSTau_pt_;
vector<float> vHPSTau_E_;
vector<float> vHPSTau_eta_;
vector<float> vHPSTau_M_;
vector<float> vHPSTau_pi_px_;
vector<float> vHPSTau_pi_py_;
vector<float> vHPSTau_pi_pz_;
vector<float> vHPSTau_pi_E_;
vector<float> vHPSTau_pi0_px_;
vector<float> vHPSTau_pi0_py_;
vector<float> vHPSTau_pi0_pz_;
vector<float> vHPSTau_pi0_E_;
vector<float> vHPSTau_strip_mass_;
vector<float> vHPSTau_strip_pt_;
vector<float> vHPSTau_pi0_dEta_;
vector<float> vHPSTau_pi0_dPhi_;
vector<float> vHPSTau_pi0_releta_;
vector<float> vHPSTau_pi0_relphi_;

// 1pr variables
vector<float> vHPSTau_rho_mass_;

// 3pr variables
vector<float> vHPSTau_pi2_px_;
vector<float> vHPSTau_pi2_py_;
vector<float> vHPSTau_pi2_pz_;
vector<float> vHPSTau_pi2_E_;
vector<float> vHPSTau_pi3_px_;
vector<float> vHPSTau_pi3_py_;
vector<float> vHPSTau_pi3_pz_;
vector<float> vHPSTau_pi3_E_;

vector<float> vHPSTau_mass0_;
vector<float> vHPSTau_mass1_;
vector<float> vHPSTau_mass2_;

vector<float> vHPSTau_tau_mva_dm_;
vector<float> vHPSTau_tau_deeptau_id_;
vector<float> vHPSTau_tau_deeptau_id_vs_mu_;
vector<float> vHPSTau_tau_deeptau_id_vs_e_;

typedef ROOT::Math::PtEtaPhiEVector PtEtaPhiELV;
const double mass_pi = 0.13498;
const double mass_rho = 0.7755;

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
  tree->Branch("jet_neutral_indv_relp", &vTaujet_jet_neutral_indv_relp_); // THIS
  tree->Branch("jet_charged_indv_releta", &vTaujet_jet_charged_indv_releta_); 
  tree->Branch("jet_neutral_indv_releta", &vTaujet_jet_neutral_indv_releta_); // THIS 
  tree->Branch("jet_charged_indv_relphi", &vTaujet_jet_charged_indv_relphi_);
  tree->Branch("jet_neutral_indv_relphi", &vTaujet_jet_neutral_indv_relphi_); // THIS

  tree->Branch("jet_charged_indv_releta_crystal", &vTaujet_jet_charged_indv_releta_crystal_);
  tree->Branch("jet_neutral_indv_releta_crystal", &vTaujet_jet_neutral_indv_releta_crystal_);
  tree->Branch("jet_charged_indv_relphi_crystal", &vTaujet_jet_charged_indv_relphi_crystal_);
  tree->Branch("jet_neutral_indv_relphi_crystal", &vTaujet_jet_neutral_indv_relphi_crystal_);

  tree->Branch("jet_neutral_relphi_crystal", &vTaujet_jet_neutral_relphi_crystal_);
  tree->Branch("jet_neutral_releta_crystal", &vTaujet_jet_neutral_releta_crystal_);
  tree->Branch("jet_neutral_relphi", &vTaujet_jet_neutral_relphi_); 
  tree->Branch("jet_neutral_releta", &vTaujet_jet_neutral_releta_);
  tree->Branch("jet_neutral_relp", &vTaujet_jet_neutral_relp_); // this
  tree->Branch("jet_neutral_relmass", &vTaujet_jet_neutral_relmass_);

  tree->Branch("jet_centre2_ieta", &vTaujet_jet_centre2_ieta_);
  tree->Branch("jet_centre2_iphi", &vTaujet_jet_centre2_iphi_);
  tree->Branch("jet_centre2_eta", &vTaujet_jet_centre2_eta_);
  tree->Branch("jet_centre2_phi", &vTaujet_jet_centre2_phi_);

  tree->Branch("tau_centre_ieta", &vTaujet_tau_centre_ieta_);
  tree->Branch("tau_centre_iphi", &vTaujet_tau_centre_iphi_);
  tree->Branch("pi0_centre_ieta", &vTaujet_pi0_centre_ieta_);
  tree->Branch("pi0_centre_iphi", &vTaujet_pi0_centre_iphi_);

  tree->Branch("tau_centre_eta", &vTaujet_tau_centre_eta_);
  tree->Branch("tau_centre_phi", &vTaujet_tau_centre_phi_);
  tree->Branch("pi0_centre_eta", &vTaujet_pi0_centre_eta_);
  tree->Branch("pi0_centre_phi", &vTaujet_pi0_centre_phi_);

  // HPS variables 
  tree->Branch("tau_dm", &vHPSTau_dm_);
  tree->Branch("tau_pt", &vHPSTau_pt_);
  tree->Branch("tau_E", &vHPSTau_E_);
  tree->Branch("tau_eta", &vHPSTau_eta_);
  tree->Branch("tau_mass", &vHPSTau_M_);
  tree->Branch("pi_px", &vHPSTau_pi_px_);
  tree->Branch("pi_py", &vHPSTau_pi_py_);
  tree->Branch("pi_pz", &vHPSTau_pi_pz_);
  tree->Branch("pi_E", &vHPSTau_pi_E_);
  tree->Branch("pi0_px", &vHPSTau_pi0_px_);
  tree->Branch("pi0_py", &vHPSTau_pi0_py_);
  tree->Branch("pi0_pz", &vHPSTau_pi0_pz_);
  tree->Branch("pi0_E", &vHPSTau_pi0_E_);
  tree->Branch("pi0_dEta", &vHPSTau_pi0_dEta_);
  tree->Branch("pi0_dPhi", &vHPSTau_pi0_dPhi_);
  tree->Branch("HPSpi0_releta", &vHPSTau_pi0_releta_);
  tree->Branch("HPSpi0_relphi", &vHPSTau_pi0_relphi_);
  tree->Branch("strip_mass", &vHPSTau_strip_mass_);
  tree->Branch("strip_pt", &vHPSTau_strip_pt_);
  tree->Branch("rho_mass", &vHPSTau_rho_mass_);
  tree->Branch("pi2_px", &vHPSTau_pi2_px_);
  tree->Branch("pi2_py", &vHPSTau_pi2_py_);
  tree->Branch("pi2_pz", &vHPSTau_pi2_pz_);
  tree->Branch("pi2_E", &vHPSTau_pi2_E_);
  tree->Branch("pi3_px", &vHPSTau_pi3_px_);
  tree->Branch("pi3_py", &vHPSTau_pi3_py_);
  tree->Branch("pi3_pz", &vHPSTau_pi3_pz_);
  tree->Branch("pi3_E", &vHPSTau_pi3_E_);
  tree->Branch("mass0", &vHPSTau_mass0_);
  tree->Branch("mass1", &vHPSTau_mass1_);
  tree->Branch("mass2", &vHPSTau_mass2_);
  tree->Branch("tau_mva_dm", &vHPSTau_tau_mva_dm_);
  tree->Branch("tau_deeptau_id", &vHPSTau_tau_deeptau_id_);
  tree->Branch("tau_deeptau_id_vs_mu", &vHPSTau_tau_deeptau_id_vs_mu_);
  tree->Branch("tau_deeptau_id_vs_e", &vHPSTau_tau_deeptau_id_vs_e_);

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
  vTaujet_jet_centre2_ieta_.clear();
  vTaujet_jet_centre2_iphi_.clear();
  vTaujet_jet_centre2_eta_.clear();
  vTaujet_jet_centre2_phi_.clear();
  vTaujet_tau_centre_ieta_.clear();
  vTaujet_tau_centre_iphi_.clear();
  vTaujet_tau_centre_eta_.clear();
  vTaujet_tau_centre_phi_.clear();
  vTaujet_pi0_centre_ieta_.clear();
  vTaujet_pi0_centre_iphi_.clear();
  vTaujet_pi0_centre_eta_.clear();
  vTaujet_pi0_centre_phi_.clear();
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

  vHPSTau_dm_.clear();
  vHPSTau_pt_.clear();
  vHPSTau_E_.clear();
  vHPSTau_eta_.clear();
  vHPSTau_M_.clear();
  vHPSTau_pi_px_.clear();
  vHPSTau_pi_py_.clear();
  vHPSTau_pi_pz_.clear();
  vHPSTau_pi_E_.clear();
  vHPSTau_pi0_px_.clear();
  vHPSTau_pi0_py_.clear();
  vHPSTau_pi0_pz_.clear();
  vHPSTau_pi0_E_.clear();
  vHPSTau_strip_mass_.clear();
  vHPSTau_strip_pt_.clear();
  vHPSTau_pi0_dEta_.clear();
  vHPSTau_pi0_dPhi_.clear();
  vHPSTau_pi0_releta_.clear();
  vHPSTau_pi0_relphi_.clear();
  vHPSTau_rho_mass_.clear();
  vHPSTau_pi2_px_.clear();
  vHPSTau_pi2_py_.clear();
  vHPSTau_pi2_pz_.clear();
  vHPSTau_pi2_E_.clear();
  vHPSTau_pi3_px_.clear();
  vHPSTau_pi3_py_.clear();
  vHPSTau_pi3_pz_.clear();
  vHPSTau_pi3_E_.clear();
  vHPSTau_mass0_.clear();
  vHPSTau_mass1_.clear();
  vHPSTau_mass2_.clear();
  vHPSTau_tau_mva_dm_.clear();
  vHPSTau_tau_deeptau_id_.clear();
  vHPSTau_tau_deeptau_id_vs_mu_.clear();
  vHPSTau_tau_deeptau_id_vs_e_.clear();


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

std::vector<std::pair<PtEtaPhiELV, std::vector<reco::CandidatePtr>>> RecHitAnalyzer::HPSGammas(
    std::vector<reco::CandidatePtr> cands) const {
  std::vector<std::pair<PtEtaPhiELV, std::vector<reco::CandidatePtr>>> strips;
  while (!cands.empty()) {
    std::vector<reco::CandidatePtr> associated = {};
    std::vector<reco::CandidatePtr> notAssociated = {};

    PtEtaPhiELV stripVector(0, 0, 0, 0);
    stripVector = cands[0]->p4();
    associated.push_back(cands[0]);

    bool repeat = true;
    unsigned int mini = 1;
    while (repeat) {
      repeat = false;
      for (unsigned int i = mini; i < cands.size(); ++i) {
        double etaAssociationDistance = 0.20 * pow(cands[i]->pt(), -0.66) + 0.20 * pow(stripVector.Pt(), -0.66);
        double phiAssociationDistance = 0.35 * pow(cands[i]->pt(), -0.71) + 0.35 * pow(stripVector.Pt(), -0.71);
        etaAssociationDistance = std::min(etaAssociationDistance, 0.15);
        etaAssociationDistance = std::max(etaAssociationDistance, 0.05);
        phiAssociationDistance = std::min(phiAssociationDistance, 0.30);
        phiAssociationDistance = std::max(phiAssociationDistance, 0.05);

        if (std::abs(cands[i]->eta() - stripVector.eta()) < etaAssociationDistance &&
            std::abs(reco::deltaPhi(cands[i]->p4(), stripVector)) < phiAssociationDistance) {
          stripVector += cands[i]->p4();
          associated.push_back(cands[i]);
          repeat = true;
        } else {
          notAssociated.push_back(cands[i]);
        }
      }
      cands.swap(notAssociated);
      notAssociated.clear();
      mini = 0;
    }

    PtEtaPhiELV strip = getPi0(associated, false);
    strips.push_back(std::make_pair(strip, associated));
  }
  std::sort(strips.begin(), strips.end(), sortStrips<PtEtaPhiELV, std::vector<reco::CandidatePtr>>);

  return strips;
}

PtEtaPhiELV RecHitAnalyzer::getPi0(std::vector<reco::CandidatePtr> gammas, bool leadEtaPhi) const {
  PtEtaPhiELV pi0;
  if (!gammas.empty()) {
    double tot_energy = 0.;
    double phi = 0.;
    double eta = 0.;
    for (const auto& g : gammas) {
      tot_energy += g->energy();
      phi += g->energy() * g->phi();
      eta += g->energy() * g->eta();
    }
    eta /= tot_energy;
    phi /= tot_energy;

    if (leadEtaPhi) {
      // if true sets the eta and phi of the pi0 to that of the leading gamma rather than using the weighted average
      eta = gammas[0]->eta();
      phi = gammas[0]->phi();
    }

    double p = sqrt(tot_energy * tot_energy - mass_pi * mass_pi);
    double theta = atan(exp(-eta)) * 2;
    double pt = p * sin(theta);
    pi0.SetCoordinates(pt, eta, phi, tot_energy);
  }
  return pi0;
}

std::pair<PtEtaPhiELV, PtEtaPhiELV> RecHitAnalyzer::getRho(const pat::Tau tau, double gammas_pt_cut) const {
  PtEtaPhiELV pi;
  PtEtaPhiELV pi0;
  gammas_.clear();

  std::vector<reco::CandidatePtr> gammas;
  for (const auto& g : tau.signalGammaCands())
    if (g->pt() > gammas_pt_cut)
      gammas.push_back(g);

  const auto& hads = tau.signalChargedHadrCands();
  if (!hads.empty())
    pi = hads[0]->p4();

  double cone_size2 = std::pow(std::clamp(3. / tau.pt(), 0.05, 0.1), 2);
  std::vector<std::pair<PtEtaPhiELV, std::vector<reco::CandidatePtr>>> strip_pairs = HPSGammas(gammas);
  std::vector<std::pair<PtEtaPhiELV, std::vector<reco::CandidatePtr>>> strips_incone;
  for (const auto& s : strip_pairs) {
    if (reco::deltaR2(s.first, tau.p4()) < cone_size2)
      strips_incone.push_back(s);
  }
  if (tau.decayMode() == 0) {
    if (!strips_incone.empty()) {
      gammas = strips_incone[0].second;
    } else if (!strip_pairs.empty()) {
      gammas = strip_pairs[0].second;
    }
  }
  if ((tau.decayMode() == 1 || tau.decayMode() == 2) && !strip_pairs.empty()){
    //std::cout<< "Strip pairs size: " << strip_pairs.size() << std::endl;
    pi0 = getPi0(strip_pairs[0].second, true);
  }
  else {
    pi0 = getPi0(gammas, true);
  }
  std::sort(gammas.begin(), gammas.end(), sortByPT<reco::CandidatePtr>);
  gammas_ = gammas;
  return std::make_pair(pi, pi0);
}

std::pair<std::vector<PtEtaPhiELV>, PtEtaPhiELV> RecHitAnalyzer::getA1(const pat::Tau tau, float gammas_pt_cut) const {
  std::vector<PtEtaPhiELV> prongs;
  PtEtaPhiELV pi0;
  std::vector<reco::CandidatePtr> hads;
  for (const auto& h : tau.signalChargedHadrCands())
    hads.push_back(h);
  if (hads.size() == 3) {
    // arrange hadrons so the oppositly charged hadron is contained in the first element
    if (hads[1]->charge() != hads[0]->charge() && hads[1]->charge() != hads[2]->charge()) {
      auto temp = hads[1];
      hads[1] = hads[0];
      hads[0] = temp;
    } else if (hads[2]->charge() != hads[0]->charge() && hads[2]->charge() != hads[1]->charge()) {
      auto temp = hads[2];
      hads[2] = hads[0];
      hads[0] = temp;
    }
    // from the two same sign hadrons place the one that gives the mass most similar to the rho meson as the second element
    double dM1 = std::abs((hads[0]->p4() + hads[1]->p4()).M() - mass_rho);
    double dM2 = std::abs((hads[0]->p4() + hads[2]->p4()).M() - mass_rho);
    if (dM2 < dM1) {
      auto temp = hads[2];
      hads[2] = hads[1];
      hads[1] = temp;
    }
  }

  std::vector<reco::CandidatePtr> gammas_merge;
  for (const auto& g : tau.signalGammaCands())
    if (g->pt() > gammas_pt_cut)
      gammas_merge.push_back(g);
  if (tau.decayMode() != 11) {
    for (const auto& g : tau.isolationGammaCands())
      if (g->pt() > gammas_pt_cut)
        gammas_merge.push_back(g);
  }
  std::sort(gammas_merge.begin(), gammas_merge.end(), sortByPT<reco::CandidatePtr>);
  std::vector<reco::CandidatePtr> gammas = {};
  gammas.reserve(gammas_merge.size());
  for (const auto& g : gammas_merge)
    gammas.push_back(g);
  double cone_size2 = std::pow(std::clamp(3. / tau.pt(), 0.05, 0.1), 2);
  std::vector<std::pair<PtEtaPhiELV, std::vector<reco::CandidatePtr>>> strip_pairs = HPSGammas(gammas);
  std::vector<std::pair<PtEtaPhiELV, std::vector<reco::CandidatePtr>>> strips_incone;
  for (const auto& s : strip_pairs)
    if (reco::deltaR2(s.first, tau.p4()) < cone_size2)
      strips_incone.push_back(s);

  std::vector<reco::CandidatePtr> signal_gammas = {};

  if (!strips_incone.empty()) {
    signal_gammas = strips_incone[0].second;
  } else if (!strip_pairs.empty()) {
    signal_gammas = strip_pairs[0].second;
  }
  pi0 = getPi0(signal_gammas, true);
  std::sort(signal_gammas.begin(), signal_gammas.end(), sortByPT<reco::CandidatePtr>);
  gammas_ = signal_gammas;

  prongs.reserve(hads.size());
  for (const auto& h : hads)
    prongs.push_back(PtEtaPhiELV(h->p4()));

  return std::make_pair(prongs, pi0);
}

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
  double magneticField = (magfield.product() ? magfield.product()->inTesla(GlobalPoint(0., 0., 0.)).z() : 0.0);

  edm::Handle<pat::TauCollection> slimmedTausH_;
  iEvent.getByToken( slimmedTausCollectionT_, slimmedTausH_ );

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

    reco::PFCandidatePtr leading_pfEG;

    double total_energy2 = 0; // total energy of event
    double eta_sum2 = 0; //sum of E_i*eta_i for all i
    double phi_sum2 = 0;

    for (const auto &pfC : pfCands){
      
      
      // Loop over all PF candidates and make energy weighted average position
      
      // propagate all particles 
      math::XYZTLorentzVector  prop_p4(pfC->p4().px(),pfC->p4().py(),pfC->p4().pz(),sqrt(pow(pfC->p(),2)+pfC->mass()*pfC->mass())); //setup 4-vector 
      BaseParticlePropagator propagator = BaseParticlePropagator(
          RawParticle(prop_p4, math::XYZTLorentzVector(pfC->vx(), pfC->vy(), pfC->vz(), 0.),
                      pfC->charge()),0.,0.,magneticField);
      propagator.propagateToEcalEntrance(false); // propogate to ECAL entrance
      auto pfC_position = propagator.particle().vertex().Vect();
      
      if (pfC->particleId() == 4 || pfC->particleId()==2){
        // Store gamma and e for position
        eta_sum2 += pfC_position.eta()*pfC->energy();
        phi_sum2 += pfC_position.phi()*pfC->energy();
        total_energy2 += pfC->energy();
      }

      if (pfC->particleId() == 4 || pfC->particleId()==2){
        if (leading_pfEG.isNull() || (pfC->p4().pt() > leading_pfEG->p4().pt())){
          // store leading egamma 
          leading_pfEG = pfC;
        }
      }
   } 

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
            // math::XYZVector direction = GetPi0Direction(match.second->vertex(), releta, relphi, jet_sum_eta2_, jet_sum_phi2_);
            // std::cout << "DIRECTION OUTPUT: " << direction << std::endl;

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
 

    // add HPS Info
    float tau_dm =-1; 
    float tau_pt =0; 
    float tau_E =0; 
    float tau_eta =0; 
    float tau_mass=0; 
    float pi_px=0;
    float pi_py=0;
    float pi_pz=0;
    float pi_E=0;
    float pi0_px =0;
    float pi0_py =0;
    float pi0_pz =0;
    float pi0_E=0;
    float pi0_dEta=0;
    float pi0_dPhi=0;
    float pi0_releta=0;
    float pi0_relphi=0;
    float strip_mass=0;
    float strip_pt=0;
    float rho_mass=0;
    float pi2_px=0;
    float pi2_py=0;
    float pi2_pz=0;
    float pi2_E=0;
    float pi3_px=0;
    float pi3_py=0;
    float pi3_pz=0;
    float pi3_E=0;
    float mass0=0;
    float mass1=0;
    float mass2=0;
    float tau_mva_dm=-1;
    float tau_deeptau_id=0;
    float tau_deeptau_id_vs_mu=0;
    float tau_deeptau_id_vs_e=0;

    // initialise centre based on jet properties only - this will then be overwritten if a matched HPS tau exists
    // For tau centering use jet direction
    math::XYZTLorentzVector  tau_p4(thisJet->px(),thisJet->py(),thisJet->pz(),thisJet->energy()); //setup 4-vector 
    BaseParticlePropagator jet_propagator = BaseParticlePropagator(
        RawParticle(tau_p4, math::XYZTLorentzVector(vtxs[0].position().x(), vtxs[0].position().y(), vtxs[0].position().z(), 0.),
                    0.0),0.,0.,magneticField);
    jet_propagator.propagateToEcalEntrance(false); // propogate to ECAL entrance
    auto tau_prop = jet_propagator.particle().vertex().Vect();

    float tau_centre_eta = tau_prop.eta();
    float tau_centre_phi = tau_prop.phi();
    DetId id_tau_centre( spr::findDetIdECAL( caloGeom, tau_centre_eta, tau_centre_phi, false ) );
    EBDetId ebId_centre( id_tau_centre );
    int tau_centre_iphi = ebId_centre.iphi() - 1;
    int tau_centre_ieta = ebId_centre.ieta() > 0 ? ebId_centre.ieta()-1 : ebId_centre.ieta();

    // For pi0 centering use leading PF e/gamma direction
    // First we initialise to tau jet direction incase a PF e/gamma doesn't exist
    float pi0_centre_eta = tau_centre_eta;
    float pi0_centre_phi = tau_centre_phi;
    float pi0_centre_ieta = tau_centre_ieta;
    float pi0_centre_iphi = tau_centre_iphi;

    if(!(leading_pfEG.isNull())) {
      math::XYZTLorentzVector  pi0_p4(leading_pfEG->p4().px(),leading_pfEG->p4().py(),leading_pfEG->p4().pz(),leading_pfEG->energy()); //setup 4-vector 
      BaseParticlePropagator pi0_propagator = BaseParticlePropagator(
          RawParticle(pi0_p4, math::XYZTLorentzVector(vtxs[0].position().x(), vtxs[0].position().y(), vtxs[0].position().z(), 0.),
                      0.0),0.,0.,magneticField);
      pi0_propagator.propagateToEcalEntrance(false); // propogate to ECAL entrance
      auto pi0_prop = pi0_propagator.particle().vertex().Vect();
      pi0_centre_eta = pi0_prop.eta();
      pi0_centre_phi = pi0_prop.phi();

      DetId id_pi0_centre( spr::findDetIdECAL( caloGeom, pi0_centre_eta, pi0_centre_phi, false ) );
      ebId_centre = EBDetId( id_pi0_centre );
      pi0_centre_iphi = ebId_centre.iphi() - 1;
      pi0_centre_ieta = ebId_centre.ieta() > 0 ? ebId_centre.ieta()-1 : ebId_centre.ieta();
    } 


    if(slimmedTausH_) {
      float minDR=-1.;
      pat::Tau HPStau;
      // loop over taus and find matching jet
      for (pat::TauCollection::const_iterator iTau = slimmedTausH_->begin(); iTau != slimmedTausH_->end(); ++iTau) {

        float dR = reco::deltaR( thisJet->eta(), thisJet->phi(), iTau->eta(), iTau->phi() );

        if ( dR > 0.5 ) continue;
        if(minDR<0 || dR<minDR) {
          minDR = dR;
          HPStau = *iTau;
        }

      }
      if(minDR>0) {
        //found a match so fill variables
        // tauID discriminators
        std::vector<std::string> vsj_wps = {"VVVLoose", "VVLoose", "VLoose", "Loose", "Medium", "Tight", "VTight", "VVTight"};
        std::vector<std::string> vsj_wps_vs_mu = {"VLoose", "Loose", "Medium", "Tight"};
        std::vector<std::string> vsj_wps_vs_e = {"VVVLoose", "VVLoose", "VLoose", "Loose", "Medium", "Tight", "VTight", "VVTight"};
        for (auto id : vsj_wps) {
          bool pass = HPStau.tauID("by"+id+"DeepTau2017v2p1VSjet");
          if (pass) tau_deeptau_id++;
        }    
        for (auto id : vsj_wps_vs_mu) {
          bool pass = HPStau.tauID("by"+id+"DeepTau2017v2p1VSmu");
          if (pass) tau_deeptau_id_vs_mu++;
        }
        for (auto id : vsj_wps_vs_e) {
          bool pass = HPStau.tauID("by"+id+"DeepTau2017v2p1VSe");
          if (pass) tau_deeptau_id_vs_e++;
        } 
        tau_mva_dm = HPStau.tauID("MVADM2017v1");

        // MVADM input variables

        tau_dm=HPStau.decayMode();
        tau_pt=HPStau.pt();
        tau_E=HPStau.energy();
        tau_eta=std::fabs(HPStau.eta());
        tau_mass=HPStau.mass();

        PtEtaPhiELV pi0;
        PtEtaPhiELV pi;
        std::pair<PtEtaPhiELV, PtEtaPhiELV> rho;
        std::vector<PtEtaPhiELV> a1_daughters = {};

        math::XYZTLorentzVector  tau_p4(HPStau.px(),HPStau.py(),HPStau.pz(),HPStau.energy()); //setup 4-vector 
        BaseParticlePropagator HPStau_propagator = BaseParticlePropagator(
            RawParticle(tau_p4, math::XYZTLorentzVector(vtxs[0].position().x(), vtxs[0].position().y(), vtxs[0].position().z(), 0.),
                        0.0),0.,0.,magneticField);
        HPStau_propagator.propagateToEcalEntrance(false); // propogate to ECAL entrance
        auto tau_prop = HPStau_propagator.particle().vertex().Vect();

        tau_centre_eta = tau_prop.eta();
        tau_centre_phi = tau_prop.phi();
        DetId id_leading( spr::findDetIdECAL( caloGeom, tau_centre_eta, tau_centre_phi, false ) );
        EBDetId ebId( id_leading );
        tau_centre_iphi = ebId.iphi() - 1;
        tau_centre_ieta = ebId.ieta() > 0 ? ebId.ieta()-1 : ebId.ieta();

        // initialise pi0 centre to tau centre in case a pi0 candidate doesn't exist
        pi0_centre_eta = tau_centre_eta;
        pi0_centre_phi = tau_centre_phi;
        pi0_centre_ieta = tau_centre_ieta;
        pi0_centre_iphi = tau_centre_iphi;
      
        if (tau_dm >= 10) {
          std::pair<std::vector<PtEtaPhiELV>, PtEtaPhiELV> a1 = getA1(HPStau, 1.0);
          a1_daughters = a1.first;
          pi0 = a1.second;
          if(a1_daughters.size()>2) {
            mass0 = (a1_daughters[0] + a1_daughters[1] + a1_daughters[2]).M();
            mass1 = (a1_daughters[0] + a1_daughters[1]).M();
            mass2 = (a1_daughters[0] + a1_daughters[2]).M();

            pi_px = a1_daughters[0].Px();
            pi_py = a1_daughters[0].Py();
            pi_pz = a1_daughters[0].Pz();
            pi_E = a1_daughters[0].E();

            pi2_px = a1_daughters[1].Px();
            pi2_py = a1_daughters[1].Py();
            pi2_pz = a1_daughters[1].Pz();
            pi2_E = a1_daughters[1].E();

            pi3_px = a1_daughters[2].Px();
            pi3_py = a1_daughters[2].Py();
            pi3_pz = a1_daughters[2].Pz();
            pi3_E = a1_daughters[2].E();

            PtEtaPhiELV a1 = a1_daughters[0]+a1_daughters[1]+a1_daughters[2];

            pi0_dEta = std::fabs(pi0.eta() - a1.eta());
            pi0_dPhi = std::fabs(ROOT::Math::VectorUtil::DeltaPhi(pi0, a1));

            if (tau_dm==11){
              math::XYZTLorentzVector  pi0_p4(pi0.Px(),pi0.Py(),pi0.Pz(),pi0.E()); //setup 4-vector 
              BaseParticlePropagator HPSpi0_propagator = BaseParticlePropagator(
                  RawParticle(pi0_p4, math::XYZTLorentzVector(vtxs[0].position().x(), vtxs[0].position().y(), vtxs[0].position().z(), 0.),
                              0.0),0.,0.,magneticField);
              HPSpi0_propagator.propagateToEcalEntrance(false); // propogate to ECAL entrance
              auto pi0_prop = HPSpi0_propagator.particle().vertex().Vect();

              pi0_releta = pi0_prop.eta()-jet_sum_eta2_;
              pi0_relphi = pi0_prop.phi()-jet_sum_phi2_;

              pi0_centre_eta = pi0_prop.eta();
              pi0_centre_phi = pi0_prop.phi();
              DetId id_pi0_centre( spr::findDetIdECAL( caloGeom, pi0_centre_eta, pi0_centre_phi, false ) );
              EBDetId ebId( id_pi0_centre );
              pi0_centre_iphi = ebId.iphi() - 1;
              pi0_centre_ieta = ebId.ieta() > 0 ? ebId.ieta()-1 : ebId.ieta();

              //if (truthDM==2){
              //  std::cout<< "*****************************************************************" << std::endl;
              //  std::cout<< "INFO: gen DM 2 reconstructed by HPS as DM 11" << std::endl;
              //  std::cout << "Pi0 energy: " << pi0.E() << " eta: " << pi0_prop.eta() << " phi: " << pi0_prop.phi() << std::endl;
              //}
              }
          }
        } else {
          //if (truthDM==2){
          //  std::cout<< "*****************************************************************" << std::endl;
          //}
          rho = getRho(HPStau, 1.0);
          pi = rho.first;
          pi0 = rho.second;
          rho_mass = (pi + pi0).M();
          pi_px = rho.first.Px();
          pi_py = rho.first.Py();
          pi_pz = rho.first.Pz();
          pi_E = rho.first.E();
          pi0_dPhi = std::fabs(ROOT::Math::VectorUtil::DeltaPhi(pi, pi0));
          pi0_dEta = std::fabs(pi.eta() - pi0.eta());

          if (tau_dm==1){
            math::XYZTLorentzVector  pi0_p4(pi0.Px(),pi0.Py(),pi0.Pz(),pi0.E()); //setup 4-vector 
            BaseParticlePropagator HPSpi0_propagator = BaseParticlePropagator(
                RawParticle(pi0_p4, math::XYZTLorentzVector(vtxs[0].position().x(), vtxs[0].position().y(), vtxs[0].position().z(), 0.),
                            0.0),0.,0.,magneticField);
            HPSpi0_propagator.propagateToEcalEntrance(false); // propogate to ECAL entrance
            auto pi0_prop = HPSpi0_propagator.particle().vertex().Vect();
            pi0_releta = pi0_prop.eta()-jet_sum_eta2_;
            pi0_relphi = pi0_prop.phi()-jet_sum_phi2_;

            pi0_centre_eta = pi0_prop.eta();
            pi0_centre_phi = pi0_prop.phi();
            DetId id_pi0_centre( spr::findDetIdECAL( caloGeom, pi0_centre_eta, pi0_centre_phi, false ) );
            EBDetId ebId( id_pi0_centre );
            pi0_centre_iphi = ebId.iphi() - 1;
            pi0_centre_ieta = ebId.ieta() > 0 ? ebId.ieta()-1 : ebId.ieta();
            //if (truthDM==2){
            //  // std::cout<< "*****************************************************************" << std::endl;
            //  std::cout<< "INFO: gen DM 2 reconstructed by HPS as DM 1" << std::endl;
            //  std::cout << "Pi0 energy: " << pi0.E() << " eta: " << pi0_prop.eta() << " phi: " << pi0_prop.phi() << " releta: " << pi0_releta << " relphi: " << pi0_relphi << std::endl;
            //}
          }
        }
        pi0_px = pi0.Px();
        pi0_py = pi0.Py();
        pi0_pz = pi0.Pz();
        pi0_E = pi0.E();

        PtEtaPhiELV gammas_vector;
        strip_pt = pi0.pt();
        for (const auto& g : gammas_){
          gammas_vector += g->p4();
          //math::XYZTLorentzVector  gamma_p4(g->p4().px(),g->p4().py(),g->p4().pz(),g->p4().E()); //setup 4-vector 
          //BaseParticlePropagator gamma_propagator = BaseParticlePropagator(
          //    RawParticle(gamma_p4, math::XYZTLorentzVector(g->vx(), g->vy(), g->vz(), 0.),
          //                g->charge()),0.,0.,magneticField);
          //gamma_propagator.propagateToEcalEntrance(false); // propogate to ECAL entrance
          //auto gamma_prop = gamma_propagator.particle().vertex().Vect();
          //if (truthDM==2){
          //  std::cout << "Gamma energy: " << g->p4().E() << " eta: " << gamma_prop.eta() << " phi: " << gamma_prop.phi()  << " releta: " << gamma_prop.eta()-jet_sum_eta2_ << " relphi: " << gamma_prop.phi()-jet_sum_phi2_ << std::endl;
          //}
          // std::cout << "Gamma energy: " << g->p4().E() << " mass: " << g->p4().M() << std::endl;
        }    
        strip_mass = gammas_vector.M();
        //if (truthDM==2){
        //  std::cout << "Strip energy: " << gammas_vector.E()  << std::endl;
        //  std::cout << "---------------------------------------------" << std::endl;
        //} 
        
        
      }
    }
    vHPSTau_dm_.push_back(tau_dm);
    vHPSTau_pt_.push_back(tau_pt);
    vHPSTau_E_.push_back(tau_E);
    vHPSTau_eta_.push_back(tau_eta);
    vHPSTau_M_.push_back(tau_mass);
    vHPSTau_pi_px_.push_back(pi_px);
    vHPSTau_pi_py_.push_back(pi_py);
    vHPSTau_pi_pz_.push_back(pi_pz);
    vHPSTau_pi_E_.push_back(pi_E);
    vHPSTau_pi0_px_.push_back(pi0_px);
    vHPSTau_pi0_py_.push_back(pi0_py);
    vHPSTau_pi0_pz_.push_back(pi0_pz);
    vHPSTau_pi0_E_.push_back(pi0_E);
    vHPSTau_strip_mass_.push_back(strip_mass);
    vHPSTau_strip_pt_.push_back(strip_pt);
    vHPSTau_pi0_dEta_.push_back(pi0_dEta);
    vHPSTau_pi0_dPhi_.push_back(pi0_dPhi);
    vHPSTau_pi0_releta_.push_back(pi0_releta);
    vHPSTau_pi0_relphi_.push_back(pi0_relphi);
    vHPSTau_rho_mass_.push_back(rho_mass);
    vHPSTau_pi2_px_.push_back(pi2_px);
    vHPSTau_pi2_py_.push_back(pi2_py);
    vHPSTau_pi2_pz_.push_back(pi2_pz);
    vHPSTau_pi2_E_.push_back(pi2_E);
    vHPSTau_pi3_px_.push_back(pi3_px);
    vHPSTau_pi3_py_.push_back(pi3_py);
    vHPSTau_pi3_pz_.push_back(pi3_pz);
    vHPSTau_pi3_E_.push_back(pi3_E);
    vHPSTau_mass0_.push_back(mass0);
    vHPSTau_mass1_.push_back(mass1);
    vHPSTau_mass2_.push_back(mass2);
    vHPSTau_tau_mva_dm_.push_back(tau_mva_dm);
    vHPSTau_tau_deeptau_id_.push_back(tau_deeptau_id);
    vHPSTau_tau_deeptau_id_vs_mu_.push_back(tau_deeptau_id_vs_mu);
    vHPSTau_tau_deeptau_id_vs_e_.push_back(tau_deeptau_id_vs_e);

    vTaujet_tau_centre_ieta_.push_back(tau_centre_ieta);
    vTaujet_tau_centre_iphi_.push_back(tau_centre_iphi);
    vTaujet_tau_centre_eta_.push_back(tau_centre_eta);
    vTaujet_tau_centre_phi_.push_back(tau_centre_phi);
    vTaujet_pi0_centre_ieta_.push_back(pi0_centre_ieta);
    vTaujet_pi0_centre_iphi_.push_back(pi0_centre_iphi);
    vTaujet_pi0_centre_eta_.push_back(pi0_centre_eta);
    vTaujet_pi0_centre_phi_.push_back(pi0_centre_phi);
  }//vJetIdxs


} // fillEvtSel_jet_taujet()
