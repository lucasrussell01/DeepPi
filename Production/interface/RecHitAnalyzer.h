#ifndef RecHitAnalyzer_h
#define RecHitAnalyzer_h
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

// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
//#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
//#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
//#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

//#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
//#include "DataFormats/SiStripDetId/interface/TECDetId.h"
//#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
//#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

#include "Calibration/IsolatedParticles/interface/DetIdFromEtaPhi.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
//#include "DataFormats/HcalDetId/interface/HcalDetId.h"
//#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DQM/HcalCommon/interface/Constants.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
//#include "DataFormats/TrackReco/interface/Track.h"
//#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TProfile2D.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TMath.h"
#include "TVector2.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h" // reco::PhotonCollection defined here
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/CandIPTagInfo.h"
#include "DeepPi/Production/interface/GenTau.h"
#include "DataFormats/PatCandidates/interface/Tau.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"


#include "Math/Vector4D.h"
#include "Math/Vector4Dfwd.h"
#include "Math/VectorUtil.h"

#include "CommonTools/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "CommonTools/BaseParticlePropagator/interface/RawParticle.h"

//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<> and also remove the line from
// constructor "usesResource("TFileService");"
// This will improve performance in multithreaded jobs.

namespace {

  template <class T, class U>
  bool sortStrips(std::pair<T, U> i, std::pair<T, U> j) {
    return (i.first.pt() > j.first.pt());
  }

  template <class T>
  bool sortByPT(T i, T j) {
    return (i->pt() > j->pt());
  }

}  // namespace

class RecHitAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
//class RecHitAnalyzer : public edm::EDAnalyzer  {
  public:
    explicit RecHitAnalyzer(const edm::ParameterSet&);
    ~RecHitAnalyzer();

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    virtual void beginJob(const edm::EventSetup&);
    virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
    virtual void endJob() override;
    
    typedef ROOT::Math::PtEtaPhiEVector PtEtaPhiELV;
    mutable std::vector<reco::CandidatePtr> gammas_;

    // ----------member data ---------------------------
    // Tokens
    edm::EDGetTokenT<EcalRecHitCollection> EBRecHitCollectionT_;
    edm::EDGetTokenT<EBDigiCollection>     EBDigiCollectionT_;
    edm::EDGetTokenT<EcalRecHitCollection> EERecHitCollectionT_;
    edm::EDGetTokenT<EcalRecHitCollection> ESRecHitCollectionT_;
    edm::EDGetTokenT<HBHERecHitCollection> HBHERecHitCollectionT_;
    edm::EDGetTokenT<TrackingRecHitCollection> TRKRecHitCollectionT_;
    edm::EDGetTokenT<reco::GenParticleCollection> genParticleCollectionT_;
    edm::EDGetTokenT<reco::PhotonCollection> photonCollectionT_;
    edm::EDGetTokenT<reco::PFJetCollection> jetCollectionT_;
    edm::EDGetTokenT<reco::GenJetCollection> genJetCollectionT_;
    edm::EDGetTokenT<reco::TrackCollection> trackCollectionT_;
    edm::EDGetTokenT<reco::VertexCollection> vertexCollectionT_;
    edm::EDGetTokenT<edm::View<reco::Jet> > recoJetsT_;
    edm::EDGetTokenT<reco::JetTagCollection> jetTagCollectionT_;
    edm::EDGetTokenT<std::vector<reco::CandIPTagInfo> >    ipTagInfoCollectionT_;
    edm::EDGetTokenT<std::vector<reco::PFRecHit>> PFEBRecHitCollectionT_;
    edm::EDGetTokenT<std::vector<reco::PFRecHit>> PFHBHERecHitCollectionT_;
    edm::EDGetTokenT<std::vector<reco::GsfTrack>> gsfTracksCollectionT_;
    edm::EDGetTokenT<pat::TauCollection> slimmedTausCollectionT_;
    
    typedef std::vector<reco::PFCandidate>  PFCollection;
    edm::EDGetTokenT<PFCollection> pfCollectionT_;
    //edm::InputTag trackTags_; //used to select what tracks to read from configuration file

    // Diagnostic histograms
    //TH2D * hEB_adc[EcalDataFrame::MAXSAMPLES]; 
    //TH1D * hHBHE_depth; 
    TH1F *h_sel;

    // Main TTree
    TTree* RHTree;

    // Objects used to fill RHTree branches
    //std::vector<float> vEB_adc_[EcalDataFrame::MAXSAMPLES];
    //std::vector<float> vFC_inputs_;
    //math::PtEtaPhiELorentzVectorD vPho_[2];
  
    // Selection and filling functions
    void branchesEvtSel         ( TTree*, edm::Service<TFileService>& );
    void branchesEvtSel_jet     ( TTree*, edm::Service<TFileService>& );
    void branchesEB             ( TTree*, edm::Service<TFileService>& );
    void branchesEE             ( TTree*, edm::Service<TFileService>& );
    void branchesES             ( TTree*, edm::Service<TFileService>& );
    //void branchesESatEE         ( TTree*, edm::Service<TFileService>& );
    void branchesHBHE           ( TTree*, edm::Service<TFileService>& );
    void branchesECALatHCAL     ( TTree*, edm::Service<TFileService>& );
    void branchesECALstitched   ( TTree*, edm::Service<TFileService>& );
    void branchesHCALatEBEE     ( TTree*, edm::Service<TFileService>& );
    void branchesTracksAtEBEE   ( TTree*, edm::Service<TFileService>& );
    void branchesTracksAtECALstitched   ( TTree*, edm::Service<TFileService>& );
    void branchesPFCandsAtEBEE   ( TTree*, edm::Service<TFileService>& );
    void branchesPFCandsAtECALstitched   ( TTree*, edm::Service<TFileService>& );
    void branchesTRKlayersAtEBEE( TTree*, edm::Service<TFileService>& );
    //void branchesTRKlayersAtECAL( TTree*, edm::Service<TFileService>& );
    void branchesTRKvolumeAtEBEE( TTree*, edm::Service<TFileService>& );
    //void branchesTRKvolumeAtECAL( TTree*, edm::Service<TFileService>& );
    void branchesJetInfoAtECALstitched   ( TTree*, edm::Service<TFileService>& );
    void branchesPFEB             ( TTree*, edm::Service<TFileService>& );
    void branchesPFHBHE           ( TTree*, edm::Service<TFileService>& );
    void branchesGsfTracksAtEBEE           ( TTree*, edm::Service<TFileService>& );

    bool runEvtSel          ( const edm::Event&, const edm::EventSetup& );
    bool runEvtSel_jet      ( const edm::Event&, const edm::EventSetup& );
    void fillEB             ( const edm::Event&, const edm::EventSetup& );
    void fillEE             ( const edm::Event&, const edm::EventSetup& );
    void fillES             ( const edm::Event&, const edm::EventSetup& );
    //void fillESatEE         ( const edm::Event&, const edm::EventSetup& );
    void fillHBHE           ( const edm::Event&, const edm::EventSetup& );
    void fillECALatHCAL     ( const edm::Event&, const edm::EventSetup& );
    void fillECALstitched   ( const edm::Event&, const edm::EventSetup& );
    void fillHCALatEBEE     ( const edm::Event&, const edm::EventSetup& );
    void fillTracksAtEBEE   ( const edm::Event&, const edm::EventSetup& );
    void fillTracksAtECALstitched   ( const edm::Event&, const edm::EventSetup& );
    void fillPFCandsAtEBEE   ( const edm::Event&, const edm::EventSetup& );
    void fillPFCandsAtECALstitched   ( const edm::Event&, const edm::EventSetup& );
    void fillTRKlayersAtEBEE( const edm::Event&, const edm::EventSetup& );
    //void fillTRKlayersAtECAL( const edm::Event&, const edm::EventSetup& );
    void fillTRKvolumeAtEBEE( const edm::Event&, const edm::EventSetup& );
    //void fillTRKvolumeAtECAL( const edm::Event&, const edm::EventSetup& );
    void fillJetInfoAtECALstitched   ( const edm::Event&, const edm::EventSetup& );
    void fillPFEB             ( const edm::Event&, const edm::EventSetup& );
    void fillPFHBHE           ( const edm::Event&, const edm::EventSetup& );
    void fillGsfTracksAtEBEE  ( const edm::Event&, const edm::EventSetup& );
    void TrackMatching ( const edm::Event& iEvent, const edm::EventSetup& iSetup );

    const reco::PFCandidate* getPFCand(edm::Handle<PFCollection> pfCands, float eta, float phi, float& minDr, bool debug = false);
    const reco::Track* getTrackCand(edm::Handle<reco::TrackCollection> trackCands, float eta, float phi, float& minDr, bool debug = false);
    int   getTruthLabel(const reco::PFJetRef& recJet, edm::Handle<reco::GenParticleCollection> genParticles, float dRMatch = 0.4, bool debug = false);
    std::pair<int, reco::GenTau*>  getTruthLabelForTauJets(const reco::PFJetRef& recJet, edm::Handle<reco::GenParticleCollection> genParticles, edm::Handle<reco::GenJetCollection> genJets, double magneticField, float dRMatch = 0.4, bool debug = false);
    float getBTaggingValue(const reco::PFJetRef& recJet, edm::Handle<edm::View<reco::Jet> >& recoJetCollection, edm::Handle<reco::JetTagCollection>& btagCollection, float dRMatch = 0.1, bool debug= false );
    math::XYZVector GetPi0Direction(math::XYZPoint vertex, double releta, double relphi, double seedeta, double seedphi);

    // Jet level functions
    std::string mode_;  // EventLevel / JetLevel
    bool doJets_;
    int  nJets_;
    double minJetPt_;
    double maxJetEta_;
    double z0PVCut_;
    std::vector<int> vJetIdxs;
    void branchesEvtSel_jet_dijet      ( TTree*, edm::Service<TFileService>& );
    void branchesEvtSel_jet_dijet_gg_qq( TTree*, edm::Service<TFileService>& );
    void branchesEvtSel_jet_taujet      ( TTree*, edm::Service<TFileService>& );
    bool runEvtSel_jet_dijet      ( const edm::Event&, const edm::EventSetup& );
    bool runEvtSel_jet_dijet_gg_qq( const edm::Event&, const edm::EventSetup& );
    bool runEvtSel_jet_taujet      ( const edm::Event&, const edm::EventSetup& );
    void fillEvtSel_jet_dijet      ( const edm::Event&, const edm::EventSetup& );
    void fillEvtSel_jet_dijet_gg_qq( const edm::Event&, const edm::EventSetup& );
    void fillEvtSel_jet_taujet      ( const edm::Event&, const edm::EventSetup& );


    int nTotal, nPassed;

    std::vector<std::pair<PtEtaPhiELV, std::vector<reco::CandidatePtr>>> HPSGammas(std::vector<reco::CandidatePtr> cands) const;
    PtEtaPhiELV getPi0(std::vector<reco::CandidatePtr> gammas, bool leadEtaPhi) const;
    std::pair<PtEtaPhiELV, PtEtaPhiELV> getRho(const pat::Tau tau, double gammas_pt_cut) const; 
    std::pair<std::vector<PtEtaPhiELV>, PtEtaPhiELV> getA1(const pat::Tau tau, float gammas_pt_cut) const;

}; // class RecHitAnalyzer

//
// constants, enums and typedefs
//
//static const bool debug = true;
static const bool debug = false;

static const int nEE = 2;
static const int nES = 2;
static const int nTOB = 6;
static const int nTEC = 9;
static const int nTIB = 4;
static const int nTID = 3;
static const int nBPIX = 4;
static const int nFPIX = 3;

static const int EB_IPHI_MIN = EBDetId::MIN_IPHI;//1;
static const int EB_IPHI_MAX = EBDetId::MAX_IPHI;//360;
static const int EB_IETA_MIN = EBDetId::MIN_IETA;//1;
static const int EB_IETA_MAX = EBDetId::MAX_IETA;//85;
static const int EE_MIN_IX = EEDetId::IX_MIN;//1;
static const int EE_MIN_IY = EEDetId::IY_MIN;//1;
static const int EE_MAX_IX = EEDetId::IX_MAX;//100;
static const int EE_MAX_IY = EEDetId::IY_MAX;//100;
static const int EE_NC_PER_ZSIDE = EEDetId::IX_MAX*EEDetId::IY_MAX; // 100*100
static const int HBHE_IETA_MAX_FINE = 20;
static const int HBHE_IETA_MAX_HB = hcaldqm::constants::IETA_MAX_HB;//16;
static const int HBHE_IETA_MIN_HB = hcaldqm::constants::IETA_MIN_HB;//1
static const int HBHE_IETA_MAX_HE = hcaldqm::constants::IETA_MAX_HE;//29;
static const int HBHE_IETA_MAX_EB = hcaldqm::constants::IETA_MAX_HB + 1; // 17
static const int HBHE_IPHI_NUM = hcaldqm::constants::IPHI_NUM;//72;
static const int HBHE_IPHI_MIN = hcaldqm::constants::IPHI_MIN;//1;
static const int HBHE_IPHI_MAX = hcaldqm::constants::IPHI_MAX;//72;
static const int ECAL_IETA_MAX_EXT = 140;

static const int ES_MIN_IX = ESDetId::IX_MIN;
static const int ES_MIN_IY = ESDetId::IY_MIN;
static const int ES_MAX_IX = ESDetId::IX_MAX;
static const int ES_MAX_IY = ESDetId::IY_MAX;
static const int ES_NC_PER_ZSIDE = ESDetId::IX_MAX*ESDetId::IY_MAX;

static const float zs = 0.;

// EE-(phi,eta) projection eta edges
// These are generated by requiring 5 fictional crystals
// to uniformly span each HCAL tower in eta (as in EB).
static const double eta_bins_EEm[5*(hcaldqm::constants::IETA_MAX_HE-1-HBHE_IETA_MAX_EB)+1] =
                  {-3.    , -2.93  , -2.86  , -2.79  , -2.72  , -2.65  , -2.62  ,
                   -2.59  , -2.56  , -2.53  , -2.5   , -2.4644, -2.4288, -2.3932,
                   -2.3576, -2.322 , -2.292 , -2.262 , -2.232 , -2.202 , -2.172 ,
                   -2.1462, -2.1204, -2.0946, -2.0688, -2.043 , -2.0204, -1.9978,
                   -1.9752, -1.9526, -1.93  , -1.91  , -1.89  , -1.87  , -1.85  ,
                   -1.83  , -1.812 , -1.794 , -1.776 , -1.758 , -1.74  , -1.7226,
                   -1.7052, -1.6878, -1.6704, -1.653 , -1.6356, -1.6182, -1.6008,
                   -1.5834, -1.566 , -1.5486, -1.5312, -1.5138, -1.4964, -1.479 }; // 56
// EE+(phi,eta) projection eta edges
static const double eta_bins_EEp[5*(hcaldqm::constants::IETA_MAX_HE-1-HBHE_IETA_MAX_EB)+1] =
                   {1.479 ,  1.4964,  1.5138,  1.5312,  1.5486,  1.566 ,  1.5834,
                    1.6008,  1.6182,  1.6356,  1.653 ,  1.6704,  1.6878,  1.7052,
                    1.7226,  1.74  ,  1.758 ,  1.776 ,  1.794 ,  1.812 ,  1.83  ,
                    1.85  ,  1.87  ,  1.89  ,  1.91  ,  1.93  ,  1.9526,  1.9752,
                    1.9978,  2.0204,  2.043 ,  2.0688,  2.0946,  2.1204,  2.1462,
                    2.172 ,  2.202 ,  2.232 ,  2.262 ,  2.292 ,  2.322 ,  2.3576,
                    2.3932,  2.4288,  2.4644,  2.5   ,  2.53  ,  2.56  ,  2.59  ,
                    2.62  ,  2.65  ,  2.72  ,  2.79  ,  2.86  ,  2.93  ,  3.    }; // 56

// HBHE eta bin edges
static const double eta_bins_HBHE[2*(hcaldqm::constants::IETA_MAX_HE-1)+1] =
                  {-3.000, -2.650, -2.500, -2.322, -2.172, -2.043, -1.930, -1.830, -1.740, -1.653, -1.566, -1.479, -1.392, -1.305,
                   -1.218, -1.131, -1.044, -0.957, -0.870, -0.783, -0.695, -0.609, -0.522, -0.435, -0.348, -0.261, -0.174, -0.087, 0.000,
                    0.087,  0.174,  0.261,  0.348,  0.435,  0.522,  0.609,  0.695,  0.783,  0.870,  0.957,  1.044,  1.131,  1.218,
                    1.305,  1.392,  1.479,  1.566,  1.653,  1.740,  1.830,  1.930,  2.043,  2.172,  2.322,  2.500,  2.650,  3.000}; // 57
//
// static data member definitions
//

template<class T>
math::XYZVector ExtrapolateToECAL(T part, double magneticField) {
  BaseParticlePropagator propagator = BaseParticlePropagator(
  RawParticle(part->p4(), math::XYZTLorentzVector(part->vx(), part->vy(), part->vz(), 0.),
              part->charge()),0.,0.,magneticField);

  propagator.propagateToEcalEntrance(false); // propogate to ECAL entrance
  math::XYZVector position = propagator.particle().vertex().Vect();
  return position;
}


#endif
