import FWCore.ParameterSet.Config as cms
process = cms.Process("MAIN")
import sys
# lost tracks imports:
import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.slimming.primaryVertexAssociation_cfi import primaryVertexAssociation
from PhysicsTools.PatAlgos.slimming.packedPFCandidates_cfi import packedPFCandidates


################################################################
# Read Options
################################################################
import FWCore.ParameterSet.VarParsing as parser
opts = parser.VarParsing ('analysis')

opts.register('file',
#'root://xrootd.unl.edu//store/mc/RunIISummer20UL18RECO/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/AODSIM/106X_upgrade2018_realistic_v11_L1v1-v1/260000/1880AB56-620B-2247-A593-BD239DC4E805.root',
'root://xrootd.unl.edu//store/mc/RunIISummer20UL18RECO/GluGluHToTauTau_M-125_TuneCP5_13TeV-amcatnloFXFX-pythia8/AODSIM/106X_upgrade2018_realistic_v11_L1v1-v1/2550000/99C007C4-6099-524F-9C01-20E7F819EA77.root',
parser.VarParsing.multiplicity.singleton,
parser.VarParsing.varType.string, "input file")
opts.register('globalTag', '106X_upgrade2018_realistic_v11_L1v1', parser.VarParsing.multiplicity.singleton,
    parser.VarParsing.varType.string, "global tag")
opts.register('isData', 0, parser.VarParsing.multiplicity.singleton,
    parser.VarParsing.varType.int, "Process as data?")

opts.parseArguments()
infile      = opts.file
isData      = opts.isData
tag         = opts.globalTag

print 'isData      : '+str(isData)
print 'globalTag   : '+str(tag)

################################################################
# Standard setup
################################################################

from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
from Configuration.ProcessModifiers.run2_miniAOD_UL_cff import run2_miniAOD_UL

process = cms.Process('PAT',Run2_2018,run2_miniAOD_UL)

process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('PhysicsTools.PatAlgos.slimming.metFilterPaths_cff')
process.load('Configuration.StandardSequences.PATMC_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string("EventTree.root"),
    closeFileFast = cms.untracked.bool(True)
)

################################################################
# Message Logging, summary, and number of events
################################################################
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.MessageLogger.cerr.FwkReport.reportEvery = 50

process.options   = cms.untracked.PSet(
    FailPath=cms.untracked.vstring("FileReadError"),
    wantSummary = cms.untracked.bool(True),
)

# run miniAOD tau sequences on the fly

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('--python_filename nevts:100'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Schedule definition
process.schedule = cms.Schedule()
process.schedule.associate(process.patTask)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

# customisation of the process.

from FWCore.ParameterSet.Utilities import convertToUnscheduled
process=convertToUnscheduled(process)

# customisation of the process.

# Automatic addition of the customisation function from PhysicsTools.PatAlgos.slimming.miniAOD_tools
from PhysicsTools.PatAlgos.slimming.miniAOD_tools import miniAOD_customizeAllMC

#call to customisation function miniAOD_customizeAllMC imported from PhysicsTools.PatAlgos.slimming.miniAOD_tools
process = miniAOD_customizeAllMC(process)

# End of customisation functions



################################################################
# Input files and global tags
################################################################
process.load("CondCore.CondDB.CondDB_cfi")
from CondCore.CondDB.CondDB_cfi import *

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(infile))
process.GlobalTag.globaltag = cms.string(tag)

process.options   = cms.untracked.PSet(
    FailPath=cms.untracked.vstring("FileReadError"),
    wantSummary = cms.untracked.bool(True),
)

## import charged hadrons - following the same code that is used for HPS
#from RecoTauTag.RecoTau.PFRecoTauChargedHadronProducer_cff import ak4PFJetsRecoTauChargedHadrons
#ak4PFJetsRecoTauChargedHadrons = ak4PFJetsRecoTauChargedHadrons.clone()

################################################################
# 
################################################################

process.recHitAnalyzer = cms.EDAnalyzer('RecHitAnalyzer'
    , reducedEBRecHitCollection = cms.InputTag('reducedEcalRecHitsEB')
    , reducedEERecHitCollection = cms.InputTag('reducedEcalRecHitsEE')
    , reducedESRecHitCollection = cms.InputTag('reducedEcalRecHitsES')
    , reducedHBHERecHitCollection = cms.InputTag('reducedHcalRecHits:hbhereco')
    , genParticleCollection = cms.InputTag('genParticles')
    , genJetCollection = cms.InputTag('ak4GenJetsNoNu')
    , gedPhotonCollection = cms.InputTag('gedPhotons')
    , ak4PFJetCollection = cms.InputTag('ak4PFJets')
    , trackRecHitCollection = cms.InputTag('generalTracks')
    , trackCollection = cms.InputTag("generalTracks")
    , vertexCollection = cms.InputTag("offlinePrimaryVertices")
    , pfCollection = cms.InputTag("particleFlow")
    , recoJetsForBTagging = cms.InputTag("ak4PFJetsCHS")
    , jetTagCollection    = cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags")
    , ipTagInfoCollection = cms.InputTag("pfImpactParameterTagInfos")
    , mode = cms.string("JetLevel")
    , PFEBRecHitCollection = cms.InputTag('particleFlowRecHitECAL:Cleaned')
    , PFHBHERecHitCollection = cms.InputTag('particleFlowRecHitHBHE:Cleaned')
    , gsfTracksCollection = cms.InputTag('electronGsfTracks')
    , slimmedTausCollection = cms.InputTag('slimmedTaus')

    # Jet level cfg
    , nJets = cms.int32(-1)
    , minJetPt = cms.double(14.)
    , maxJetEta = cms.double(2.5)
    , z0PVCut  = cms.double(1000000)
    )

process.recHitAnalyzerSequence = cms.Sequence(process.recHitAnalyzer)


process.p = cms.Path(
    process.recHitAnalyzerSequence
)

process.schedule += cms.Schedule(process.p)
