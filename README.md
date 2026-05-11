# MC_tuning
Tuning scripts used to align the charge response of WCSim simulated data to real ANNIE beam data, by using an effective QE factor applied to all hit charges in an event. Three datasets can be compared: Michel electrons - electrons produced from the decay of muons in the ANNIE water tank which originate in the dirt usptream the detector (dirt muons) - throughgoing muons - muons that are similarly produced upstream the detector, but travel through the ANNIE tank, exit, and pass through the MRD geometry - and neutron captures from a radioactive neutron source (AmBe). As part of the NC tuning, Michels and AmBe neutrons were thoroughly compared and WCSim was tuned to align the O(MeV) low-energy scale of the ANNIE Detector. See my (Steven Doran) dissertation for more details. Given the tilting uncertainties, throughgoing muons were compared but the tuning was conducted against AmBe neutron capture candidates, and validated against Michel data.

The following scripts and the data presented are the final, tuned (NC tuning) samples used for the NCQE cross section analysis.

### Datasets

Beam data was extracted by converting ProcessedData files into ntuples using the `BeamClusterAnalysis` [toolchain](https://github.com/ANNIEsoft/ToolAnalysis/tree/Application/configfiles/BeamClusterAnalysis). As these filesizes are rather large, another [script](https://github.com/S81D/BeamCluster_extract) was used to filter the information into smaller file sizes for each run. Data files used in this analysis can be found on the gpvms here:
- ProcessedData: `/pnfs/annie/persistent/processed/processingData_EBV2/processed_EBV2/`
- BeamCluster ntuples: `/pnfs/annie/persistent/processed/processingData_EBV2/BeamClusterTrees/`
- The AmBe neutron data filtered file was produced using [this workflow](https://github.com/S81D/AmBe_neutron_candidates) (`lib/neutron_candidates_v5.root`) and consists of data recorded during the AmBe v2.0 (summer-fall 2023) campaign (https://docs.google.com/spreadsheets/d/1xfZc4Mf8tUMylV1UllDURjIOhiAvIC-Maua3FkUIaBk/edit?gid=0#gid=0).
- The Michel and throughoing data were filtered from FY22-FY23 beam data (`lib/Michels_v1.root`, `lib/throughgoing_muons_v1.root`).

MC data were produced by running the `BeamClusterAnalysisMC` [toolchain](https://github.com/ANNIEsoft/ToolAnalysis/tree/Application/configfiles/BeamClusterAnalysisMC) on WCSim samples. Details on each sample production can be found in the python scripts, embedded as comments, and in my dissertation. 
- AmBe WCSim files: `/pnfs/annie/persistent/users/doran/WCSim_AmBe_samples/PMT_tilt/QE_1.50_HM_1.25/`.
- AmBe BeamCluster ntuples: `/exp/annie/data/users/doran/WCSim_AmBe_samples/BC_files/pmt_tilting_v1/QE_1.50_WB_ETEL_LUX_1.0_HM_WM_1.25_corrected_waveforms/`
- Michel WCSim files: `/pnfs/annie/persistent/users/doran/WCSim_dirt_samples/PMT_tilt/QE_1.50_HM_1.25/` and `/pnfs/annie/persistent/users/doran/WCSim_dirt_samples/PMT_tilt/QE_1.50_HM_1.25_2/` (more stats)
- Michel BeamCluster ntuples: `/exp/annie/data/users/doran/WCSim_dirt_muons/corrected_waveform/QE_1.50_HM_1.25/` and `/exp/annie/data/users/doran/WCSim_dirt_muons/corrected_waveform/QE_1.50_HM_1.25_2/`
- Throughgoin WCSim files: `/pnfs/annie/persistent/users/doran/WCSim_thru_samples/PMT_tilt/QE_1.5_HM_1.25/`
- Throughgoing BeamCluster ntuples: `/exp/annie/data/users/doran/WCSim_thru_muons/corrected_waveform/QE_1.50_HM_1.25/`

The Michel WCSim files ntuples can be found on the gpvms here: `/pnfs/annie/persistent/users/doran/WCSim_tuning_BC_root_files/`

WCSim samples used in this analysis can be found on the gpvms in the following locations:
- Michels: `/pnfs/annie/persistent/simulations/wcsim/wcsim_QE_retuning_michel/`
- Throughgoing Muons: `/pnfs/annie/persistent/simulations/wcsim/wcsim_QE_retuning_thru_going/`

### Usage
`python3 Michel_tuning.py`

`python3 ThruGoingMuons_tuning.py`

Each script loads many events and performs selection cuts, so the scripts take long to compile (~20 minutes each). These scripts were ported from a jupyter notebook; while the scripts run as intended and produce comparison plots between MC and Data, if you plan on doing active tuning / assesing the agreement between data and MC in a more rigorous way, it is recommended to load the data first (perhaps in a notebook), then play around with the charge tuning factor and plot production (to save time).

There is a variable set at the beginning of each script `TEST_RUN` - setting this to 'True' only runs the script over a small subset of the data and each script compiles in under 5 minutes (in case you want to test the scripts initially or are debugging). The scripts may look janky though :) 
