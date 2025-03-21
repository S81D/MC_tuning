# MC_tuning
Tuning scripts used to align the charge response of WCSim simulated data to real ANNIE beam data, by using an effective QE factor applied to all hit charges in an event. Michel electrons - electrons produced from the decay of muons in the ANNIE water tank which originate in the dirt usptream the detector (dirt muons) - and throughgoing muons - muons that are similarly produced upstream the detector, but travel through the ANNIE tank, exit, and pass through the MRD geometry - were compared between data and MC to do the tuning.

### Datasets

Beam data was extracted by converting ProcessedData files into ntuples using the `BeamClusterAnalysis` [toolchain](https://github.com/ANNIEsoft/ToolAnalysis/tree/Application/configfiles/BeamClusterAnalysis). As these filesizes are rather large, another [script](https://github.com/S81D/BeamCluster_extract) was used to filter the information into smaller file sizes for each run. Data files used in this analysis can be found on the gpvms here: `/pnfs/annie/persistent/users/doran/datasets/2023_beamdata_v1.tar.gz`

MC data were produced by running the `BeamClusterAnalysisMC` [toolchain](https://github.com/ANNIEsoft/ToolAnalysis/tree/Application/configfiles/BeamClusterAnalysisMC) on WCSim samples. Details on each sample production can be found in the python scripts, embedded as comments. The ntuples can be found on the gpvms here: `/pnfs/annie/persistent/users/doran/WCSim_tuning_BC_root_files/`

WCSim samples used in this analysis can be found on the gpvms in the following locations:
- Michels: `/pnfs/annie/persistent/simulations/wcsim/wcsim_QE_retuning_michel/`
- Throughgoing Muons: `/pnfs/annie/persistent/simulations/wcsim/wcsim_QE_retuning_thru_going/`

### Usage
`python3 Michel_tuning.py`

`python3 ThruGoingMuons_tuning.py`

Each script loads many events and performs selection cuts, so the scripts take long to compile (~20 minutes each). These scripts were ported from a jupyter notebook; while the scripts run as intended and produce comparison plots between MC and Data, if you plan on doing active tuning / assesing the agreement between data and MC in a more rigorous way, it is recommended to load the data first (perhaps in a notebook), then play around with the charge tuning factor and plot production (to save time).

There is a variable set at the beginning of each script `TEST_RUN` - setting this to 'True' only runs the script over a small subset of the data and each script compiles in under 5 minutes (in case you want to test the scripts initially or are debugging). The scripts may look janky though :) 
