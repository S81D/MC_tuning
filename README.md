# MC_tuning
Tuning scripts used to align the charge response of WCSim simulated data to real ANNIE beam data, by using an effective QE factor applied to all hit charges in an event. Three datasets can be compared: Michel electrons - electrons produced from the decay of muons in the ANNIE water tank which originate in the dirt usptream the detector (dirt muons) - throughgoing muons - muons that are similarly produced upstream the detector, but travel through the ANNIE tank, exit, and pass through the MRD geometry - and neutron captures from a radioactive neutron source (AmBe). As part of the NC tuning, Michels and AmBe neutrons were thoroughly compared and WCSim was tuned to align the O(MeV) low-energy scale of the ANNIE Detector. See my (Steven Doran) dissertation for more details. Given the tilting uncertainties, throughgoing muons were compared but the tuning was conducted against AmBe neutron capture candidates, and validated against Michel data.

The following scripts and the data presented are the final, tuned (NC tuning) samples used for the NCQE cross section analysis.

In practice what is done is to take an un-tuned version of the simulation (say for relative collection efficiency i.e. $r_{CE} = 1.0$), and compare the charge, hit, or charge per hit distributions. As a proxy for $r_{CE}$, the total charge, charge per hit, or total hits can be multiplied by a scalar ($R$) to better align the MC with data and achieve the lowest $\chi^2$ value. The simulation samples are then reprocessed with an adjustment to the relative collection efficiency of $R \times r_{CE}$, and re-compared. Since the scalar $R$ does not map cleany onto a true adjustment of the relative collection efficiency, this process is iteratively completed multiple times until the simulated cluster parameters are tuned against a particular dataset. Rather than re-simulating many events with small increments to the implemented collection efficiency (which takes time and a large amount of computational resources), the scalar factor applied directly to the cluster parameters provides a faster approximation that can then be fine-tuned.

Following the tuning of the simulation, an independent dataset can be compared to validate the response.

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
- Throughgoing WCSim files: `/pnfs/annie/persistent/users/doran/WCSim_thru_samples/PMT_tilt/QE_1.5_HM_1.25/`
- Throughgoing BeamCluster ntuples: `/exp/annie/data/users/doran/WCSim_thru_muons/corrected_waveform/QE_1.50_HM_1.25/`


### Usage
`AmBe_tuning.ipynb` - the tuning script I used to compare the AmBe simulated samples to the calibration data. This particular implementation (and the files outlined above) should be interpreted as the tuned version of the simulation, and thus the cluster parameters agree closely. The machinery can be adjusted if re-tuning from scratch.

`tuning_MC_Data_agreement.ipynb` - consolidated tuning script for all three datasets that shows the MC / Data agreement for AmBe, Michel, and throughgoing muons. This is a condensed version of multiple scripts I used in the tuning analysis, filtered by Claude AI. I have verified everything seems to be working the same as the original machinery. Please refer to the `AmBe_tuning.ipynb` script as the authoritative MC/Data comparison tuning script. 

`create_datasets.ipynb` - can apply selection cuts to create a single root file for each tuning dataset. Used to produce the tuning data root file analyzed in the tuning scripts described above. 

These scripts can be modified to adjust the charge per hit and thus the total charge to re-tune the simulation. Right now they exist purely to compare the tuned response and create plots.


Run each jupyter cell, making sure to adjust paths accordingly. The script will first load in both the data and MC for each dataset (depending on the script), before performing a chi squared comparison and producing plots. Depending on the dataset, the script may take long to compile (~10 minutes max for a given cell).
