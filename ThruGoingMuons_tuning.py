print('\nLoading packages...\n')
import os
import uproot
import numpy as np
import pandas as pd
from tqdm import trange
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
font = {'family' : 'serif', 'size' : 10 }
mpl.rc('font', **font)
mpl.rcParams['mathtext.fontset'] = 'cm' # Set the math font to Computer Modern
mpl.rcParams['legend.fontsize'] = 1

TEST_RUN = True    # True = test the script by running over a much smaller sample of the data (to make sure there are no bugs)
                   # False = run over everything

# ----------------------------------------------------- #

''' Selection criteria for throughgoing muons
    - Only 1 track in the MRD (signaling its a single muon)
    - TankMRDCoinc + Veto coincidence, telling us it originated upstream the detector and passed all the way through
    - Brighest cluster
    - Very high energy cluster (1000 < cluster pe < 6000) -- this can be restricted further
    - charge distribution is focused, with almost every PMT seeing some level of light (charge balance < 0.2)
    - beam-correlated, cluster time within spill window
    - charge barycenter downstream
    - (optional, not employed for this script) "throughgoing" track, in which all layers of the MRD are hit
                + this is arguably the biggest selection you *should* include, but it was found to barely affect
                  the overall charge distribution, and severely reduce statistics (by a factor of 2-3), so i omitted it
'''

# throughgoing muons
def throughgoing(T1,TMRD1,NV1,B1,CPE1,CCB1,CT,hitZ,hitPE):
    if(T1==0):    # 1 track (1st because this is the strictest selection criteria)
        return False
    if(TMRD1==0): # TankMRDCoinc
        return False
    if(NV1==1):   # NoVeto
        return False
    if(B1==0):    # Brightest
        return False
    if(CPE1<1000 or CPE1>6000):  # 1000 < cluster PE < 6000
        return False
    if(CCB1>0.2 or CCB1<0): # Cluster Charge Balance < 0.2
        return False
    if(CT>1800 or CT<200):  # inside spill window
        return False
    a = 0
    for i in range(len(hitZ)):  # charge barycenter downstream
        a += hitZ[i]*hitPE[i]
    if(a<0):
        return False
    
    return True

# ----------------------------------------------------- #

# Data arrays
cluster_time = []; cluster_pe = []; hitID = []; hitPE = []; hitT = []; thru = []; cluster_qb = []

# ----------------------------------------------------- #
# ----------------------------------------------------- #
# ----------------------------------------------------- #

directory = 'data/'    # change if needed, build file names based on whats present in that directory

file_names = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

# fetch channel information for downstream/upstream information, and if we want to do channel-by-channel tuning
df = pd.read_csv('lib/FullTankPMTGeometry.csv')
loc = []; ch_num = []; downstream = []; pmt_type = []
for i in range(len(df['channel_num'])):
    ch_num.append(df['channel_num'][i])
    loc.append(df['detector_tank_location'][i])
    downstream.append(df['z_pos'][i]-1.681)       # transform geometric center to the center of the water tank
    pmt_type.append(df['PMT_type'][i])


print('There are: ', len(file_names), ' files\n')
print('This may take a while... time for a coffee break!\n')
print('\n(Please say goodbye to our heroes and close the window to continue the script)\n')
img = plt.imread("lib/CoffeeBreak.jpg")
plt.imshow(img)
plt.axis("off")
plt.show()

print('Proceeding with script...')

if TEST_RUN == True:
    print('\n\n####### TEST RUN ENABLED - ONLY RUNNING SCRIPT OVER SMALL DATASET #######')
    print('#################### WARNING: PLOTS MAY LOOK JANKY! #####################\n')

counter = 0
for file_name in file_names:
    with uproot.open(directory + file_name) as file:

        if TEST_RUN == True:
            if counter > 20:
                break
        
        print('\nRun: ', file_name, '(', (counter), '/', len(file_names), ')')
        print('------------------------------------------------------------')
    
        Event = file["data"]
        CT2 = Event["cluster_time_BRF"].array()     # BRF-corrected timing; you can opt to use the normal cluster time
        CPE1 = Event["cluster_PE"].array()
        CCB1 = Event["cluster_Qb"].array()
        NOC1 = Event["number_of_clusters"].array()
        CH1 = Event["cluster_Hits"].array()
        B1 = Event["isBrightest"].array()
        TMRD1 = Event["TankMRDCoinc"].array()
        MRD_yes = Event["MRD_activity"].array()
        T1 = Event["MRD_Track"].array()
        NV1 = Event["NoVeto"].array()
        HZ1 = Event['hitZ'].array()
        HPE1 = Event['hitPE'].array()
        HT1 = Event['hitT'].array()
        HID1 = Event['hitID'].array()
        TG1 = Event['MRDThrough'].array()

        counter += 1

        events_per_run = 0
        for i in trange(len(CT2)):

            is_thru = throughgoing(T1[i],TMRD1[i],NV1[i],B1[i],CPE1[i],CCB1[i],CT2[i],HZ1[i],HPE1[i])
            if(is_thru==False):
               continue

            cluster_time.append(CT2[i])
            cluster_pe.append(CPE1[i])
            hitID.append(HID1[i])
            hitPE.append(HPE1[i])
            hitT.append(HT1[i])
            thru.append(TG1[i])
            
            cluster_qb.append(CCB1[i])
            
            events_per_run += 1

        print(events_per_run, ' events')


print('\nAfter selection cuts, we have: ', len(cluster_time), ' thru-going muon candidates\n\n')

################################################################################################################################
# For tuning, breakup this first section into a jupyter notebook, then do your tuning separately after you've loaded the data
################################################################################################################################

# any additional event selection

# for tuning comparisons, we can break the charge up into upstream, downstream, and total

up_down_ratio = []; up_total = []; down_total = []
charge_per_PMT = [[] for i in range(332, 463+1)]       # for PMT-by-PMT tuning

upstream_timez = []; downstream_timez = []
upstream_chargez = []; downstream_chargez = []

for i in trange(len(cluster_pe), desc = 'applying additional selection cuts'):
    
    # additional selection cuts (MRD inefficiencies likely contribute to badd agreement in the lower charge region)
    if cluster_qb[i] < 0.17 and cluster_pe[i] > 3000:
        #if thru[i] == 1:     # throughgoing, can enable this if you want

        up_charge = 0; down_charge = 0
        for k in range(len(hitPE[i])):
            indy = int(hitID[i][k] - 332)    # grab channelkey of hit
            if 350 > hitPE[i][k] > 0:        # neglect any negative hit charges and limit saturation effects

                charge_per_PMT[indy].append(hitPE[i][k])

                if downstream[indy] < 0:
                    up_charge += hitPE[i][k]
                    upstream_timez.append(hitT[i][k] - min(hitT[i])) 
                    upstream_chargez.append(hitPE[i][k]) 
                else:
                    down_charge += hitPE[i][k]
                    downstream_timez.append(hitT[i][k] - min(hitT[i])) 
                    downstream_chargez.append(hitPE[i][k]) 

        # up/down ratio
        if up_charge != 0:
            up_down_ratio.append(up_charge/down_charge)
            up_total.append(up_charge); down_total.append(down_charge)
            

print('\n', len(up_down_ratio), 'throughgoing events\n')


# ----------------------------------------------------- #
# ----------------------------------------------------- #
# ----------------------------------------------------- #

# Now we load in the MC samples

print('\nLoading MC samples....\n')

'''
This WCSim dataset contain throughgoing muons, with energies, vertices, and angular distributions
taken directly from the GENIE world samples ()
Default QE = 1.0 and default reflectivities are set. 
Changes are up to date with WCSim (as of Jan 2025) https://github.com/ANNIEsoft/WCSim
EXCEPT that the QE ratio is set to 1.0 to perform the tuning

grid scripts for producing the samples: https://github.com/S81D/grid_wcsim/tree/main
GENIE throughgoing muon parameters can be found in that repository: genie_muons/thru_genie_muons.txt

As each event was taken directly a GENIE event, the WCSim.mac file has to be dynamically created 100 events at a time.
Thus each job consisted of 100 events each. The MC samples produced from the BeamClusterAnalysis toolchain were produced
PRIOR to knowing we could just sample* the file list and produce a single ntuple :/ hence why there are so many damn files.

* details of the WCSim.mac file used to generate the 25,000 events:

#/mygen/generator gps
#/gps/particle mu-
### insert GENIE properties
#/gps/energy <energy> MeV
#/gps/direction <unit_dir_x> <unit_dir_y> <unit_dir_z>
#/gps/pos/centre <vx> <vy> <vz> cm
#/gps/ang/rot1 -1 0 0
#/gps/ang/rot2 0 1 0
#/run/beamOn 1
## repeat for each event...
'''


directory = 'MC/'
file_list = os.listdir(directory)

class Cluster_MC:
    def __init__(self,pe,time,Qb,hitX,hitY,hitZ,hitT,hitPE,hitID,coinc,noveto,numTracks,thru_going):
        self.pe = pe
        self.time = time
        self.Qb = Qb
        self.hitX = hitX
        self.hitY = hitY
        self.hitZ = hitZ
        self.hitT = hitT
        self.hitPE = hitPE
        self.hitID = hitID
        self.coinc = coinc
        self.noveto = noveto
        self.numTracks = numTracks
        self.thru_going = thru_going
        
clusters_MC = []

count = 0; counter = 0
for file in trange(len(file_list)):

    if TEST_RUN == True:
        if counter > 2000:
            break
    counter += 1

    # default BeamClusterAnalysisMC structure
    root = uproot.open(directory + file_list[file])
    T = root['phaseIITankClusterTree']
    CEN = T['eventNumber'].array()
    CPE = T['clusterPE'].array()
    CCB = T['clusterChargeBalance'].array()
    CT = T['clusterTime'].array()
    X1 = T['hitX'].array()
    Y1 = T['hitY'].array()
    Z1 = T['hitZ'].array()
    T1 = T['hitT'].array()
    PE1 = T['hitPE'].array()
    ID1 = T['hitDetID'].array()
    TMC = T['TankMRDCoinc'].array()
    NV1 = T['NoVeto'].array()

    # truth information
    truth = root['phaseIITriggerTree']
    mtr = truth['numMRDTracks'].array()
    mrd_tg = truth['MRDThrough'].array()

    for i in range(len(CEN)):

        event = Cluster_MC(pe=CPE[i],time=CT[i],Qb=CCB[i],
                hitX=X1[i],hitY=Y1[i],hitZ=Z1[i],hitT=T1[i],hitPE=PE1[i],hitID=ID1[i],
                coinc=TMC[i],noveto=NV1[i],numTracks=mtr[CEN[i]],thru_going=mrd_tg[CEN[i]]
        )
        clusters_MC.append(event)

        count += 1


print('\n', count, 'MC throughgoing muon events\n')


# ----------------------------------------------------- #

# Now here is where we trim our sample and do any potential tuning

up_down_ratio_MC = []; up_total_MC = []; down_total_MC = []
charge_per_PMT_MC = [[] for i in range(332, 463+1)]

upstream_timez_mc = []; downstream_timez_mc = []
upstream_chargez_mc = []; downstream_chargez_mc = []

# similar to the Michels, we can apply a corrective QE factor to each individual hit to avoid re-producing samples with different QE ratios
corrective_factor = 1.025

for i in trange(len(clusters_MC), desc = 'Applying QE tuning factor and performing selection cuts...'):

    # selection cuts
    if clusters_MC[i].Qb < 0.17 and 3000 < clusters_MC[i].pe < 6000:
        #if clusters_MC[i].numTracks == 1:   # not much of an impact if we use this, as the muons taken from GENIE were selected to pass through MRD geometry
        if clusters_MC[i].coinc == 1 and clusters_MC[i].noveto == 0:
            a = 0
            for ix in range(len(clusters_MC[i].hitZ)):
                a += clusters_MC[i].hitZ[ix]*clusters_MC[i].hitPE[ix]
            if a > 0:

                # upstream/downstream
                up_charge_mc = 0; down_charge_mc = 0
                for k in range(len(clusters_MC[i].hitPE)):
                    indy_mc = int(clusters_MC[i].hitID[k] - 332)
                    if 350 > clusters_MC[i].hitPE[k]*corrective_factor > 0:

                        #charge_per_PMT_MC[indy_mc].append(hitPE[i][k])

                        if downstream[indy_mc] < 0:
                            up_charge_mc += (clusters_MC[i].hitPE[k])*corrective_factor
                            upstream_timez_mc.append(clusters_MC[i].hitT[k] - min(clusters_MC[i].hitT)) 
                            upstream_chargez_mc.append(clusters_MC[i].hitPE[k]) 
                        else:
                            down_charge_mc += (clusters_MC[i].hitPE[k])*corrective_factor
                            downstream_timez_mc.append(clusters_MC[i].hitT[k] - min(clusters_MC[i].hitT)) 
                            downstream_chargez_mc.append(clusters_MC[i].hitPE[k]) 

                if up_charge_mc != 0:
                    up_down_ratio_MC.append(up_charge_mc/down_charge_mc)
                    up_total_MC.append(up_charge_mc); down_total_MC.append(down_charge_mc)


print('\n', len(up_down_ratio_MC), 'MC events after selection cuts\n')

# cumulative charge distributions
total = []
for i in range(len(up_total)):
    total.append(up_total[i] + down_total[i])
total_MC = []
for i in range(len(up_total_MC)):
    total_MC.append(up_total_MC[i] + down_total_MC[i])

# ----------------------------------------------------- #

# Lastly, plot MC vs data cumulative distributions (up, down, total, ratio)

# adjust as needed
dir_name = 'plots/'
pic_base_name = ' _ MC data agreement.png'
legend_label = 'MC EC = 1.025, hitPE < 350'
hist_MC_color = 'darkorange'


plot_type = ['total', 'downstream', 'upstream', 'fraction']


for which in plot_type:
    
    # Prepare samples with equal counts
    data_ = []; mc_ = []
    
    print('\n\n', which)
    
    if which == 'total':
        # All
        bin_size = 150
        binning = np.arange(2600,6000,bin_size)   # 1000:6000
        for i in range(len(total)):
            data_.append(total[i])
        for i in range(len(total_MC)):
            mc_.append(total_MC[i])
        title_str = 'MC / Data agreement of through-going muons'
        
    elif which == 'downstream':
        # downstream
        bin_size = 150
        binning = np.arange(1800,4500,bin_size)   # 700:4500
        for i in range(len(down_total)):
            data_.append(down_total[i])
        for i in range(len(down_total_MC)):
            mc_.append(down_total_MC[i])
        title_str = 'MC / Data agreement of through-going muons | downstream PMTs only'
            
    elif which == 'upstream':
        # upstream
        bin_size = 60
        binning = np.arange(500,1800,bin_size)   # 400:2000
        for i in range(len(up_total)):
            data_.append(up_total[i])
        for i in range(len(up_total_MC)):
            mc_.append(up_total_MC[i])
        title_str = 'MC / Data agreement of through-going muons | upstream PMTs only'
            
    elif which == 'fraction':
        # up/down ratio
        bin_size = 0.03   # 0.03
        binning = np.arange(0.15,0.75,bin_size)
        for i in range(len(up_down_ratio)):
            data_.append(up_down_ratio[i])
        for i in range(len(up_down_ratio_MC)):
            mc_.append(up_down_ratio_MC[i])
        title_str = 'MC / Data agreement of through-going muons | upstream / downstream charge ratio'
        

    count_factor = len(mc_)/len(data_)

    # get counts and bins from histograms
    counts_data, bins_data = np.histogram(data_, bins = binning)
    counts_mc, bins_mc = np.histogram(mc_, bins = binning)

    counts_mc = np.array(counts_mc) / count_factor

    fill_bins = []   # data and mc (it will be the same bins)
    skip = [-1]
    for i in range(len(counts_data)):
        if i not in skip:
            if counts_data[i] >= 10 and counts_mc[i] >= 10:
                fill_bins.append(bins_data[i])
            else:
                # note the end of the previous bin
                fill_bins.append(bins_data[i])
                sum_d = counts_data[i]; sum_mc = counts_mc[i]
                for j in range((i+1), len(counts_data)):
                    if (sum_d + counts_data[j]) >= 10 and (sum_mc + counts_mc[j]) >= 10:    # combine bins, make sure counts >= 5 for chisq fit
                        skip.append(j)                                                      # require 10 for a more valid comparison
                        break
                    else:
                        sum_d += counts_data[j]
                        sum_mc += counts_mc[j]
                        skip.append(j)


    bin_edges = fill_bins[:-1]
    counts_data, bins_data = np.histogram(data_, bins = fill_bins)
    counts_mc, bins_mc = np.histogram(mc_, bins = fill_bins)

    counts_mc = np.array(counts_mc) / count_factor

    # fill x-error array as the bin sizes have changed
    xwidth = []
    for i in range(len(fill_bins)-1):
        # custom xwidth for ratio plot
        if which == 'fraction':
            xwidth.append(bin_size/2)
        else:
            xwidth.append(int((fill_bins[i+1]-fill_bins[i])/2))
    

    # simple errors (sqrt of bin count)
    mc_errors = [np.sqrt(counts_mc[i])/count_factor for i in range(len(counts_mc))]   # errors for mc
    data_errors = [np.sqrt(counts_data[i]) for i in range(len(counts_data))]

    binscenters = np.array([0.5 * (bins_data[i] + bins_data[i + 1]) for i in range (len(bins_data)-1)])

    # bottom plot (MC/Data)
    mc_data = []; lower_error_bars = []; x_info = []
    for i in range(len(counts_data)):
        if counts_data[i] == 0:   # no point on plot if there is no data
            place_holder = True
        elif counts_mc[i] == 0:   # if MC counts = 0, put point at 0
            mc_data.append(0)
            lower_error_bars.append(0)
            x_info.append(binscenters[i])
        else:
            mc_data.append(counts_mc[i]/counts_data[i])
            lower_error_bars.append((counts_mc[i]/counts_data[i])*(data_errors[i]/counts_data[i] + \
                                                                  mc_errors[i]/counts_mc[i]) )
            x_info.append(binscenters[i])


    chi2 = np.sum(((np.array(counts_data) - np.array(counts_mc)) ** 2) / 
              (np.array(data_errors) ** 2 + np.array(mc_errors) ** 2))
    dof = np.count_nonzero(counts_data) - 1    # omit nonzero bins in case we re-binned above
    print('\nchisq =',chi2/dof,'\n')


    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])  # 2 rows, 1 column

    # Add subplots
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax1.set_title(title_str)
    
    if which == 'fraction':
        ax2.set_xlabel('fraction of upstream / downstream charge', fontsize = 12)
    else:
        ax2.set_xlabel('cluster charge [p.e.]', fontsize = 12)
        ax1.set_ylabel('clusters / ' + str(bin_size) + ' p.e. bin', fontsize = 12)
        
    ax2.set_ylabel('MC / Data', fontsize = 12)
    ax2.set_ylim([0,3])
    
    if which == 'total':
        # all
        ax1.set_xlim([2000,7000])    # 900:6000 if using the full energy selection (1000:6000)
        ax2.set_xlim([2000,7000])
    elif which == 'downstream':
        # downstream
        ax1.set_xlim([1500,5500])    # 500:6000 if using the full energy selection (1000:6000)
        ax2.set_xlim([1500,5500])
    elif which == 'upstream':
        # upstream
        ax1.set_xlim([300,2000])      # 0:2250 if using the full energy selection (1000:6000)
        ax2.set_xlim([300,2000])
    elif which == 'fraction':
        # up / down ratio
        ax1.set_xlim([0.05,0.8])
        ax2.set_xlim([0.05,0.8])


    ax1.errorbar(binscenters, counts_data, yerr = np.sqrt(counts_data), xerr = xwidth,
                 fmt = 'none', color = 'k', label = 'Data')
   
    ax1.hist(bin_edges, fill_bins, weights = counts_mc,
             histtype = 'step', color = hist_MC_color, label = legend_label, linewidth = 1.5)

    ax2.errorbar(x_info, mc_data, xerr = xwidth, yerr = lower_error_bars,
                 fmt = 'none', color = 'k')
    ax2.axhline(1, linestyle = 'dashed', color = 'k')

    ax1.text(0.7,0.55,r"$\chi^2$" + " / ndf = " + str(round(chi2,1)) + ' / ' + \
             str(int(dof)),size = 12,transform = ax1.transAxes)

    ax1.legend(fontsize = 12, frameon = False)

    plt.savefig(dir_name + which + pic_base_name, dpi=300,bbox_inches='tight',pad_inches=.3,facecolor = 'w')

    plt.show()

    print('\nPlot saved: ' + dir_name + which + pic_base_name)


print('\ndone\n')
