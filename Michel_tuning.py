print('\nLoading packages...\n')
import os
import uproot        
import numpy as np
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

''' Selection criteria for Michels
    1. Select mother dirt muon
        - Veto activity, but no MRD (or Tank/MRD coincidence)
        - Must be a secondary cluster (candidate for a Michel)
        - Must have an extended readout (so we don't bias our Michel sample towards earlier times within the 2us prompt window)
        - Must be the brightest cluster in the prompt window
        - Have at least 50 PMT hits
        - Large cluster charge, but upper limit to account for potential MRD inefficiencies (1000 < cluster PE < 4000) --> 4000 will limit muons from even reaching the MRD
        - Isotropic charge distribution (charge balance < 0.2) --> removes weird events with a ton of charge in one corner of the tank
        - Muon candidate is within spill window, correlated with beam activity
        - charge barycenter downstream (it is traveling in the beam direction) 

    2. Select Michel candidate
        - Must be between 200ns and 5us after the dirt muon (lifetime is 2us, cutoff at 5us to avoid afterpulsing effects that take place at 6us)
        - total charge less than 800 pe
        - at least 20 PMT hits --> helps centralize the vertices
        - charge balance < 0.2 to avoid highly concentrated charge that could be due to other particle types or noise effects
'''

# dirt muons
def dirt(MRD_yes,TMRD1,NV1,NOC1,EXT,B1,CH1,CPE1,CCB1,CT,hitZ,hitPE):
    if(MRD_yes==1):  # any MRD activity
        return False
    if(TMRD1==1):    # TankMRDCoinc
        return False
    if(NV1==1):      # NoVeto
        return False
    if(NOC1==1):     # can't be the only cluster
        return False
    if(EXT==0):      # must have extended window to look for michels
        return False
    if(B1==0):       # Brightest cluster
        return False
    if(CH1<50):      # at least 50 PMT hits
        return False
    if(CPE1<1000 or CPE1>4000):  # 1000 < cluster PE < 4000
        return False
    if(CCB1>0.2 or CCB1<0):      # Cluster Charge Balance < 0.5
        return False
    if(CT>1800 or CT<200):       # inside spill window
        return False
    a = 0
    for i in range(len(hitZ)):   # charge barycenter downstream
        a += hitZ[i]*hitPE[i]
    if(a<0):
        return False
    return True

def Michel(CT2,CPE1,CH1,CCB1):
    if(CT2>5000 or CT2<200):     # at least 200ns after prior cluster, and no more than 5us after
        return False
    if(CPE1<=0 or CPE1>=650):    # total charge less than 800pe
        return False
    if(CH1<20):                  # at least 20 PMT hits
        return False
    if(CCB1>=0.2 or CCB1<=0):    # CCB < 0.2
        return False
    return True

# ----------------------------------------------------- #

# Data arrays ([0] = all dirt muon candidates, [1] = Michel candidates, [2] = dirt muons with Michel candidates (to check any biases in dirt muon selection))

cluster_time = [[], [], []]; cluster_charge = [[], [], []]; cluster_QB = [[], [], []]; cluster_hits = [[], [], []];
hitID = [[], [], []]; hitPE = [[], [], []]; hitT = [[], [], []]

# ----------------------------------------------------- #
# ----------------------------------------------------- #
# ----------------------------------------------------- #

directory = 'data/'     # change if needed, build file names based on whats present in that directory
file_names = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

print('There are: ', len(file_names), ' files\n')
print('This may take a while... hold onto your butts\n')
print('\n(Please say goodbye to Sam L Jackson and close the window to continue the script)\n')
img = plt.imread("lib/SamLJackson.png")
plt.imshow(img)
plt.axis("off")
plt.show()

print('Proceeding with script...')

if TEST_RUN == True:
    print('\n\n####### TEST RUN ENABLED - ONLY RUNNING SCRIPT OVER SMALL DATASET #######')
    print('#################### WARNING: PLOTS MAY LOOK JANKY! #####################\n')

# Load the data
counter = 0
for file_name in file_names:
    with uproot.open(directory + file_name) as file:

        if TEST_RUN == True:
            if counter > 50:
                break
        
        print('\nRun: ', file_name, '(', (counter), '/', len(file_names), ')')
        print('------------------------------------------------------------')
    
        # extracted data taken from the BeamClusterAnalysis ntuples
        Event = file["data"]
        CT2 = Event["cluster_time"].array()
        CPE1 = Event["cluster_PE"].array()
        CCB1 = Event["cluster_Qb"].array()
        CN1 = Event["cluster_Number"].array()
        NOC1 = Event["number_of_clusters"].array()
        CH1 = Event["cluster_Hits"].array()
        B1 = Event["isBrightest"].array()
        TMRD1 = Event["TankMRDCoinc"].array()
        MRD_yes = Event["MRD_activity"].array()
        EXT = Event["hadExtended"].array()
        NV1 = Event["NoVeto"].array()
        HZ1 = Event['hitZ'].array()
        HPE1 = Event['hitPE'].array()
        HT1 = Event['hitT'].array()
        HID1 = Event['hitID'].array()

        counter += 1

        muons_per_run = 0; michels_per_run = 0;
        for i in trange(len(CT2)):

            # select dirt muon candidates
            is_dirt = dirt(MRD_yes[i],TMRD1[i],NV1[i],NOC1[i],EXT[i],B1[i],CH1[i],CPE1[i],CCB1[i],CT2[i],HZ1[i],HPE1[i])
            if(is_dirt==False):
               continue
            
            # now look for michel candidates
            is_michel = False
            for k in range(i+1, len(CT2)):
                if CN1[k] != 0:     # not a new event
                    adj_time = CT2[k] - CT2[i]
                    is_michel = Michel(adj_time,CPE1[k],CH1[k],CCB1[k])
                    if(is_michel==False):
                        continue
                    # michel candidate parameters
                    cluster_time[1].append(adj_time)   
                    cluster_charge[1].append(CPE1[k])
                    cluster_QB[1].append(CCB1[k])
                    cluster_hits[1].append(CH1[k])
                    hitID[1].append(HID1[k])
                    hitPE[1].append(HPE1[k])
                    hitT[1].append(HT1[k])
                    
                    michels_per_run += 1
                    
                else:
                    break
                    
            if is_michel == True:
                # dirt muons with michel candidate parameters
                cluster_time[2].append(CT2[i])   
                cluster_charge[2].append(CPE1[i])
                cluster_QB[2].append(CCB1[i])
                cluster_hits[2].append(CH1[i])
                hitID[2].append(HID1[i])
                hitPE[2].append(HPE1[i])
                hitT[2].append(HT1[i])
            
            muons_per_run += 1

            # all dirt muons candidates
            cluster_time[0].append(CT2[i])   
            cluster_charge[0].append(CPE1[i])
            cluster_QB[0].append(CCB1[i])
            cluster_hits[0].append(CH1[i])
            hitID[0].append(HID1[i])
            hitPE[0].append(HPE1[i])
            hitT[0].append(HT1[i])

        print(muons_per_run, ' dirt muons, ', michels_per_run, ' michels')


print('\nAfter selection cuts, we have: ', len(cluster_time[0]), ' dirt muon candidates\n')
print('\nAfter selection cuts, we have: ', len(cluster_time[1]), ' michel candidates\n\n')
        
################################################################################################################################
# For tuning, breakup this first section into a jupyter notebook, then do your tuning separately after you've loaded the data
################################################################################################################################

# any additional event selection

# for the michels we can further reduce other backgrounds by dropping the charge balance 
# and remove any potentially saturated hits (only really relevant for the thru-going muons)

ct = []; cc = []; ch = []; cb = []

for i in trange(len(cluster_time[1]), desc = 'Apply additional selection cuts'):
    
    if cluster_QB[1][i] < 0.18:
        # enable to limit saturation (last i checked there weren't any hits above this threshold)
        #a = 0;
        #for k in range(len(hitPE[1][i])):
         #   if 350 > hitPE[1][i][k] > 0:
          #      a += hitPE[1][i][k]
        ct.append(cluster_time[1][i])
        cc.append(cluster_charge[1][i])
        ch.append(cluster_hits[1][i])
        cb.append(cluster_QB[1][i])
        
print('\n', len(cc), 'events after further selection cuts\n')

# ----------------------------------------------------- #
# ----------------------------------------------------- #
# ----------------------------------------------------- #

# Now we load in the MC samples

print('\nLoading MC samples....\n')

'''
This WCSim dataset contain isotropic, uniformly distributed electrons
throughout the entire ANNIE tank with energies according to the analytical expectation. 
Default QE = 1.0 and default reflectivities are set. 
Changes are up to date with WCSim (as of Jan 2025) https://github.com/ANNIEsoft/WCSim
EXCEPT that the QE ratio is set to 1.0 to perform the tuning

grid scripts for producing the samples: https://github.com/S81D/grid_wcsim/tree/main

* details of the WCSim.mac file used to generate the 50,000 events:

#/mygen/generator gps
#/gps/particle e-
#/gps/pos/type Volume
#/gps/pos/shape Cylinder
#/gps/pos/radius 1.5 m   # 1.524 in WCSim::DetectorConfigs
#/gps/pos/halfz 1.95 m   # 1.98m in WCSim::DetectorConfigs
#/gps/pos/rot1 1 0 0
#/gps/pos/rot2 0 0 1
#/gps/pos/centre 0 -14.46 168.1 cm
#/gps/ang/type iso
#/gps/ene/type Arb
#/gps/ene/emspec 1
#/gps/hist/type energy
#/gps/hist/file michel_spectrum.dat
#/gps/hist/inter Lin
'''


file_list = ['lib/michels_full_volume_default.root']

# define a class to hold cluster information
class Cluster_MC:
    def __init__(self,pe,Qb,nhits,hitPE,hitID,vtx,vty,vtz):
        self.pe = pe
        self.Qb = Qb
        self.nhits = nhits
        self.hitPE = hitPE
        self.hitID = hitID
        self.vtx = vtx
        self.vty = vty
        self.vtz = vtz
        
clusters_MC = []

count = 0
for file in trange(len(file_list)):

    # default BeamClusterAnalysisMC structure
    root = uproot.open(file_list[file])
    T = root['phaseIITankClusterTree']
    CEN = T['eventNumber'].array()
    CPE = T['clusterPE'].array()
    CCB = T['clusterChargeBalance'].array()
    CH = T['clusterHits'].array()
    PE1 = T['hitPE'].array()
    ID1 = T['hitDetID'].array()
    
    # truth information
    truth = root['phaseIITriggerTree']
    vx = truth['trueVtxX'].array()
    vy = truth['trueVtxY'].array()
    vz = truth['trueVtxZ'].array()

    for i in trange(len(CPE)):

        # convert truth vertex units accodingly
        event = Cluster_MC(pe=CPE[i],Qb=CCB[i],nhits=CH[i],hitPE=PE1[i],hitID=ID1[i],
                           vtx=vx[CEN[i]]/100,vty=vy[CEN[i]]/100,vtz=vz[CEN[i]]/100)
        clusters_MC.append(event)

        count += 1

print('\n', count, 'MC michel events\n')


# ----------------------------------------------------- #

# Now here is where we trim our sample and do any potential tuning

# effective QE ratio        # alter this and re-run to investigate agreement with different QE ratios
corrective_factor = 1.075   # effective QE factor applied to all MC hits (saves time as to not having to reproduce samples with different QE ratios)

charge_mc = []; c1 = []; 

for i in trange(len(clusters_MC), desc = 'Applying QE tuning factor and performing selection cuts...'):
    
    vx1 = clusters_MC[i].vtx; vy1 = clusters_MC[i].vty; vz1 = clusters_MC[i].vtz

    # angular effects of the PMTs were found to be pretty poor (more prominent at low energies)
    # as a result, there is a large excess of events with low energy tnat is not present in the data
    # since the ang effects are currently not modeled in the sim well, to find better agreement with data (which doesn't see a huge spike towards low energy)
    # we can limit our distribution towards the PMT active volume

    # if needed, we can apply a GENIE-realistic distribution (modeled from the dirt muons) to be extra precise (code included in the repository)

    if np.sqrt( vx1**2 + vz1**2 ) < 1.1 and np.sqrt( vy1**2 ) < 1.5:
    
        # selection cuts similar to data (PMT hits cut can serve to remove excess events at low energy due to the poor ang acceptance modeling)
        # parametric model does not have a good agreement with data for the number of PMT hits per event, so i am leaving that selection cut out
        if clusters_MC[i].Qb < 0.18 and clusters_MC[i].pe < 650:# and clusters_MC[i].nhits > 50:

            # apply effective QE factor
            b = 0
            for k in range(len(clusters_MC[i].hitPE)):
                b += clusters_MC[i].hitPE[k]*corrective_factor

            charge_mc.append(clusters_MC[i].pe)     # default, 1.0
            c1.append(b)                            # with corrective QE factor


print('\n', len(charge_mc), 'MC events after selection cuts\n')

# ----------------------------------------------------- #

# Lastly, we can compare the cumulative charge distributions to the data to see the agreement / disagreement
print('\nPlotting MC / Data agreement with tuning factor: ' + str(corrective_factor))

bin_size = 15
binning = np.arange(50,650,bin_size)

EC_val = corrective_factor   # here for labeling

path_str = 'plots/Michels ' + str(EC_val) + ' _ MC data agreement.png'    # saved plot path

# # # # # # # # # # # # # # # # # # # # # # # # #

# Prepare samples with equal counts
data_ = []; mc_ = []
for i in range(len(cc)):
    data_.append(cc[i])
for i in range(len(c1)):
    mc_.append(c1[i])

count_factor = len(mc_)/len(data_)

# get counts and bins from histograms
counts_data, bins_data = np.histogram(data_, bins = binning)
counts_mc, bins_mc = np.histogram(mc_, bins = binning)

counts_data = [int(counts_data[i]*count_factor) for i in range(len(counts_data))]

fill_bins = []   # data and mc (it will be the same bins)
skip = [-1]
for i in range(len(counts_data)):
    if i not in skip:
        if counts_data[i] >= 5 and counts_mc[i] >= 5:
            fill_bins.append(bins_data[i])
        else:
            # note the end of the previous bin
            fill_bins.append(bins_data[i])
            sum_d = counts_data[i]; sum_mc = counts_mc[i]
            for j in range((i+1), len(counts_data)):
                if (sum_d + counts_data[j]) >= 5 and (sum_mc + counts_mc[j]) >= 5:    # combine bins to ensure we have at least 5 counts for a chisq comparison
                    skip.append(j)
                    break
                else:
                    sum_d += counts_data[j]
                    sum_mc += counts_mc[j]
                    skip.append(j)


bin_edges = fill_bins[:-1]
counts_data, bins_data = np.histogram(data_, bins = fill_bins)
counts_mc, bins_mc = np.histogram(mc_, bins = fill_bins)

counts_data = [int(counts_data[i]*count_factor) for i in range(len(counts_data))]

# fill x-error array as the bin sizes have changed
xwidth = []
for i in range(len(fill_bins)-1):
    xwidth.append(int((fill_bins[i+1]-fill_bins[i])/2))

# simple errors (sqrt of bin count)
mc_errors = [np.sqrt(counts_mc[i]) for i in range(len(counts_mc))]   # errors for mc
data_errors = [np.sqrt(counts_data[i])*count_factor for i in range(len(counts_data))]

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
        

obs = np.array([counts_mc, counts_data])
chi2, p, dof, ex = chi2_contingency(obs)
print('\nchisq =',chi2/dof,'\n')


fig = plt.figure(figsize=(8, 8))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])  # 2 rows, 1 column

# Add subplots
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

ax1.set_title('MC/ Data agreement of Michels')
ax2.set_xlabel('cluster charge [p.e.]', fontsize = 12)
ax1.set_ylabel('clusters / ' + str(bin_size) + ' p.e. bin', fontsize = 12)
ax2.set_ylabel('MC / Data', fontsize = 12)
ax2.set_ylim([0,2])
# all
ax1.set_xlim([0,700])
ax2.set_xlim([0,700])

ax1.errorbar(binscenters, counts_data, yerr = np.sqrt(counts_data), xerr = xwidth,
             fmt = 'none', color = 'k', label = 'Data')

ax1.hist(bin_edges, fill_bins, weights = counts_mc,
         histtype = 'step', color = 'dodgerblue', label = 'MC EC = ' + str(EC_val), linewidth = 1.5)

ax2.errorbar(x_info, mc_data, xerr = xwidth, yerr = lower_error_bars,
             fmt = 'none', color = 'k')
ax2.axhline(1, linestyle = 'dashed', color = 'k')

ax1.text(0.7,0.55,r"$\chi^2$" + " / ndf = " + str(round(chi2,1)) + ' / ' + \
         str(int(dof)),size = 12,transform = ax1.transAxes)

ax1.legend(fontsize = 12, frameon = False)

plt.savefig(path_str,dpi=300,bbox_inches='tight',pad_inches=.3,facecolor = 'w')

plt.show()

print('\nPlot saved: ' + path_str)
print('\ndone\n')
