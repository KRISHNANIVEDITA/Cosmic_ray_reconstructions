"""
Author: @knivedita
Email: @krishna.gopinath@ru.nl
Created: 2026-01-28


Goal: Panel Reconstructions - Framework for core position and energy of CR shower for RET-CR 

Parts of this framework is adapted from previous LORA reconstructions

"""

import uproot
import numpy as np
import glob
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import math
import ROOT as r
from ROOT import TH2F, TF2, TF1, TH1F,TMath
from ROOT import TVirtualFitter
from matplotlib.backends.backend_pdf import PdfPages
import helper as helper
import re
import datetime
import os
import csv




"""
#*******************************************************************************************************************************************************#
    This part is for arrival direction calculations.
    
    Parameters:
    x,y -> positions of the station
    dt -> arrival times at the stations 
    length -> number of data points/stations
    
    Functions:

    Both 'fit_arrival_direction' as well 'arrival_dir_reco' calculates the direction with a plane wave fitting (both fitting and analytically).
    Choose any function. Currently arrival_dir_reco is used
#*******************************************************************************************************************************************************#
"""
    
def func_plane(x, par):
    return par[0] - par[1] * x[0] - par[2] * x[1]

def fit_arrival_direction(x, y, dt, length):
    plane = TH2F("Shower plane", "", 1000, -100, 100, 1000, -100, 100)
    cdt = dt * (1.0e-9 * 3 * 1.0e8)
    for i in np.arange(length):
        plane.SetBinContent(plane.GetXaxis().FindBin(x[i]), plane.GetYaxis().FindBin(y[i]), cdt[i])

    fit_plane = TF2("fit_plane",  "[0] - [1]*x - [2]*y", -100, 100, -100, 100)
    fit_plane.SetParameter(0, -100)
    fit_plane.SetParameter(1, 0.5)
    fit_plane.SetParameter(2, 0.5)
    fit_plane.SetParLimits(1, -1, 1)
    fit_plane.SetParLimits(2, -1, 1)
    plane.Fit(fit_plane, "Q0")  
    t0 = fit_plane.GetParameter(0)
    l = fit_plane.GetParameter(1)
    m = fit_plane.GetParameter(2)
    n = np.sqrt(1.0 - (l * l + m * m))
    err_l=fit_plane.GetParError(1)
    err_m=fit_plane.GetParError(2)
    theta = np.arcsin(np.sqrt(l * l + m * m)) * (180.0 / np.pi)
    phi = np.arccos(m / np.sqrt(l * l + m * m)) * (180.0 / np.pi)
    
    if l < 0:
        phi = 360.0 - phi
    df_f2=np.power((l*l)/(m*(l*l+m*m)),2)*np.power(err_m,2)+np.power(l/(l*l+m*m),2)*np.power(err_l,2)
    err_theta=np.sqrt(np.power(l*err_l/(n*n),2)+np.power(m*err_m/(n*n),2))/np.tan(theta*np.pi/180.0)
    err_theta=err_theta*(180.0/np.pi)
    err_phi=np.sqrt(df_f2/np.power(np.tan(phi*np.pi/180.0),2))
    err_phi=err_phi*(180.0/np.pi)
    print(theta,phi)
    return theta, phi, t0, l, m,err_phi,err_theta


def arrival_dir_reco(x,y,dt,length):
    #coord = df['coordinates']
    P=0
    Q=0
    R=0
    S=0
    W=0
    T1=0
    S1=0
    S2=0
    S3=0
    S4=0
    S5=0
    S6=0
    counter=0
    DT = [x for x in dt]
    print(DT)
    for i in np.arange(length):
        counter=counter+1
        S=S+x[i]**2
        W=W+y[i]**2
        T1=T1+x[i]
        S1=S1+x[i]*DT[i]*(1.e-9*3e8)
        S2=S2+x[i]*y[i]
        S3=S3+y[i]*DT[i]*(1.e-9*3e8)
        S4=S4+y[i]
        S5=S5+DT[i]*(1.e-9*3e8)
        S6=S6+1
    if counter>2:
        P=(S1*S2)/S
        Q=(T1*S2)/S
        R=-((S2*S2)/S)+W

        t0=(T1*S1*R-R*S*S5+T1*P*S2-T1*S2*S3-S*S4*P+S*S4*S3)/(T1*Q*S2-T1*S2*S4+R*T1*T1-S*S4*Q+S*S4*S4-R*S*S6)
        m=(P-t0*Q-S3+t0*S4)/R
        l=(-S1/S)-((P*S2)/(R*S))+((t0*Q*S2)/(R*S))+((S2*S3)/(R*S))-((t0*S2*S4)/(R*S))+((t0*T1)/S)
        #print(l,m)
        n=np.sqrt(1.0-(l*l+m*m))



        #print('direction params:  {0} {1}  {2}  {3}'.format(t0,m,l,n))

        if l*l+m*m<1:

            theta=(np.arcsin(np.sqrt(l*l+m*m)))*(180.0/np.pi)    #Zenith in degrees (from vertical direction +Z)
            phi=(np.arccos(m/np.sqrt(l*l+m*m)))*(180.0/np.pi)    #in degrees (Eastward from North)
        else:
            theta=0.0
            phi=(np.arccos(m/np.sqrt(l*l+m*m)))*(180.0/np.pi)    #in degrees (Eastward from North)


        if l<0:
            phi=360.0-phi
        elevation=90.0-theta
        #print('theta: {0:.2f}   phi: {1:.2f}    el: {2:.2f}'.format(theta,phi,elevation))
        return theta,phi,  t0, l, m 

"""
#*******************************************************************************************************************************************************#
    This part is for estimating the panel calibrations and getting the ADC counts
    
    Functions:

    1) getPanelTemp() calculates the temperature of the panel at the timestamp 
    Parameters:
    panel-> panel id
    radar_sec,radar_nsec->timestamps
    
    2)calculate_energy_deposit() calculates the energy in MeV from the ADC deposits 
    Required:
    We need panel calibration fits for this. The files for high gain ADC and medium gain ADC are stored in the /utilities/ folder
#*******************************************************************************************************************************************************#
"""
    

def getPanelTemp(panel, radar_sec, radar_nsec):
    datetime_object = datetime.datetime.fromtimestamp(radar_sec)
    #print(f'Getting temp from data for panel {panel} date {datetime_object}')
    #load in the panels root file to get the run number and panel number
    coincidentPanelDir = '/fs/project/PAS1968/retcr/data/2024/SUR/2024/root/coincidentOnly/'
    pChain = r.TChain("tree")
    fileBlank = '2024_panel'
    pChain.Add(f'{coincidentPanelDir}2024_panel{panel}.root')
    pChain.BuildIndex("centralCoincidence_utc_sec","centralCoincidence_utc_nsec")
    pChain.GetEntryWithIndex(radar_sec,radar_nsec)
    runNumber = pChain.runNumber
    station = int(re.findall(r'\d+', str(pChain.panelID))[0])
    #print(station)
    panelID = str(pChain.panelID)
    if 'a' in panelID:
        panelNumber = pChain.stationPanels[0]
    if 'b' in panelID:
        panelNumber = pChain.stationPanels[1]
    #open the panel run json to check if the temp is correct there
    runDir = '/fs/project/PAS1968/retcr/data/2024/SUR/2024/'
    runFileName = f'panel_{panelNumber}_run_{runNumber:07d}'
    fileName = f'{runDir}{station}/runs/2024-{datetime_object.month:02d}-{datetime_object.day:02d}/hitBufferRuns/{runFileName}/{runFileName}.json'
    if not os.path.isfile(fileName):
        print('file does not exist')
        return 0
    #print(fileName)
    try:
        with open(fileName) as jsonFile:
            inList = jsonFile.readlines()
            for i in range(len(inList)):
                line = inList[i].split(',')
                if 'temp' in inList[i]:
                    split = (line[0].split(':'))
                    temp = float(split[1])
                    break
        if (temp - 273.15) > -50:
            return temp - 273.15
        else:
            raise RuntimeError("Fake error for testing")
    except:
        # if the temp in the json is wrong too, just grab the nearest in-time valid temp
        fileDir = f'{runDir}{station}/runs/2024-{datetime_object.month:02d}-{datetime_object.day:02d}/hitBufferRuns/'
        dirlist = os.listdir(fileDir)
        panelRunList = []
        panelString = f'panel_{panelNumber}'
        for i in dirlist:
            if panelString in i:
                panelRunList.append(i)
        allTempList = []
        for n,i in enumerate(panelRunList):
            #print(i)
            temp = 0
            jsonFileName = f'{fileDir}{i}/{i}.json'
            try:
                with open(jsonFileName) as jsonFile:
                    inList = jsonFile.readlines()
                    for i in range(len(inList)):
                        line = inList[i].split(',')
                        if 'temp' in inList[i]:
                            split = (line[0].split(':'))
                            temp = float(split[1])
                            break
                allTempList.append(temp)
            except:
                #allTempList.append(-273.15)
                continue
        number = len(allTempList)
        if runNumber==len(allTempList):
            for i in range(1,100):
                if( allTempList[runNumber-i] !=0.0) and (runNumber-i) > 0:
                    return allTempList[runNumber-i] - 273.15
                if (runNumber-i) <0 and (runNumber+i) < len(allTempList) and  ( allTempList[runNumber+i] !=0.0):
                    return allTempList[runNumber+i] - 273.15
        else :
            for i in range(100):
                if( allTempList[runNumber-i] !=0.0) and (runNumber-i) > 0:
                    return allTempList[runNumber-i] - 273.15
                if (runNumber-i) <0 and (runNumber+i) < len(allTempList) and  ( allTempList[runNumber+i] !=0.0):
                    return allTempList[runNumber+i] - 273.15
        



def calculate_energy_deposit(event):
    energy_deposits = []
    means =[]
    event_trig = event[event['panel_sec']>0.0]
    event_trig['temperature'] = event_trig['temperature']+273.15
    #print(event_trig)
    for event_st in range(len(event_trig)):
        min_diff = 300.0
        panel_no_event = event_trig.iloc[event_st]['panels']
        given_temp = event_trig.iloc[event_st]['temperature']
        bias_volt = event_trig.iloc[event_st]['panel_bias_volt']
        with open('/users/PAS1968/knivedita/data_analysis/calibration/fit_results_high.txt', 'r') as file:
            next(file)
            for line in file:
                parts = line.strip().split()
                panel = int(parts[0])                 
                if panel == panel_no_event:
                    temp = float(parts[1]) 
                    slope = float(parts[2])
                    intercept = float(parts[3])
                    diff = abs(temp - given_temp)
                    if (panel == panel_no_event):
                        #print("row",panel_no_event, temp, slope, intercept)
                        if (diff < min_diff):  
                            min_diff = diff
                            closest_row = (panel_no_event, temp, slope, intercept)
        #print(closest_row)
        mean_high = closest_row[2]*bias_volt+closest_row[3]
        min_diff = 300.0
        with open('/users/PAS1968/knivedita/data_analysis/calibration/fit_high_mid.txt', 'r') as file_mid:
            next(file_mid)
            for line_mid in file_mid:
                parts = line_mid.strip().split(',')
                panel = int(parts[0])                 
                if panel == panel_no_event:
                    temp = float(parts[1]) 
                    slope = float(parts[2])
                    intercept = float(parts[3])
                    diff = abs(temp - given_temp)
                    if (diff < min_diff) & (panel == panel_no_event):
                        min_diff = diff
                        closest_row_hm = (panel_no_event, temp, slope, intercept)
        mean_mid = closest_row_hm[2]*mean_high+closest_row_hm[3]
        energy_w = 2.52 * event_trig.iloc[event_st]['adc_counts'] / mean_mid
        means.append(mean_mid)
        energy_deposits.append(energy_w)
    event_trig['energy_deposits(MeV)'] = energy_deposits
    event_trig['mean_mid_gain'] = means
    return event_trig


"""
#*******************************************************************************************************************************************************#
    Energy reconstruction function
#*******************************************************************************************************************************************************#
"""
def energy_reco(Nc,theta):
    """
    This function calculates the energy of the air shower from the panel deposits
    
    Parameters:
    
    Nc = Amplitude value from the NKG  fit 
    theta = zenith arrival angle of the shower

    Required:

    Simulated amplitude parameters from NKG fits with CORSIKA and Geant4 simulation -
    here we have file 'nkg_fits_predicted.csv' for IceTop scintillators for CR showers at summit
    """
    
    #print("The cvalue of shower",Ne_fit)
    C_shower = Nc
    nkg_results = pd.read_csv('/users/PAS1968/knivedita/reconstruction/nkg_fits_predicted.csv')
    # Extract data from DataFrame for interpolation
    energy_vals = np.power(10, nkg_results["Energy"].astype(float)).astype(np.float64)
    energy_vals=np.log10(energy_vals)
    zenith_vals = nkg_results["Zenith"].astype(float).values
    c_vals = nkg_results["C"].values *2*3.14*1600
    query_energy = np.linspace(energy_vals.min(), energy_vals.max(), 6)   # or directly: 17.2
    query_zenith = theta
    c_energies=[]
    for E in query_energy:
        #print('entering energies')
        C_interp = griddata(
            (energy_vals, zenith_vals),    # known points
            c_vals,                        # known values
            [(E, query_zenith)],# point to interpolate
            method='cubic'                 # same method as your grid
        )
        c_energies.append(C_interp)
        #print(f"Interpolated C at log10(Energy)={E:.2f}, Zenith={query_zenith}: {C_interp[0]}")
    coeffs = np.polyfit(query_energy, np.log10(c_energies), 1)  # returns [slope, intercept]
    slope, intercept = coeffs
    plt.scatter(query_energy,np.log10(c_energies),c='b')
    # Generate the fitted line
    energies = np.linspace(14.0, 20.0, 48) 
    C_fit = slope * np.array(energies) + intercept
    #plt.plot(energies, C_fit, 'r--', label='Fit: y = x + '.format(slope,intercept))
    log_C_shower = np.log10(C_shower)
    C_14 = slope * np.array(14.0) + intercept
    if log_C_shower < C_14:
        reco_energy_log10 = 10
    else :
        closest_idx = np.argmin(np.abs(C_fit - log_C_shower))
        reco_energy_log10 = ((log_C_shower - intercept)/slope)[0]
    reco_energy = 10**reco_energy_log10  # convert back from log10(E)
    print("RECONSTRUCTED ENERGY IS ::::::::  ", '10**{0}'.format(reco_energy_log10))
    return reco_energy_log10

"""
#*******************************************************************************************************************************************************#
    Main code FRAMEWORK
    Here an event timestamp is taken from the 10% sample(T0P,T1P...), and matched to get the corresponding event from the ROOT files.     
    A pandas dataframe is created for each event - Arrival direction, core position and energy is estimated 
#*******************************************************************************************************************************************************#
"""
"""
# Load the 10% sample 
"""
T1P5 = uproot.open("/fs/project/PAS1968/retcr/data/2024/root/datasets/tenPercent/T0P3A0H4TenPercent.root")
tree_t1p5 = T1P5["tree"]
utc_secs10 = tree_t1p5["utc_sec"].array(library='np')
utc_nsecs10 = tree_t1p5["utc_nsec"].array(library='np')

# Paths to load the data from the ROOT files
root_files = (
    glob.glob("/fs/project/PAS1968/retcr/data/2024/root/june0.root") +  glob.glob("/fs/project/PAS1968/retcr/data/2024/root/june1.root")+ glob.glob("/fs/project/PAS1968/retcr/data/2024/root/june2.root")+
    glob.glob("/fs/project/PAS1968/retcr/data/2024/root/july0.root") + glob.glob("/fs/project/PAS1968/retcr/data/2024/root/july1.root")+glob.glob("/fs/project/PAS1968/retcr/data/2024/root/july2.root")+
    glob.glob("/fs/project/PAS1968/retcr/data/2024/root/august*.root")
)
"""
# Define some important lists to store the information about the events
"""
matching_events = []
thetas=[]
azs=[]
theta_planes=[]
phi_planes =[]
theta_error = []
phi_error = []
panel_trig = []
xcores=[]
ycores = []
xerror = []
yerror =[]
nsecs_store = []
secs_store=[]

"""
# Define the station labels, the coordinates of the surface stations, and the panel numbers (same order as used in panel calibration data)
#IMPORTANT NOTE : Here the panel coordinates are swapped because there was a cable swap between the stations!!! (which was derived from arrival directions
"""
labels = [
        "SS1_scint1", "SS1_scint2", "SS2_scint1", "SS2_scint2", 
        "SS3_scint1", "SS3_scint2", "SS5_scint1", "SS5_scint2", 
        "SS6_scint1", "SS6_scint2"
        ]
panel_nos = [13,9, 8, 4, 2, 1, 6, 5, 7, 14]

coordinates_swap = [[-3.308,40.268],[11.547,39.701],[31.191,-28.257],[46.391,-15.1983],[-25.454,-31.413],[-35.032,-16.506],[41.206,50.988],[25.66,62.593],[-37.798,-58.042],[-20.423,-64.921]]

#coordinates = [[-3.308,40.268],[11.547,39.701], [46.391,-15.1983],[31.191,-28.257],[-25.454,-31.413],[-35.032,-16.506],[25.66,62.593],[41.206,50.988],[-20.423,-64.921], [-37.798,-58.042]]
coordinates = coordinates_swap

x_coords = [coord[0] for coord in coordinates]
y_coords = [coord[1] for coord in coordinates]
df = pd.DataFrame({'labels': labels,'coordinates':coordinates})
x = df['coordinates'].apply(lambda c: c[0])  # X-coordinates
y = df['coordinates'].apply(lambda c: c[1]) # Y-coordinates
z=np.zeros([len(x)])

"""
#LOAD THE DATA from the ROOT files and reconstructions below
"""
chain = r.TChain("tree")
for root_file in root_files:
    with uproot.open(root_file) as f:
        if "tree" in f:
            chain.Add(root_file



with open('/users/PAS1968/knivedita/reconstruction/wrapped/T0P3_results_1.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['loop_i','sec', 'nsec', 'theta', 'phi', 'rms_err', 'core_x', 'core_y', 'xerror', 'yerror','energylog','Ne_fit','Ne_err'])
    counter=0
    ### event 71 , 100
    for sec, nsec in zip(utc_secs10[169:170], utc_nsecs10[169:170]):  # All events in the 10% sample are run in a loop here 
        try:
            counter=counter+1
            print(sec, nsec)
            chain.BuildIndex("utc_sec", "utc_nsec")  # Build index using utc_sec and utc_nsec
            randomEntry = chain.GetEntryNumberWithIndex(int(sec), int(nsec))  # Cast to Python int
            if randomEntry >= 0:
                #print(randomEntry)
                chain.GetEntry(randomEntry)
                index = chain.index
                adc_counts = np.array([chain.panel_medium_gain[i] for i in range(12)])
                temperature = np.array([chain.panel_temp[i] for i in range(12)])
                #print("Original temperature:", temperature)

                # Make a copy and fix unrealistic temps
                temperature_copy = temperature.copy()
                for n, j in enumerate(temperature_copy):
                    if n == 6 or n == 7:
                        continue  # Skip indices 6 and 7
                    if j < -50:
                        temperature_copy[n] = getPanelTemp(n, int(sec), int(nsec))
                temperature = temperature_copy
                # Fetch other necessary info
                triggerinfo = np.full(12, chain.panel_trigger_info)
                panel_bias_volt = np.array([chain.panel_bias_volt[i] for i in range(12)])
                panel_sec = np.array([chain.panel_sec[i] for i in range(12)])
                times = np.array([chain.l0_dt[i] for i in range(12)])
                panel_volts = np.array([chain.panel_bias_volt[i] for i in range(12)])

                # Trim channels 6 and 7
                adc_counts = np.delete(adc_counts, [6, 7])
                panel_bias_volt = np.delete(panel_bias_volt, [6, 7])
                #print(temperature)
                temperature = np.delete(temperature, [6, 7])
                #print(temperature)
                times = np.delete(times, [6, 7])
                panel_sec = np.delete(panel_sec, [6, 7])
                panel_volts = np.delete(panel_volts, [6, 7])
                index = np.full(len(panel_sec), index)
                triggerinfo = np.full(len(panel_sec), triggerinfo[0])
                '''
                print("Length of labels:", len(labels))
                print("Length of panel_nos:", len(panel_nos))
                print("Length of x:", len(x))
                print("Length of y:", len(y))
                print("Length of z:", len(z))
                print("Length of index:", len(index))
                print("Length of triggerinfo:", len(triggerinfo))
                print("Length of panel_bias_volt:", len(panel_bias_volt))
                print("Length of temperature:", len(temperature))
                print("Length of adc_counts:", len(adc_counts))
                print("Length of times:", len(times))
                print("Length of panel_volts:", len(panel_volts))
                '''
                
                # Build event DataFrame
                
                event = pd.DataFrame({
                    'stations': labels,
                    'panels': panel_nos,
                    'x': x,
                    'y': y,
                    'z': z,
                    'index': index,
                    'triggerinfo': triggerinfo,
                    'panel_bias_volt': panel_bias_volt,
                    'temperature': temperature,
                    'adc_counts': adc_counts,
                    'times': times,
                    'panel_volts': panel_volts,
                    'panel_sec': panel_sec
                })

                ## IMPORTANT : Add the MeV energy deposits from the ADC Counts to the dataframe 'event_new' dataframe
                event_new = calculate_energy_deposit(event)
                check = (event_new['panel_sec'] > 0)
                X = np.array(event_new['x'])[check]
                Y = np.array(event_new['y'])[check]
                dt = np.array(event_new['times'])[check]
                length = len(dt)
                dt = dt + np.random.normal(loc=0, scale=11, size=length)
                #theta, phi, t0, l, m, phi_err, theta_err = fit_arrival_direction(X, Y, dt, length)
                
                ## Calling the function for Arrival direction measurements
                theta, phi, t0, l, m = arrival_dir_reco(X, Y, dt, length)
                dt_reco = t0 - l * X - m * Y
                rms_error = np.sqrt(np.mean((dt*0.3 - dt_reco)**2))
                theta_rad = np.radians(theta)
                phi_rad = np.radians(phi)
                rad = 10
                a = rad * np.sin(phi_rad)
                b = rad * np.cos(phi_rad)

                ## Calling the helper file for Core position reconstructions (Also have another helper which uses only python, contact if needed)
                Ne_fit,rM_fit,s_fit,real_x, real_y, realxerr, realyerr, Ne_err = helper.fit_NKG(theta_rad, phi_rad, event_new[check],100)
                Nc=Ne_fit
                theta_temp=theta

                ###Energy reconstructions begins here :
                ## CORSIKA+Geant4 NKG fits from simulation for upto 50 degree is in the nkg_fits.csv file
                if theta_temp > 50:
                    theta_temp = 50
                if theta_temp >= 0 and theta_temp <= 50:
                    reco_energy_log10 = energy_reco(Nc,theta_temp)
                else : 
                    reco_energy_log10 = 'None'
                    
                writer.writerow([counter,sec, nsec, theta, phi, rms_error, real_x, real_y, realxerr, realyerr, reco_energy_log10, Ne_fit,Ne_err])
                """
                fig, ax = plt.subplots(figsize=(10, 8))
                sc = ax.scatter(X, Y, c=dt, s=event_new['energy_deposits(MeV)'][check], cmap='hot', edgecolors='black')
                ax.quiver(0, 0, a, b, angles='xy', scale_units='xy', scale=0.5, width=0.005, color='r')
                ax.set_xlabel('X (East)')
                ax.set_ylabel('Y (North)')
                ax.set_xlim(-100, 100)
                ax.set_ylim(-100, 100)
                #ax.set_title(f'Event {randomEntry}: θ={theta:.2f}, ϕ={phi:.2f}')
                cbar = plt.colorbar(sc, ax=ax)
                cbar.set_label('Arrival Times (ns)')
                ax.scatter(real_x, real_y, c='blue', marker='s', s=50)
                #pdf.savefig(fig)    
                xcores.append(real_x)
                secs_store.append(sec)
                nsecs_store.append(nsec)
                ycores.append(real_y)
                xerror.append(realxerr)
                yerror.append(realyerr)
                print('final', real_x, real_y)
                theta_error.append(theta_err)
                phi_error.append(phi_err)
                theta_planes.append(theta)
                phi_planes.append(phi)
                """
        except:
            continue