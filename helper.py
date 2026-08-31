"""

Goal: Helper file- Framework for core position and energy of CR shower for RET-CR 

This framework is adapted from previous LORA reconstructions

"""
import uproot
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import math
import ROOT
from ROOT import TH2F, TF2, TF1, TH1F,TMath
from ROOT import TVirtualFitter
from IPython.display import display
from scipy.special import gamma
from boost_histogram import Histogram
from boost_histogram.axis import Regular
from boost_histogram.storage import Weight
def theta_phi(theta,phi,psi,x0,y0,z0):
    #1st ROTATION: counterclockwise from X-axis(N) for angle 'phi' about Z-axis.
    x1= x0*np.cos(phi)+y0*np.sin(phi)
    y1=-x0*np.sin(phi)+y0*np.cos(phi)
    z1= z0
    #-------------xxx----------------
    #2nd ROTATION: clockwise from Z-axis for angle 'theta' about Y-axis(W).
    x2= x1*np.cos(theta)-z1*np.sin(theta) ;
    y2= y1
    z2= x1*np.sin(theta)+z1*np.cos(theta) ;
    #-------------xxx----------------
    #3rd ROTATION: counterclockwise from X-axis(N) for angle 'psi' about Z-axis.
    x3= x2*np.cos(psi)+y2*np.sin(psi)
    y3=-x2*np.sin(psi)+y2*np.cos(psi)
    z3= z2
    #-------------xxx----------------
    return x3,y3,z3
def back_theta_phi(theta,phi,psi,x0,y0,z0):
    #1st ROTATION:clockwise from X-axis(N) for angle 'psi' about Z-axis.
    x1= x0*np.cos(psi)-y0*np.sin(psi)
    y1= x0*np.sin(psi)+y0*np.cos(psi)
    z1= z0
    #--------------xxx------------------
    #2nd ROTATION: counterclockwise from Z-axis for angle 'theta' about Y-axis(W).
    x2= x1*np.cos(theta)+z1*np.sin(theta)
    y2= y1
    z2= -x1*np.sin(theta)+z1*np.cos(theta)
    #--------------xxx-------------------
    #3rd ROTATION:clockwise from X-axis(N) for angle 'phi' about Z-axis.
    x3= x2*np.cos(phi)-y2*np.sin(phi) ;
    y3= x2*np.sin(phi)+y2*np.cos(phi) ;
    z3= z2 ;
    #--------------xxx------------------
    return x3,y3,z3

def read_attenuation():
    try:
        data=np.genfromtxt('atm_summer.dat')
        return data
    except:
        #print('problem reading atm file')
        #print('test23')
        return 0

def func_nkg_show(x,par):    #NKG function for fitting 5 parameters: (x_core,y_core,Ne,rM,s)
    return (par[0]/np.power(par[1],2))*(TMath.Gamma(4.5-par[2]))/(2*np.pi*(TMath.Gamma(par[2]))*(TMath.Gamma(4.5-2*par[2])))*np.power(np.sqrt(np.power(par[3]-x[0],2)+np.power(par[4]-x[1],2))/par[1],par[2]-2)*np.power(1.0+np.sqrt(np.power(par[3]-x[0],2)+np.power(par[4]-x[1],2))/par[1],par[2]-4.5)

def func_nkg_ind_show(x,par):    #//NKG function for fitting 3 parameters: (Ne,rM,s)
    return (par[0]/np.power(par[1],2))*(TMath.Gamma(4.5-par[2]))/(2*np.pi*(TMath.Gamma(par[2]))*(TMath.Gamma(4.5-2*par[2])))*np.power(x[0]/par[1],par[2]-2)*np.power(1.0+x[0]/par[1],par[2]-4.5)

def TF1_Fit_NKG(h1,N0,r0,s0,R_Min,R_Max,token):

    fit_R_Min=0
    if(R_Min<5):
        fit_R_Min=R_Min-2
    else:
        fit_R_Min=R_Min-5

    if(fit_R_Min<0):
        fit_R_Min=0.1


    fitf1= ROOT.TF1("fitf1","( [0] / pow([1], 2) )"
    " * tgamma(4.5 - [2])"
    " / (2.0 * 3.1415926 * tgamma([2]) * tgamma(4.5 - 2*[2]))"
    " * pow(x / [1], [2] - 2)"
    " * pow(1.0 + x / [1], [2] - 4.5)",fit_R_Min,R_Max+5) ;

    fitf1.SetParameter(0,N0)        #//Setting starting values (Ne)
    fitf1.SetParameter(1,r0)        #//Setting starting values (r_M)
    fitf1.SetParameter(2,s0)       #//Setting starting values (s)
    fitf1.SetParLimits(1,1,300)        #//Setting r_M limits
    #fitf1->SetParLimits(1,r0,r0) ;        //Setting r_M limits
    # fitf1->SetParLimits(2,1,2.5) ;        //Setting Age limits (keeping fixed)
    fitf1.SetParLimits(2,s0,s0)       # //Setting Age limits (keeping fixed)
    fitf1.SetParName(0,"N_{e}")
    fitf1.SetParName(1,"r_{M}")
    fitf1.SetParName(2,"s")

    #//h1->Fit("fitf1","V+","",fit_R_Min-5,R_Max+5) ;
    h1.Fit("fitf1", "Q", "", fit_R_Min, R_Max + 5)
    #print(h1.GetEntries())

    c1 = ROOT.TCanvas("c1", "Canvas", 800, 600)
    # Styling
    h1.SetYTitle("Charged particle density (m^{-2})")
    
    h1.GetYaxis().CenterTitle()
    h1.SetXTitle("Distance from shower axis (m)")
    h1.GetXaxis().CenterTitle()
    h1.SetMarkerColor(4)
    h1.SetMarkerStyle(20)
    h1.SetMarkerSize(1)
    h1.SetFillColor(0)

    # Global style
    ROOT.gStyle.SetOptTitle(False)
    ROOT.gStyle.SetOptStat(10)
    ROOT.gStyle.SetStatColor(0)
    ROOT.gStyle.SetTitleFillColor(0)

    # Draw the histogram
    h1.Draw("HIST")
    fitf1.Draw("SAME")     # Draw fit on top
    c1.Draw()     
    import time
    c1.Update()
    time.sleep(0.1)  # Give it time to render and write
    c1.SaveAs("hist_.png")
    Ne=fitf1.GetParameter(0)
    Ne_err=fitf1.GetParError(0)
    rM=fitf1.GetParameter(1)
    s=fitf1.GetParameter(2)
    return Ne,rM,s,Ne_err
def TF2_Fit_NKG(h2,N0,r0,s0,x0,y0,x_min,x_max,y_min,y_max):
    fitf2 = ROOT.TF2(
    "fitf2",
    (
        "([0] / pow([1], 2)) * "
        "tgamma(4.5 - [2]) / (2.0 * 3.1415926 * tgamma([2]) * tgamma(4.5 - 2.0*[2])) * "
        "pow(sqrt((x - [3])**2 + (y - [4])**2) / [1], [2] - 2.0) * "
        "pow(1.0 + sqrt((x - [3])**2 + (y - [4])**2) / [1], [2] - 4.5)"
    ),
    x_min, x_max, y_min, y_max)
    fitf2.SetParameter(0,N0) #        Setting starting values (Ne)
    fitf2.SetParameter(1,r0) #            ,,    ,,     (r_M)
    fitf2.SetParameter(2,s0) #            ,,    ,,     (s)
    fitf2.SetParameter(3,x0) #            ,,    ,,     (x_core)
    fitf2.SetParameter(4,y0) #            ,,    ,,     (y_core)
    
    #//fitf2->SetParLimits(1,10,300) ;        //Setting r_M limits
    fitf2.SetParLimits(1,r0,r0)#        //Setting r_M limits
    #//fitf2->SetParLimits(2,1,2.5) ;        //Setting Age limits (keeping fixed)
    fitf2.SetParLimits(2,s0,s0)#        //Setting Age limits (Keeping fixed)
    h2.Fit(fitf2,"Q0")
    fit_result = h2.Fit(fitf2,"SQ")
    #fit_result.Print("V") 
    #h2SetXTitle("X (m)") ;
    #h2->GetXaxis()->CenterTitle() ;
    #h2->SetYTitle("Y (m)") ;
    #h2->GetYaxis()->CenterTitle() ;
    #//h2->Draw("HIST") ;
    #//fitf2->DrawCopy("surf1 same");
    #gStyle->SetOptFit(1111) ;
    Ne_fit=fitf2.GetParameter(0)
    rM_fit=fitf2.GetParameter(1)
    s_fit=fitf2.GetParameter(2)
    x_core_fit=fitf2.GetParameter(3)
    y_core_fit=fitf2.GetParameter(4)
    x_core_fit_err=fitf2.GetParError(3)
    y_core_fit_err=fitf2.GetParError(4)
    Ne_err = fitf2.GetParError(0)
    fitter=TVirtualFitter.GetFitter()   

    corMatrix=np.zeros([5,5])    
    sig_i=0
    sig_j=0
    cov_ij=0
    
    ''' #temporarily out
    for i in np.arange(4):

        sig_i=np.sqrt(fitter.GetCovarianceMatrixElement(i,i))
        for j in np.arange(4):
            sig_j=np.sqrt(fitter.GetCovarianceMatrixElement(j,j))
            cov_ij=fitter.GetCovarianceMatrixElement(i,j)
            corMatrix[i][j]=cov_ij/(sig_i*sig_j)
            if(sig_i==0 or sig_j==0):
                corMatrix[i][j]=0
    '''
    #corCoef_xy=corMatrix[1][2] #Correlation coeff. between the x & y core position
    corCoef_xy=0 #temporary fix
    return Ne_fit,rM_fit,s_fit,x_core_fit,y_core_fit,x_core_fit_err,y_core_fit_err, corCoef_xy,Ne_err


def fit_NKG(theta_rad,phi_rad,event,token):
    #print(event['adc_counts'])
    #print('fit core position')
    theta=theta_rad
    phi=phi_rad  #Northward (anticlockwise) from East (X-axis)
    psi=2*np.pi-phi
    x = np.array(event['x'])
    y = np.array(event['y'])
    z = np.array(event['z'])
    npanels = len(np.array(event['stations']))
    x_shower=np.zeros([npanels])
    y_shower=np.zeros([npanels])
    x_pos=np.zeros([npanels])
    y_pos=np.zeros([npanels])
    z_shower=np.zeros([npanels])

    temp_det_array=[]
    temp_den_array=[]
    temp_x_pos=[]
    temp_y_pos=[]
    RET_Area = 1.0 ####### WARNING...LETS KEEP IT 1.0 for now
    rM = 30.0
    Age = 1.7
    Ref_angle=21     #Reference zenith angle (deg.) for calculating atmos. attenuation
    density = np.array(event['energy_deposits(MeV)'])/(RET_Area*np.cos(theta))
    # find core position in shower plane for first guess for fit. 4 densest detectors are used
    
    for i in np.arange(npanels):
        x_shower[i],y_shower[i],z_shower[i]=theta_phi(theta,phi,psi,x[i],y[i],z[i])
        x_pos[i]=x[i]
        y_pos[i]=y[i]
        #if detectors[i].density>=LORA.Density_Cut and detectors[i].density<=LORA.Density_Cut_High: #CHECK INTO THIS
        temp_den_array.append(density[i])
        temp_det_array.append(i)
        temp_x_pos.append(x_shower[i])
        temp_y_pos.append(x_shower[i])
        
    ind=np.argsort(np.asarray(temp_den_array))[::-1]
    temp_den_array=np.asarray(temp_den_array)
    temp_det_array=np.asarray(temp_det_array)
    temp_x_array=np.asarray(temp_x_pos)
    temp_y_array=np.asarray(temp_y_pos)

    N_TOP = 6
    temp_total_den = np.sum(temp_den_array[ind[:N_TOP]])
    x_core = np.sum([
        x_shower[temp_det_array[ind[j]]] * temp_den_array[ind[j]] for j in range(N_TOP)
    ]) / temp_total_den
    y_core = np.sum([
        y_shower[temp_det_array[ind[j]]] * temp_den_array[ind[j]] for j in range(N_TOP)
    ]) / temp_total_den

    # find min/max detector positions

    nBinsX=int(((np.max(x_pos)+10)-(np.min(x_pos)-10))/1) ## warning: Here 1 was from the LORA bin size in the original code
    nBinsY=int(((np.max(y_pos)+10)-(np.min(y_pos)-10))/1)
    x_min=(np.min(x_pos)-10)
    x_max=(np.max(x_pos)+10)
    y_min=(np.min(y_pos)-10)
    y_max=(np.max(y_pos)+10)

    nBinsX_shower=int(((np.max(x_shower)+10)-(np.min(x_shower)-10))/1)
    nBinsY_shower=int(((np.max(y_shower)+10)-(np.min(y_shower)-10))/1)
    x_min_shower=(np.min(x_shower)-150)
    x_max_shower=(np.max(x_shower)+150)
    y_min_shower=(np.min(y_shower)-150)
    y_max_shower=(np.max(y_shower)+150)
    

    evt_display = TH2F('Event','Event',nBinsX,x_min,x_max,nBinsY,y_min,y_max) # ground plane
    evt_display_shower = TH2F('Event_shower','Event_shower',nBinsX_shower,x_min_shower,x_max_shower,nBinsY_shower,y_min_shower,y_max_shower) # shower plane
    lat_den_show=TH1F('Lateral_density','Lateral_density',1600,0,200) ### warning!!  No.of bins,Rmin,Rmax These values are also taken from lora 
    time_display = TH2F('time','time',nBinsX,x_min,x_max,nBinsY,y_min,y_max)    #ground plane

    rho=np.zeros([npanels])
    rho_shower=np.zeros([npanels])
    
    rho_err=np.zeros([npanels])
    nDet_triggered=0
    f_den=0
    shower_size=0

    for i in np.arange(npanels):
        rho[i]=density[i]
        rho_err[i]==np.power(density,0.5)  ### putting a poissonian error
        #if rho[i]>=LORA.Density_Cut and rho[i]<=LORA.Density_Cut_High: ## i dont think you need this
        #nDet_triggered=nDet_triggered+1
        evt_display.SetBinContent(evt_display.GetXaxis().FindBin(x_pos[i]),evt_display.GetYaxis().FindBin(y_pos[i]),rho[i])
        evt_display_shower.SetBinContent(evt_display_shower.GetXaxis().FindBin(x_shower[i]),evt_display_shower.GetYaxis().FindBin(y_shower[i]),rho[i])
        f_den=f_den+(1.0/np.power(rM,2))*(TMath.Gamma(4.5-Age))/(2*np.pi*(TMath.Gamma(1.0))*(TMath.Gamma(4.5-2*Age)))*np.power(np.sqrt(np.power(x_core-x_shower[i],2)+np.power(y_core-y_shower[i],2))/rM,Age-2)*np.power(1.0+np.sqrt(np.power(x_core-x_shower[i],2)+np.power(y_core-y_shower[i],2))/rM,(Age-4.5))
        shower_size=shower_size+rho[i]


    # first iteration of fit -> doing Ne, xcore, ycore
    Ne_fit,rM_fit,s_fit,x_core_fit,y_core_fit,x_core_fit_err,y_core_fit_err, corr_coef_xy,Ne_err=TF2_Fit_NKG(evt_display_shower,shower_size/f_den,rM,Age,x_core,y_core,x_min_shower,x_max_shower,y_min_shower,y_max_shower)
    #print('x:   {0:.2f}   ->  {1:.2f}'.format(x_core, x_core_fit))
    #print('y:   {0:.2f}   ->  {1:.2f}'.format(y_core, y_core_fit))

    #print 'Ne:  {0:.2f}   ->  {1:.2f}'.format(shower_size/f_den, Ne_fit)
    R_Max=0
    R_Min=1000000
    radius_show=np.zeros([npanels])
    radius_bin_show=np.zeros([npanels])

    for i in np.arange(npanels):
        #if rho[i]>=LORA.Density_Cut and rho[i]<=LORA.Density_Cut_High:
        radius_show[i]=np.abs(np.sqrt(np.power(x_core_fit-x_shower[i],2)+np.power(y_core_fit-y_shower[i],2)))
        radius_bin_show[i]=lat_den_show.FindBin(radius_show[i])
        if radius_show[i]>R_Max:
            R_Max=radius_show[i]
        if radius_show[i]<R_Min:
            R_Min=radius_show[i]
        lat_den_show.SetBinContent(int(radius_bin_show[i]),rho[i])
        lat_den_show.SetBinError(int(radius_bin_show[i]),rho_err[i])

    Ne_fit,rM_fit,s_fit,Ne_fit_er=TF1_Fit_NKG(lat_den_show,Ne_fit,rM,Age,R_Min,R_Max,token)

    #print('Ne:  {0:.2f},   Rm:  {1:.2f}, xCore:  {2:.2f},  yCore:  {3:.2f}'.format(Ne_fit, rM_fit,x_core_fit, y_core_fit))

        # iterate 2d, 1d fits

    for k in np.arange(3):
        del lat_den_show
        lat_den_show=TH1F('Lateral_density','Lateral_density',1600,0,200)
        Ne_fit,rM_fit,s_fit,x_core_fit,y_core_fit,x_core_fit_err,y_core_fit_err,corr_coef_xy,Ne_err=TF2_Fit_NKG(evt_display_shower,Ne_fit,rM_fit,s_fit,x_core_fit,y_core_fit,x_min_shower,x_max_shower,y_min_shower,y_max_shower)
        
        R_Max=0
        R_Min=1000000
  
    
        for i in np.arange(npanels):
            #if rho[i]>=LORA.Density_Cut and rho[i]<=LORA.Density_Cut_High:
            radius_show[i]=np.abs(np.sqrt(np.power(x_core_fit-x_shower[i],2)+np.power(y_core_fit-y_shower[i],2)))
            radius_bin_show[i]=lat_den_show.FindBin(radius_show[i])
            if radius_show[i] > R_Max:
                R_Max=radius_show[i]
            if radius_show[i] < R_Min:
                R_Min=radius_show[i]
            lat_den_show.SetBinContent(int(radius_bin_show[i]),rho[i])
            lat_den_show.SetBinError(int(radius_bin_show[i]),rho_err[i])

        Ne_fit,rM_fit,s_fit,Ne_fit_err=TF1_Fit_NKG(lat_den_show,Ne_fit,rM_fit,s_fit,R_Min,R_Max,token)

        #print('Ne:  {0:.2f},   Rm:  {1:.2f}, xCore:  {2:.2f},  yCore:  {3:.2f}'.format(Ne_fit, rM_fit,x_core_fit,y_core_fit))
    # do atm correction ##################################
    '''
    atm_data=read_attenuation()
    size_theta=np.zeros([30])
    if len(atm_data)!=11:
        #print('no atm corrections')
    else:
        f_no=len(atm_data)
        f_int=atm_data.T[0]
        f_logN_Ref=atm_data.T[1]
        err1=atm_data.T[2]
        f_X0=atm_data.T[3]
        err2=atm_data.T[4]
        f_lamb=atm_data.T[5]
        err3=atm_data.T[6]
        Lambda0=0
        err_Lambda0=0
        for k in np.arange(len(f_int)):
            size_theta[k]=f_logN_Ref[k]-(f_X0[k]/f_lamb[k])*(1/np.cos(theta)-1/np.cos(np.pi*Ref_angle/180.0))*0.4342944819 #log10(size) for attenuation curve kk at zenith angle 'theta'
            #print size_theta[k]
            
            if np.log10(Ne_fit) >= size_theta[1]: ## typo?   1->k?
                Lambda0=f_lamb[k]    #  //Extropolation
                err_Lambda0=err3[k]
                break
            
            elif np.log10(Ne_fit) >= size_theta[k]:
                Lambda0=(f_lamb[k]*(np.log10(Ne_fit)-size_theta[k-1])+f_lamb[k-1]*(size_theta[k]-np.log10(Ne_fit)))/(size_theta[k]-size_theta[k-1]) #  //Interpolation
                err_Lambda0=(err3[k]*(np.log10(Ne_fit)-size_theta[k-1])+err3[k-1]*(size_theta[k]-np.log10(Ne_fit)))/(size_theta[k]-size_theta[k-1]) #  //Interpolation
         
                break
    ####

    size_theta[f_no-1]=f_logN_Ref[f_no-1]-(f_X0[f_no-1]/f_lamb[f_no-1])*(1/np.cos(theta)-1/np.cos(np.pi*Ref_angle/180))*0.4342944819

    if np.log10(Ne_fit) >= size_theta[f_no-1]:
        Lambda0=f_lamb[f_no-1] #; //Extropolation
        err_Lambda0=err3[f_no-1]

    X0=1024 ##1024 is from LORA warning!!!!
    par_a=1.23    #Eneregy reconstruction paramter (From Kickelbick 2008, Kascade thesis)
    par_b=0.95    #  ,,  ,,  ,,
    err_a=0.14    #Error on "par_a"
    err_b=0.02    #Error on "par_b"
    
    log_size_Ref=np.log10(Ne_fit)+(X0 /Lambda0)*(1/np.cos(theta)-1/np.cos(np.pi*Ref_angle/180.0))*0.4342944819# ; //log10() 
    size_Ref=np.power(10,log_size_Ref)
    err_size_Ref=np.sqrt(np.power(Ne_fit_err/Ne_fit,2)+np.power(np.log(10)*X0*(1/np.cos(theta)-1/np.cos(np.pi*Ref_angle/180.0))*0.4342944819,2)*np.power(err_Lambda0/np.power(Lambda0,2),2))*size_Ref

    #energy_Ref=pow(size_Ref,par_b)*pow(10,par_a)*pow(10,-6) ; //Energy(PeV) at Ref_angle: Formula from KASCADE simulation (2008)
    #err_energy_Ref=sqrt(pow(log(10)*err_a,2)+pow(log(size_Ref)*err_b,2)+pow(par_b*err_size_Ref/size_Ref,2))*energy_Ref ;    //error on energy at Ref_angle
    ###ACTIVATE BELOW AFTER CHECKING
    #energy_Ref=np.power(size_Ref,par_b)*np.power(10.0,par_a)*np.power(10.0,-6.0) #; #//Energy(PeV) at Ref_angle: Formula from KASCADE simulation (2008)
    #err_energy_Ref=np.sqrt(np.power(np.log(10)*err_a,2)+np.power(np.log10(size_Ref)*err_b,2)+np.power(par_b*err_size_Ref/size_Ref,2))*energy_Ref ##;    //error on energy at
    
    '''
    #undo core rotation
    X3,Y3,Z3=back_theta_phi(theta,phi,psi,x_core_fit,y_core_fit,0)
    l,m,n=back_theta_phi(theta,phi,psi,0,0,1)
    x_core_ground=l*(-Z3/n)+X3 ;
    y_core_ground=m*(-Z3/n)+Y3 ;
    #energy=np.power(Ne_fit,par_b)*np.power(10.0,par_a)*np.power(10.0,-6.0)# ; //Energy (PeV): Formula from KASCADE simulation (2008)
    #err_energy=np.sqrt(np.power(np.log(10)*LORA.err_a,2)+np.power(np.log10(Ne_fit)*LORA.err_b,2)+np.power(LORA.par_b*Ne_fit_err/Ne_fit,2))*energy #;    //error on energy    
    print("inside print",x_core_ground,y_core_ground,Ne_err)
    return Ne_fit,rM_fit,s_fit,x_core_ground,y_core_ground,x_core_fit_err,y_core_fit_err,Ne_err


