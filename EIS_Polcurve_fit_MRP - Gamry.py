###########################################
#this script was developed by Makenzie Parimuha for Gamry EIS / Pol curve / CV analyses
#if you have any questions, email Makenzie.Parimuha@nrel.gov
#most recently updated June 2024
###########################################

from tkinter import *
import tkinter as Tkinter
import tkinter.filedialog as tkFileDialog
import tkinter.messagebox as tkMessageBox

import matplotlib
matplotlib.use('TkAgg')
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv 
import re

#%%

class EISPROC:
    def __init__(self, master):


#%%making the GUI
        master.title("GAMRY EIS Processing V2.0")
        self.master = master
        master.geometry("650x230")

        
        inputFrame  = Frame(master, padx=10)
        outputFrame = Frame(master, padx=10)
        buttonFrame = Frame(master, pady=10)
        
        
        inputFrame.grid(row = 1, column=0, sticky=N, pady=3)
        outputFrame.grid(row=1, column=1, sticky=N, pady=3)
        buttonFrame.grid(row=0,columnspan=2)  

        
        self.directoryIN = Entry()
        self.directoryOUT = Entry()
        self.ModeSelectBox = Entry()
        self.currentFileName = Tkinter.StringVar(master)
      
    
        # Dropdown menu to select the mode of processing, EIS or Polarization
        self.var = StringVar(master)
        self.var.set("---") # default value
        Label(buttonFrame, text="Mode Select:").grid(row=0, column=0)
        self.ModeSelectBox = OptionMenu(buttonFrame, self.var, 'Visualization', 'EIS Manual Fit', "EIS Summary", 'CV')
        self.ModeSelectBox.configure(width = 40)
        self.ModeSelectBox.grid(row=0, column=1)

        # File input directory, chosen via finder/etc
        self.InDir = Button(buttonFrame, width = 18, text="Select Input Directory", command=lambda : self.SelectINPUT())
        self.InDir.grid(row=1, column=0, sticky = W)
        Label(buttonFrame).grid(row=1, column=1, sticky=W)
        self.directoryIN = Entry(buttonFrame, width=40)
        self.directoryIN.grid(row=1, column=1, sticky=W)#, columnspan=2)
        self.directoryIN.insert(0, "---")
        self.directoryIN.config(state='readonly')

        
        # String entry box for the name of the output file. 
        Label(buttonFrame, text = "CCM area (cm2):").grid(row = 3, column = 0)
        self.ccmarea = Entry(buttonFrame, width = 18)
        self.ccmarea.insert(0, '5')
        self.ccmarea.grid(row=3, column=1, sticky =W)

        # String entry box for the name of the output file. 
        Label(buttonFrame, text = "Output File Name:").grid(row = 4, column = 0)
        self.OFile = Entry(buttonFrame, width = 40)
        self.OFile.insert(0, "ccm-r1_Ir")
        self.OFile.grid(row=4, column=1, sticky =W)
        
        # Import Files, Process, Export to CSV
        self.Run = Button(buttonFrame, text = 'Run', command = self.RunIt)
        self.Run.grid(row = 6, column = 0)

#%%start of functions (selectINPUT for GUI)

    def SelectINPUT(self):
        # Allows user to select the input directory from the file navigation
        NewInDir = tkFileDialog.askdirectory(title="Select input directory")+'/'
        self.directoryIN.config(state='normal')
        self.directoryIN.delete(0, END)
        self.directoryIN.insert(0, NewInDir)
        self.directoryIN.config(state='readonly')
        
#%%this part removes all nonledgible symbols from EIS files and determines which version of GAMRY output files are being processed
    
    def file_clean_up(self):
        # get list of all EIS.dta files         
        directoryIN = self.directoryIN.get()
        self.eis_dfs = []
        paths = []
        for dirpath, dirname, files in os.walk(directoryIN):
            for x in files:
                if 'EIS' in x and x.endswith('.DTA'):
                    paths.append(os.path.join(dirpath, x))
        paths.sort()
                   
        #clean up all the degree symbols        
        for i, File in enumerate(paths):
            check = True
            while check == True:
                with open(File, 'rb') as rfile: #opens file to check for degree symbol
                    content = rfile.read()
                    degree_symbol_bytes=b'\xb0'
                    index = content.find(degree_symbol_bytes)
                rfile.close()            
                if index != -1: #if degree symbol is detected, replace the degree symbol with deg
                    Filefix = open(File,"r+b")
                    Filefix.seek(index)
                    Filefix.write(bytes("deg","utf-8"))
                    Filefix.close()
                else:
                    check = False
     
        #determine if .DTA file is from old or new gamry (there are differences)
        # determine necessary parameters to open a dataframe correctly based on which old v new gamry 
        file1 = open(paths[0], 'r')
        content = file1.readlines()
        if 'DRIFTCOR' in str(content[15]): #new gamry if statement
            self.gamry_version = 'new'
            columns = ['','Pt',	'Time-s','Freq','Zreal','Zimag','Zsig','Zmod',
                       'Zphz','Idc','Vdc','IERange', 'Imod','Vmod','Temp']
            df_start = 63
        else:
            self.gamry_version = 'old'
            columns = ['','Pt',	'Time-s','Freq','Zreal','Zimag','Zsig','Zmod',
                       'Zphz','Idc','Vdc','IERange']
            df_start = 57
        file1.close()
        
        for f, File in enumerate(paths):
            path, tail = os.path.split(File)  
            #open up data lines into dataframe
            df=pd.read_csv(File, skiprows=df_start, sep='\t', header=None, names=columns)
            df.rename(columns={'Freq':'Frequency (Hz)', 'Zreal':'Z\' (Ω)', 'Zimag':'-Z\'\' (Ω)',	
                               'Zmod':'Z (Ω)',	'Zphz':'-Phase (°)'}, inplace=True)
            df['-Z\'\' (Ω)'] = df['-Z\'\' (Ω)']*-1
            df['Zimag'] = df['-Z\'\' (Ω)']
            df['Filename'] = [tail for x in range(len(df))]
            self.eis_dfs.append(df)
        

        # set the ccm variable for the whole script from the starter GUI
        self.ccm_area = float(self.ccmarea.get())

        
#%%Process impedence
    def Impedance(self):
        #clear all data storage lists
        self.hfr_ohm = []
        self.hfr = []
        self.fit_regime = []       
        dir_out = self.directoryOUT
        FileOut = self.OFile.get()
            
        N_fig, N_ax = plt.subplots()
        N_ax.set_xlabel('RealZ (ohm)')
        N_ax.set_ylabel('-ImagZ (ohm)')
        N_ax.set_title(FileOut)
        N_ax.set_aspect('equal')
        
        for f, df in enumerate(self.eis_dfs):
            path, tail = os.path.split(df['Filename'].iloc[0])
            if not df.empty:
                df['Zimag'] = df['-Z\'\' (Ω)']
                df['Zreal'] = df['Z\' (Ω)']
                current = df['Idc'].mean()
                plot_title = np.mean(current) / self.ccm_area
                
                if 'POT' in tail:
                    self.CL_resistance(df)
                    continue  
                
                if self.var.get() == 'EIS Manual Fit':                    
                    dir_out = os.path.join(self.directoryOUT, 'manual fit')
                    if os.path.exists(dir_out) == False:
                        os.mkdir(dir_out)                      
                    completepath = os.path.join(dir_out, f'{plot_title/self.ccm_area:.3f}Acm2.csv')        
                    df.to_csv(completepath, index=False)
                
                if f != 0 and f%3 == 0:
                    N_ax.scatter(df['Zreal'], df['Zimag'], s=2, label=f'{plot_title/self.ccm_area:.3}')
                
                if self.var.get() == 'Visualization':
                    #if mode = Visualization then only bode plots and clean nyquist plots + csv to plot them will be made
                    plotsout = os.path.join(dir_out, 'Nyquist+Bode Plots')
                    if os.path.exists(plotsout) == False:
                        os.mkdir(plotsout)
                    #nyquist plot                        
                    fig1, ax1 = plt.subplots()
                    ax1.set_xlabel('RealZ (ohm)')
                    ax1.set_ylabel('-ImagZ (ohm)')
                    ax1.scatter(df['Zreal'], df['Zimag'], color = 'red')
                    ax1.hlines(xmin=min(df['Zreal']), xmax=max(df['Zreal']), y=0, color ='k', linestyles='dashed') 
                    ax1.set_xlim(min(df['Zreal']), max(df['Zreal']))
                    ax1.set_aspect('equal')
                    ax1.set_title(f'{plot_title:.3f} A')
                    fig1.savefig(os.path.join(plotsout, f'{plot_title:.3f}-Nyquist.png'), dpi=200, bbox_inches='tight') 
                    plt.close()
                    #bode plot
                    fig, ax = plt.subplots()
                    ax0 = ax.twinx()
                    ax.set_xlabel('Frequency (Hz)')
                    ax.set_ylabel(r'$\lvertZ\rvert (\Omega)$')
                    ax0.set_ylabel(r'$\phi (^{\circ})$')
                    ax.scatter(df['Frequency (Hz)'], df['Zimag'], color = 'red')
                    ax0.scatter(df['Frequency (Hz)'], df['-Phase (°)'], color='black')
                    ax.set_xscale("log", base=10)
                    ax.set_yscale("log", base=10)
                    ax.yaxis.label.set_color('red')
                    ax.set_title(f'{plot_title:.3f} A')
                    fig.savefig(os.path.join(plotsout, f'{plot_title:.3f}-Bode.png'), dpi=200, bbox_inches='tight') 
                    plt.close()
                                    
                if self.var.get() == 'EIS Summary':
                    #will run if EIS or summary is selected. fits nyquist data
                    plotsout = os.path.join(dir_out, 'Nyquist fit check')
                    if os.path.exists(plotsout) == False:
                        os.mkdir(plotsout)

                    try: 
                        intercept, regime = self.full_EIS(df, plotsout)
                    except:
                        print(f'unable to fit {plot_title:.3f}A cm-2 ({tail}) EIS')
                        intercept = 0
                        regime = 'manual fit required'
                            
                    self.hfr_ohm.append(intercept)
                    self.hfr.append(intercept*self.ccm_area)
                    self.fit_regime.append(regime)
                        
            
        if not df.empty:
            N_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
            N_fig.savefig(os.path.join(self.directoryOUT, 'Nyquist_series.png'), dpi=300, bbox_inches='tight') 
            plt.close()

#%%Process CL resistance   
    def CL_resistance(self, df0):
        dir_out = os.path.join(self.directoryOUT, 'manual fit')
        if os.path.exists(dir_out) == False:
            os.mkdir(dir_out)         
        #this part extracts high frequency intercept of 
        df_low = df0[df0['Zimag']<0].copy()
        df = df0[df0['Zimag']>0].copy()   
        if not df_low.empty:
            fit_x = [df_low['Zimag'].iloc[-1], df['Zimag'].iloc[0]]
            fit_y = [df_low['Zreal'].iloc[-1], df['Zreal'].iloc[0]]
            m,b = np.polyfit(fit_x,fit_y,1)
        
        #to fit linear portion of non-faradaic EIS
        imgZ=list(df['Zimag'].iloc[-23:-15])
        rlZ=list(df['Zreal'].iloc[-23:-15])
        lin_func = np.poly1d(np.polyfit(imgZ,rlZ,1))
        linear_range = np.arange(0, max(imgZ), 0.001)

        #nyquist plot 
        plot_title = df['Vdc'].mean()                       
        fig1, ax1 = plt.subplots(figsize=(5,10))
        ax1.set_xlabel('RealZ (ohm)')
        ax1.set_ylabel('-ImagZ (ohm)')
        ax1.hlines(xmin=0.4*lin_func(0), xmax=1.3*max(lin_func(linear_range)), y=0, color = 'black', linestyles='dashed')         
        ax1.plot(df['Zreal'], df['Zimag'], color='k',ls='none',marker='o', label='R_cl EIS')        
        ax1.plot(rlZ, imgZ, c='magenta',marker='o', mfc='none', mec='magenta',ls='none')
        ax1.plot(lin_func(linear_range), linear_range, c='magenta', ls='--', label=f'{lin_func(0):.3g}')  
        if not df_low.empty:
            self.cl_resistance.append(3*self.ccm_area*(lin_func(0)-b))
            ax1.plot(fit_y, fit_x,marker='o', c='orange',mfc='none',mec='orange',ls='none')              
            ax1.scatter(b,0, c='orange', label=f'{b:.3g}')   
        else:
            self.cl_resistance.append('manual calculation required')
        ax1.set_xlim(0.4*lin_func(0), 1.3*max(lin_func(linear_range)))
        ax1.set_ylim(-0.03, 1.3*max(linear_range))
        ax1.set_aspect('equal')
        ax1.set_title(f'{plot_title:.3f} V - CL resistance')
        ax1.legend()
        fig1.savefig(os.path.join(dir_out, f'{plot_title:.2f}V-CL resistance.png'),bbox_inches='tight', dpi=300) 
        #zoom into curved region
        ax1.set_xlim(0.8*lin_func(0), 1.05*lin_func(0))
        ax1.set_ylim(-0.005, 0.05*max(linear_range))
        fig1.savefig(os.path.join(dir_out, f'{plot_title:.2f}V-CL resistance_ZOOM.png'),bbox_inches='tight', dpi=300) 
        plt.close()
        
        #export to a csv and save the calculated Rcl value 
        df.to_csv(os.path.join(dir_out, f'{plot_title:.2f}V.csv'), index=False)
        
        
#%%
    def full_EIS(self, df, plotsout):
        intercept = 0 
        function = 0 
                   
        df_pos = df[df['Zimag']>0].copy() 
        #print(df['Zimag'])
        cut_off = df_pos['Frequency (Hz)'].iloc[0]
        df_neg = df[(df['Zimag']<0)&(df['Frequency (Hz)']>cut_off)].copy()
        #setting up a lower and upper limit based on the last Z'' before crossing y=0
        up_lim = df_pos['Zreal'].iloc[0]
        low_lim = 0.9*up_lim
        
        #check for inductance range / and also that tail is not curling to the right
        if not df_neg.empty:
            if df_neg['Zreal'].iloc[-1] < df_pos['Zreal'].iloc[0]:
                index = -1
            else:
                index = -2               
            fit_y = [df_neg['Zreal'].iloc[index], df_pos['Zreal'].iloc[0]]
            fit_x = [df_neg['Zimag'].iloc[index], df_pos['Zimag'].iloc[0]]
            fit = np.polyfit(fit_x, fit_y, 1)
            if low_lim<fit[-1]<up_lim:
                intercept = fit[-1]
                function = np.poly1d(fit)
                regime = 'linear interpolation'
            
       
        if intercept == 0:                              
            #determine position of the peak in the Nyquist plot
            max_ind = df_pos['Zimag'].idxmax()
            #df_wip excludes all data points after Nyquist peak
            df_wip = df_pos.loc[:max_ind] 
            if len(df_wip) > 10:
                df_wip = df_pos[:8]
            
            fitorder = [2, 3, 4]#, 5]
            bestfunc = {}
            for o in fitorder:
                fit_y = df_wip['Zreal']
                fit_x = df_wip['Zimag']
                fit = np.polyfit(fit_x, fit_y, o)
                func = np.poly1d(fit)  
                R2 = r2_score(df_wip['Zreal'], func(df_wip['Zimag']))
                if low_lim<fit[-1]<up_lim:
                    bestfunc[R2] = [func, o, fit[-1]]        
            if bestfunc:
                r = max(list(bestfunc.keys()))
                function = bestfunc[r][0]
                order = bestfunc[r][1]                            
                intercept = bestfunc[r][2]
                regime = f'{order}order extrapolation'
                        
            if intercept == 0:
                for k in reversed(range(2, 5)):
                    fit_y = df_wip['Zreal'].iloc[0:k]
                    fit_x = df_wip['Zimag'].iloc[0:k]
                    func = np.poly1d(np.polyfit(fit_x, fit_y, 1))
                    R2 = r2_score(fit_y, func(fit_x))    
                    if low_lim<func(0)<up_lim:
                        intercept = func(0)
                        function = func
                        regime ='linear extrapolation'
                        break
                    elif k == 2:
                        intercept = up_lim-0.5*(abs(up_lim-func(0)))
                        regime = 'modified linear extrapolation'
                        
        plot_title = df['Idc'].mean()/self.ccm_area
        #plot the fitted nyquist plots to check fitting quality  
        fig, ax = plt.subplots(dpi=200)
        ax.set_xlabel('RealZ (ohm)')
        ax.set_ylabel('-ImagZ (ohm)')
        ax.plot(df['Zreal'], df['Zimag'],ls='dashed',marker='o', label=f'{plot_title:.3f}Acm-2 data')
        ax.set_aspect('equal', 'box')
        if intercept != 0:
            ax.plot(fit_y, fit_x, ls='none',marker='o',mfc='none',mec='orange',mew=2.5, label='fit points')
            ax.plot(intercept, 0, 'r', marker = 'x', markersize=8)
            data = np.arange(0,max(fit_x), 0.0001)
            ax.plot(function(data), data, 'r', marker='o', markersize = 2,alpha=0.3, label=regime)                 
        ax.legend()                 
        fig.savefig(os.path.join(plotsout, f'{plot_title:.3f}.png'), dpi=200, bbox_inches='tight') 
        if (max(df['Zreal'])-min(df['Zreal'])) > 0.015:
            ax.set_xlim(1.05*low_lim, 1.1*up_lim)
            imZ_lim = df_neg['Zimag'].iloc[-1]
            ax.set_ylim(8*imZ_lim,-8*imZ_lim)
            fig.savefig(os.path.join(plotsout, f'{plot_title:.3f}-intercept.png'), dpi=200, bbox_inches='tight') 
        plt.close()
                        
        return(intercept, regime)


#%%process pol curve files
    def Polarization(self):
        # Clears the lists every time the Import is called. 
        self.voltage = []
        self.current_density = []
        directoryIN = self.directoryIN.get()
        pol_path = []
        
        #this is specific to the Gamry script
        for dirpath, dirname, files in os.walk(directoryIN):
            for x in files:
                if 'polcurve' in x and x.endswith('.DTA'):
                    pol_path.append(os.path.join(dirpath, x))
        pol_path.sort()
        if len(pol_path) > len(self.eis_dfs):
            del pol_path[-1]
            
        for f, File in enumerate(pol_path):
            if self.gamry_version == 'new':
                df_start = 65
            else:
                df_start = 59
            path, tail = os.path.split(File)
            df=pd.read_csv(File, skiprows=df_start, sep='\t', header=None, names=['','Pt','Time-s',
                            'Voltage','Im','Vu','Sig','Ach','IERange','Over','Temp','other'])
            
            Voltage = df['Voltage']
            Current = df['Im']
            meanV = np.mean(Voltage)
            meanC = np.mean(Current)
            volt_sd = np.std(Voltage)
            #for data export and pol curve plot
            self.current_density.append(meanC/self.ccm_area)
            self.voltage.append(meanV)
            
            if volt_sd/meanV > 0.05:
                print('\n', f'{meanC/self.ccm_area:.3g}Acm-2 pol curve: {meanV:.3g}V +/- {volt_sd:.2g}V')
                plotsout = os.path.join(self.directoryOUT, 'Pol Stabilization Plots')
                if os.path.exists(plotsout) == False:
                    os.mkdir(plotsout)
                fig, ax = plt.subplots()
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Voltage (V)')
                ax.scatter(df['Time-s'], df['Voltage'], label=f'{meanC/self.ccm_area}A cm-2')
                ax.set_ylim(0, 1.25*meanV)
                fig.savefig(os.path.join(plotsout, f'{meanC/self.ccm_area:.3g}Acm2.png'), dpi=300)
                plt.close()
        
        if self.var.get() != 'EIS Summary':
            fig, ax = plt.subplots()
            ax.set_title(self.OFile.get())
            ax.set_ylabel('Potential (V)')
            ax.set_xlabel(r'J $A/cm^2$')
            ax.plot(self.current_density, self.voltage, marker='o', color='blue')
            fig.savefig(os.path.join(self.directoryOUT, 'pol only.png'), dpi=300)
            plt.close()
            
            
        
            
#%%extract final CV cycles from benchmarking CV files
    def CVAnalyses(self):
        # Clears the lists every time the Import is called. 
        self.mvps_rate = []
        self.cv_xlsx = []
        directoryIN = self.directoryIN.get()
        cv_path = []
        for dirpath, dirname, files in os.walk(directoryIN):
            for x in files:
                if 'mVps' in x and x.endswith('.DTA'):
                    cv_path.append(os.path.join(dirpath, x))
                    self.mvps_rate.append(int(re.search('\d{2,3}', x).group()))
        cv_path.sort()        
        self.cv_xlsx.append(self.mvps_rate)    
        for i, File in enumerate(cv_path):
            dataFile = open(File, "r+")
            if dataFile:
                Vlist = [[], [], [], [], [], [], [], []]
                Clist = [[], [], [], [], [], [], [], []]
                k = 0
                i = 0
                num = 0                
                dataLineString = dataFile.readline()
                while dataLineString:
                    if 'CURVE' in dataLineString:
                        num = int(re.search('\d{1,2}', dataLineString).group())
                        n = num - 3
                        k = i                    
                    if (num > 2) & (num < 11) & (k != 0) & (i > k+2) & (len(dataLineString) > 2) & (dataLineString[0] != '#'):                         
                        lineList = dataLineString.split("\t")
                        Vlist[n].append(float(lineList[3]))
                        Clist[n].append(float(lineList[4]))                                                                        
                    dataLineString = dataFile.readline()
                    i += 1                
                while Vlist:
                    if len(Vlist[-1]) < len(Vlist[0]):
                        Vlist.remove(Vlist[-1])
                        Clist.remove(Clist[-1])
                        #print(len(Vlist))
                    else:
                        break                
                self.cv_xlsx.append(Clist[-1])
                self.cv_xlsx.append(Vlist[-1])                
                dataFile.close()
        

#%%start printing the requested files                        
    def PrintCVtoCSV(self):
        directoryOUT = self.directoryOUT
        FileOut = self.OFile.get()                
        row0 = ['{}mVps'.format(self.cv_xlsx[0][0]), '', '{}mVps'.format(self.cv_xlsx[0][1]), '',  
                '{}mVps'.format(self.cv_xlsx[0][2]), '', '{}mVps'.format(self.cv_xlsx[0][3]), '', '{}mVps'.format(self.cv_xlsx[0][4]), '']
        row1 = ["Current", "Voltage", "Current", "Voltage", "Current", "Voltage", "Current", 
                "Voltage", "Current", "Voltage"]
        row2 = ['A/cm2', 'V', 'A/cm2', 'V', 'A/cm2', 'V', 'A/cm2', 'V', 'A/cm2', 'V']
        list_results = []
        list_results.append(row0)
        list_results.append(row1)
        list_results.append(row2)
        for n in range(len(self.cv_xlsx[1])):
            col0 = self.cv_xlsx[1][n]
            col1 = self.cv_xlsx[2][n]
            col2 = self.cv_xlsx[3][n]
            col3 = self.cv_xlsx[4][n]
            col4 = self.cv_xlsx[5][n]
            col5 = self.cv_xlsx[6][n]
            col6 = self.cv_xlsx[7][n]
            col7 = self.cv_xlsx[8][n]
            col8 = self.cv_xlsx[9][n]
            col9 = self.cv_xlsx[10][n]            
            list_results.append([col0, col1, col2, col3, col4, col5, col6, col7, col8, col9])
        completeName = os.path.join(directoryOUT, FileOut)
        with open(completeName, "wt") as out:
            csv_writer = csv.writer(out, delimiter = ",", lineterminator = '\n')
            csv_writer.writerows(list_results)
        if out:
            print ("CV Exported!")

#%%Export combined Pol and EIS data
    def Summarize(self):
        directoryOUT = self.directoryOUT
        FileOut = self.OFile.get()
        plot_name = FileOut
        if self.ccm_area != 5:
            FileOut = FileOut+'_'+str(self.ccm_area)
        
        datalist = [self.current_density, self.voltage, self.hfr, self.hfr_ohm, self.fit_regime, ['(,)']]
        ## create dataframe of data for export to csv format
        columnlist = ["Current density", "Voltage", 'HFR', "HFR-ohm", 'Fit regime', 'Tafel range']
        units = ["A/cm2", "V", "Ohm*cm2", "Ohm", '', ''] 
        if self.cl_resistance:
            datalist.append(self.cl_resistance)
            columnlist.append('R_cl')
            units.append('Ohm*cm2')
                     
        df_list = []         
        for i, dataset in enumerate(datalist):
            temp = []
            temp.append(units[i])
            temp.extend(dataset)
            df0 = pd.DataFrame({columnlist[i]:temp})
            df_list.append(df0)
        df = pd.concat(df_list, axis=1)
        
        completeName = os.path.join(directoryOUT, FileOut+'.csv')
        if os.path.exists(completeName) and self.var.get() == 'EIS Manual Fit':
            pass
        else:
            df.to_csv(completeName, index=False)  
            print ('\n', plot_name, " Summary Exported!")
        
        if self.var.get() == 'EIS Summary':
            df.drop([0], inplace = True)        
            df['hfr free'] = df['Voltage'] - df['HFR']*df['Current density']
                    
            fig, ax = plt.subplots()
            ax.set_title(plot_name)
            ax.set_xlabel('Current density {}'.format(r'$\frac{A}{cm^2}$'))
            ax.set_ylabel('Potential (V)')
            ax.plot(df['Current density'], df['Voltage'], 'o', linestyle = 'solid')
            
            index = df[df['HFR'] == 0].index
            df.drop(index, inplace = True)
            hfr = df['HFR-ohm']
            print(f'{np.mean(hfr):.3g}$\Omega$ +- {np.std(hfr):.2g}')
            
            ax.plot(df['Current density'], df['hfr free'], '--')
            lasty=df['Voltage'].iloc[-1]
            lastx=df['Current density'].iloc[-1]
            ax.annotate('{:.2f}V'.format(lasty), (lastx, lasty - 0.05), ha='center', color='blue')
            ax.grid(color='#7f7f7f', linestyle='--', linewidth=0.5)
            ax.hlines(xmin=3.2, xmax=3.5, y=1.45, color = 'black', linestyles="dashed")
            ax.text(x=3.75, y=1.45, s='ir-free', ha='center', va='center', backgroundcolor='white')
            fig.savefig(os.path.join(directoryOUT, 'pol-ir free curve.png'), dpi=300)
            plt.close()

            fig0, ax0 = plt.subplots()
            ax0.set_title(plot_name + ' HFR')
            ax0.set_xlabel('Current density {}'.format(r'$\frac{A}{cm^2}$'))
            ax0.set_ylabel('HFR*{} ({})'.format(r'$10^3$', r'$m\Omega*cm^2$'))
            ax0.plot(df['Current density'], df['HFR']*1000, 'o', linestyle = 'solid')
            ax0.grid(color='#7f7f7f', linestyle='--', linewidth=0.5)
            ax0.set_ylim(0, 1.1*max(df['HFR'])*1000)
            fig0.savefig(os.path.join(directoryOUT, 'hfr.png'), dpi=300)
            plt.close()
            

#%%the machine that runs it all
    def RunIt(self):        
        self.cl_resistance = []
        self.ccm_area = float(self.ccmarea.get())
        self.file_clean_up()

        #create processed folder for all data to go into
        self.directoryOUT = os.path.join(self.directoryIN.get(), 'Processed EIS')
        if os.path.exists(self.directoryOUT) == False:
            os.mkdir(self.directoryOUT)
            
        if self.var.get() == "CV":
            self.CVAnalyses()
            self.PrintCVtoCSV()
            
            
        else: 
            self.Impedance()
            
            if self.var.get() == "EIS Summary" or self.var.get() == "EIS Manual Fit":
                self.Polarization()
                self.Summarize()
        
        print ('\n script finished!')

if __name__ == '__main__':

    root = Tk()
    App = EISPROC(root)  
    root.mainloop()
