# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 11:30:09 2024

@author: mparimuh
"""

from tkinter import *
import sys
sys.path.append(r"C:\Users\mparimuh\Documents\GitHub\Durability_PEMWE")
from VBA_v5b_MP_import import voltagebreakdown as vba

import tkinter as Tkinter
import tkinter.filedialog as tkFileDialog
import matplotlib.dates as md
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 15})

#%%
class cellcharacteristics:
    def __init__(self):
        ########sets up parameters used throughout script
        #Fill out station parameters
        self.station = 'BB' #can be 'BB' or 'DS'
        self.cond_programs = ['25 cm^2 conditioningCurrentCtrl'] #all conditioning protocol names in MATRIX_WORKSHEET
        self.hold_programs = [] #all hold program names in MATRIX_WORKSHEET
        self.hold_J = 0 #hold current density, unit = A/cm2
        self.temperature = 80 #units are in celcius
        self.pressure = 0 #units in bar
        ###########################################################################################
        ####EXAMPLE: Mott5-hfr-r1.csv (all hfr should be labelled sequentially i.e. r1, r2, r3)####
        ###########################################################################################
        #pos1 = label for plots
        #pos2 = active area size unit in cm2
        #i.e. self.cell = ['Sinter 2',25]
        self.cell = ['NR212 1000EW',25] 
        #dates of eis (does not have to be exAcT. COULD BE A TIME HALFWAY THROUGH eis)
        #enter dates in format: 'YYMMDD hhmmss', using military time/24h clock
        #separate dates by a comma
        self.eis_dates = [] 
        ###########################################################################################
        #############OPTIONAL#################OPTIONAL###################OPTIONAL##################
        #enter tafel range for hfr in order of collected
        #enter data following style [(1st point,end point),(1,10),(2,11)]
        self.tafel = []

        #END USER INPUT SECTION
        self.p_sequential = ['black', "#0000b3", "#0010d9", "#0040ff", "#0080ff", "#00bfff"]
        self.p_sequential.reverse() #blue gradient color palette
        self.m = ['P','X','v','^','D','d']
        self.c = '#E94C1F'
              
    def metadata(self, dir_in):
        #if init cell data is empty and the file Cell MetaData.xlsx file exists, the script will pull info from it
        meta_data_file = 0
        for dirpath, dirname, files in os.walk(dir_in):
            for x in files:            
                if x == 'CellMetaData.xlsx':
                    meta_data_file = os.path.join(dirpath, x)
                    break
        #this section checks on if the metadata sheet or the init function has more information
        if self.eis_dates or meta_data_file !=0 :
            if meta_data_file != 0:
                df = pd.read_excel(meta_data_file)
                if len(self.eis_dates)<len(df['EIS dates']):
                    self.station = df.at[0,'Station']
                    self.cond_programs = df['Cond Protocols'].to_list()
                    self.hold_programs = df['Hold Protocols'].to_list()
                    self.temperature = int(df.at[0,'Hold T (C)'])
                    self.pressure = int(df.at[0,'Hold P (Bar)'])
                    self.cell = [df.at[0, 'Label'],df.at[0, 'Active Area']]
                    self.hold_J = int(df.at[0,'Hold J'])
                    self.eis_dates = df['EIS dates'].to_list()
                    self.tafel = df['tafel range'].to_list()
        else:
            printstatement = '\nInadequate information provided \nNo Cell MetaData.xlsx found \nPlease fill out CellCharacteristics \nAnd rerun program \n\nbye bye!\n'
            sys.exit(printstatement)

    
    def metadata_export(self, dir_in):
        #this function exports all metadata chich can be stored for future use of the script
        keys = ['Station','Cond Protocols','Hold Protocols', 'Hold T (C)','Hold P (Bar)','Hold J','Active Area','Label','EIS dates','tafel range']
        values = [self.station,self.cond_programs,self.hold_programs,self.temperature,self.pressure,self.hold_J,self.cell[1],self.cell[0],self.eis_dates,self.tafel]
        values = [pd.Series(a) for a in values]
        metainfo = dict(zip(keys, values))
        df = pd.DataFrame.from_dict(metainfo)
        df.to_excel(os.path.join(dir_in,'CellMetaData.xlsx'),index=False)

class Durability(cellcharacteristics):
    def __init__(self, master):
        super().__init__()       
        
        master.title("Single-cell Durability")
        self.master = master
        master.geometry("650x230")
        buttonFrame = Frame(master, pady=10)        
        buttonFrame.grid(row=0,columnspan=2)          
        self.directoryIN = Entry()
        self.ModeSelectBox = Entry()
      
#%%to make the GUI
        # Dropdown menu to select the mode of processing
        self.var = StringVar(master)
        self.var.set("---") # default value
        Label(buttonFrame, text="Mode Select:").grid(row=0, column=0)
        self.ModeSelectBox = OptionMenu(buttonFrame, self.var, 'breakin', 'v(t) plots', 'hfr durability', 'crossover durability')
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
        
#%%this part is callable to get the desired file type ##do not mix file flags when naming files
    def file_collect(self, dtype, directory):        
        # get list of all requested files types
        path, tail= os.path.split(directory)
        name_keyword = {'hfr':['hfr','HFR'], #these names are integral to the script detecting the correct files for use. do not mix naming conventions 
                    'hold': ['BB', 'DS'],
                    'crossover':['xover'],
                    'nyquist':['V', 'Acm2']}
        file_type = {'hfr':['.csv'], #these are native the software which creates the filetypes (should remain constant), 
                    'hold':['.csv'],
                    'crossover':['.csv'],
                    'nyquist':['.csv']}
        paths = []
        for dirpath, dirname, files in os.walk(directory):            
            for x in files:
                for v in name_keyword[dtype]:
                    for end in file_type[dtype]:
                        if v in x and x.endswith(end):
                            paths.append(os.path.join(dirpath, x))
                                                          
        return(paths) #when called, this function yields list of all file paths which match keyword and end string


#%%this part extracts true test duration and creates df for use as well as being individually callable to plot voltage over time

    def test_stand_data(self):        
        #to open hold files and determine real elapsed time
        files = self.file_collect('hold', self.directoryIN)
        self.holds_df = pd.DataFrame(columns=['file id', 'hold start', 'hold duration', 'total'])
        if len(files) > 0: 
            df_list = []
            for i, file in enumerate(files):
                path, tail = os.path.split(file)
                #this will read BB files and rename the columns to the logger naming convention (this is what came first so...)
                columns = ['Timestamp','CELL_V_FB','PS_CURRENT_DENSITY_FB','PS_CURRENT_DENSITY_SP','MATRIX_WORKSHEET']
                #takes a data point ever 2 minutes. this drastically speeds up data processing of BB files
                df0 = pd.read_csv(file, usecols=columns)
                df0.rename(columns={'CELL_V_FB':'cell pot','PS_CURRENT_DENSITY_FB':'current density', 'PS_CURRENT_DENSITY_SP':'current density sp', 
                                    'MATRIX_WORKSHEET':'program'}, inplace=True) 
                df0['Timestamp'] = pd.to_datetime(df0['Timestamp']) 
                grouped = df0.groupby('program')
                dfs = [group for _, group in grouped]
                for f in dfs:
                    program = f['program'].iloc[0]
                    f = f.drop('program', axis=1)
                    f = f.resample('1min', on='Timestamp').mean().reset_index()
                    f['program'] = [program]*len(f)
                    df_list.append(f)   
            resamp_df = pd.concat(df_list)
            resamp_df = resamp_df.sort_values(by='Timestamp')
            resamp_df=resamp_df.reset_index(drop=True)            
            
            if self.mode == 'breakin':
                breakin = resamp_df[resamp_df['program'].isin(self.cond_programs)].copy()                
                dir_out = os.path.join(self.directoryOUT, 'Conditioning')
                os.mkdir(dir_out) if not os.path.exists(dir_out) else None
                                    
                fig, ax = plt.subplots(figsize=(10,5))
                ax1 = ax.twinx()
                ax.set_xlabel('time')
                ax.set_ylabel('potential (V)')
                ax1.set_ylabel(r'Current density $(A/cm^2)$', color='blue')
                ax.scatter(breakin['Timestamp'], breakin['cell pot'], color=self.c, label=self.cell[0])
                ax1.plot(breakin['Timestamp'], breakin['current density'], 'b:', label='current density')
                labels = ax.get_xticklabels()
                plt.setp(labels, rotation=45, horizontalalignment='right')    
                ax.xaxis.set_major_formatter(md.DateFormatter('%m/%d %H'))
                fig.tight_layout()
                fig.savefig(os.path.join(dir_out,f'{self.cell[0]} break in.png'), dpi=400)
                plt.close()
                #to plot pol curves over time
                df_filtered = breakin[breakin['current density sp'].shift() != breakin['current density sp']]
                fig1, ax1 = plt.subplots(figsize=(6,5))
                ax1.set_ylabel('Potential (V)')
                ax1.set_xlabel(f'Current density $(Acm^{{{{-2}}}})$')
                ax1.grid(which='major', color='#DDDDDD', linewidth=0.8)
                colors = np.arange(len(df_filtered))
                s = ax1.scatter(df_filtered['current density'], df_filtered['cell pot'], c=colors, cmap='cool')
                cbar = fig1.colorbar(s, ax=ax1, ticks=[colors.min(), colors.max()], orientation='horizontal')
                cbar.ax.set_xticklabels(['First', 'Last'])  # vertically oriented colorbar
                ax1.set_ylim(1.4,None)
                fig1.tight_layout()
                fig1.savefig(os.path.join(dir_out,f'{self.cell[0]}-pol break in.png'), dpi=400)
                plt.close()
                #to extract the final pol curve and export breakin data
                dfs = np.array_split(df_filtered, 10)
                dfs[-1].to_csv(os.path.join(dir_out,f'{self.cell[0]}-lastPOL.csv'))
                breakin.to_csv(os.path.join(dir_out,f'{self.cell[0]}-fillbreakin.csv')) 
                print('\n','breakin exported')       
            else:
                dur_data = resamp_df[resamp_df['program'].isin(self.hold_programs)].copy()
                hold_data = dur_data[dur_data['current density sp'].between(self.hold_J-0.05,self.hold_J+0.05)]
                hold_data = hold_data.reset_index(drop=True)
                hold_data['time_diff'] = hold_data['Timestamp'].diff()
                #this splits the dataframe into separate dfs coordinated to each hold between hfr measurements
                gap_dates = [pd.to_datetime(x,format='%y%m%d %H%M%S') for x in self.eis_dates]              
                if not gap_dates:
                    holds = [hold_data]
                else:
                    holds = [hold_data[(hold_data['Timestamp'] >= date) & 
                                (hold_data['Timestamp'] < gap_dates[d+1])] 
                                for d, date in enumerate(gap_dates[:-1])]

                    last = hold_data[hold_data['Timestamp'] >= gap_dates[-1]]
                    if last:
                        holds.append(last)
                            
                #this determines real hold duration per hold in holds  
                self.holds_xref = pd.DataFrame(columns=['hold #', 'hold duration', 'total', 'cell pot'])
                hold_time = []
                total_hours = {}
                for f, df in enumerate(holds):
                    df.reset_index(drop=True)
                    df = df[df['time_diff']<pd.Timedelta(minutes=20)]
                    hold_hours = df['time_diff'].sum().total_seconds()/3600
                    hold_time.append(hold_hours)                
                    total_hours[f] = sum(hold_time)
                    #create plotting x column with values in hours
                    start_time = total_hours[f-1] if f > 0 else 0
                    end_time = total_hours[f]
                    x_hours = np.linspace(start_time, end_time, num=len(df), endpoint=False)
                    df['hours'] = x_hours
                    self.holds_xref.loc[len(self.holds_xref.index)] = [f+1, hold_time[f], total_hours[f], df['cell pot'].iloc[-1]]    
                
                if self.mode == 'v(t) plots':    
                    print(self.holds_xref) 

                self.hold_data = holds #makes list of data frames        
        else:
            print('no hold files found')
       
       
#%%
    def durability_plot(self):
        dir_out = os.path.join(self.directoryOUT, 'durability holds')
        os.mkdir(dir_out) if not os.path.exists(dir_out) else None
            
        #this will make output excel files of all data
        for f, df in enumerate(self.hold_data):            
            df_cp = df[['Timestamp', 'current density', 'cell pot', 'hours']].copy()
            df_cp.to_excel(os.path.join(dir_out, f'durability_data{f+1}.xlsx'), index = False)
        
        #plots each individual hold file(s) versus date to verify the durability plot
        #will make new plot if they don't already exist at expected file path
        if self.mode == 'v(t) plots':
            for f, df in enumerate(self.hold_data):
                save_to = os.path.join(dir_out,f'hold{f+1}.png')
                if os.path.exists(save_to) is False or f==len(self.hold_data)-1:            
                    fig, ax = plt.subplots()
                    ax.set_ylabel('Potential (V)')
                    ax.set_title(f'hold{f+1}')
                    ax.scatter(df['Timestamp'], df['cell pot'], color=self.c,s=3, label=self.cell[0])
                    labels = ax.get_xticklabels()
                    plt.setp(labels, rotation=45, horizontalalignment='right') 
                    if self.holds_xref['hold duration'].iloc[f] > 100:
                        fig.set_size_inches(10, 5)
                        ax.set_xlabel('date')
                        ax.xaxis.set_major_formatter(md.DateFormatter('%m/%d'))
                    else:
                        ax.set_xlabel('date + time')
                        ax.xaxis.set_major_formatter(md.DateFormatter('%m/%d %H'))
                    ax.legend() 
                    fig.savefig(save_to, bbox_inches='tight', dpi=400)   
                    plt.close()       


        #create durability plots which show 1: potential & shutdowns
        note = ''
        end = self.holds_xref['total'].iloc[-1]
        fig_length = 15 if end > 500 else 10
        fig, ax = plt.subplots(figsize=(fig_length,5))
        ax.set_ylabel('Potential (V)')
        ax.set_xlabel('Time (hours)')         
        for f, df in enumerate(self.hold_data): 
            ax.scatter(df['hours'], df['cell pot'], color=self.c, s=2, label=self.cell[0])
            if f == len(self.hold_data)-1:
                ax.set_xlim(0,df['hours'].iat[-1]) 
                ax.set_ylim(1.76, 1.01*max(df['cell pot']))
        if self.mode == 'hfr durability':
            note = '_annotated'          
            df_filtered = self.cell_df[self.cell_df['Current density'].between(self.hold_J-0.05,self.hold_J+0.05)].copy()  
            sizes = [200] * len(df_filtered)
            ax.scatter(df_filtered['age (hours)'], df_filtered['Voltage'],marker='*', color=self.c, s=sizes, label=f'Pol Curve - {self.hold_J}${{{{Acm^{-2}}}}}$')
        legend = ax.legend(fontsize='large')# bbox_to_anchor=(1, 0.5))
        for i in legend.legend_handles:
            i._sizes = [50]
        fig.savefig(os.path.join(dir_out, f'durability{note}.png'), bbox_inches='tight', dpi=400)

            
#%%this part handles crossover data
    def crossover_extraction(self):
        files = self.file_collect('crossover', self.directoryIN.get())
        df_list = []
        if len(files) > 0:
            for i, file in enumerate(files): #should be adjusted based on the format of your hfr files
                df = pd.read_csv(file)
                df['file'] = [file] * len(df)
                df_list.append(df)
            xover_df = pd.concat(df_list)
            xover_df.sort_values(by = ['date'], inplace=True)
            #this groups all data by run# (chronological order of holds)
            #and makes a list of data frames where the first DF = all BoL data, df 2  = all run2 data, etc
            grouped = xover_df.groupby(pd.Grouper(key='Date', axis=0,  
                      freq='D', sort=True)).sum()
            run_data = [group for _, group in grouped] 
            #this determines which hold this data applies to
            for i, df in enumerate(run_data):
                total_age = self.holds_xref.at[self.holds_xref['hold #']==i,'total'].values[0] if i != 0 else 0
                df['age (hours)'] = [total_age] *  len(df)
            self.crossover_data = run_data

            #Exports hfr data as a big df
            name = self.cell[0]
            xover_df.to_csv(os.path.join(self.directoryOUT, f'{name}-study.csv'), index=False)


#%%this part creates an hfr dictionary for each test hour condition
    def hfr_extraction(self):
        files = self.file_collect('hfr', self.directoryIN.get())
        df_list = []

        if len(files) > 0:
            for i, file in enumerate(files): #should be adjusted based on the format of your hfr files
                path, tail = os.path.split(file) 
                name_items = tail.split('-')
                run = int(name_items[1].replace('r',''))
                df = pd.read_csv(file)
                df.drop([0], inplace = True) 
                df.drop(['HFR-ohm','Fit regime'], axis=1, inplace = True) 
                df[['Current density','Voltage', 'HFR','R_cl']] = df[['Current density','Voltage', 'HFR','R_cl']].astype(float)
                df['ir free'] = df['Voltage'] - df['Current density']*df['HFR']
                df['run #'] = [run] * len(df)
                df['file'] = [file] * len(df)
                df_list.append(df)
            hfr_df = pd.concat(df_list)
            hfr_df.sort_values(by = ['run #', 'Current density'], na_position='first', inplace=True)
            
            #this groups all data by run# (chronological order of holds)
            #and makes a list of data frames where the first DF = all BoL data, df 2  = all run2 data, etc
            grouped = hfr_df.groupby('run #')
            run_data = [group for _, group in grouped]    
            
            for i, df in enumerate(run_data):
                total_age = self.holds_xref.at[self.holds_xref['hold #']==i,'total'].values[0] if i != 0 else 0
                df['age (hours)'] = [total_age] *  len(df)
                    
            self.run_data = run_data
            self.cell_df = pd.concat(run_data)
            self.cell_df.sort_values(by = ['run #', 'Current density'], na_position = 'first', inplace=True)

            #Exports hfr data as a big df
            name = self.cell[0]
            self.cell_df.to_csv(os.path.join(self.directoryOUT, f'{name}-study.csv'), index=False)
                
        else:
            print('no hfr files found')
        
                                        
#%%this part plots transience at each different time stamp and calculates 'exciting' data

    def characterization_plotting(self):
        label = self.cell[1]
        
        dir_out = os.path.join(self.directoryOUT,'Cell transience')
        os.mkdir(dir_out) if not os.path.exists(dir_out) else None
            
        all_plot = [[r'HFR $(m\Omega*cm^2)$','HFR']]
        if self.mode == 'hfr_durability':
            cell_df = self.cell_df.copy()
            all_plot.append([r'Catalyst Layer Resistance $(m\Omega*cm^2)$', 'R_cl'])
            #this plots hold vs characterization voltage at the same current density
            age = [0]
            age.extend(self.holds_xref['total'].to_list())
            cell_df.sort_values(by = ['age (hours)', 'Current density'], na_position = 'first', inplace=True)
            fig1, ax1 = plt.subplots(1,1, figsize=(5,5))
            ax1.set_xlabel('Time (hours)')
            ax1.set_ylabel('Cell Potential (V)')
            ax1.plot(self.holds_xref['total'],self.holds_xref['cell pot'],c=self.c,marker='o',label=f'hold ({self.hold_J}$Acm^{{{{-2}}}}$)')
            new_df = cell_df[cell_df['Current density'].between(self.hold_J-0.05, self.hold_J+0.05)].copy()
            ax1.plot(new_df['age (hours)'],new_df['Voltage'],c=self.c,marker='s',ls='--',label='Characterization')
            ax1.legend()
            fig1.savefig(os.path.join(dir_out,f'{label} hold_V_EIS.png'), bbox_inches='tight', dpi=400)
            plt.close()

            #this plots the classic pol + ir free over hfr plot at every characterization
            fig, axs = plt.subplots(2,1, sharex=True, figsize=(6,8), gridspec_kw={'height_ratios': [3,1]})
            axs[0].set_ylabel('Potential (V)')
            axs[1].set_ylabel(r'HFR $(\Omega*cm^2)$')
            axs[1].set_xlabel(r'Current density $(A/cm^2)$')
            axs[0].grid(which='major', color='#DDDDDD', linewidth=0.8)
            axs[1].grid(which='major', color='#DDDDDD', linewidth=0.8)
            axs[0].hlines(xmin=2.5, xmax=3.2, y=1.45, color = 'black', linestyles="dashed")
            axs[0].text(x=3.5, y=1.45, s='hfr-free', ha='center', va='center', backgroundcolor='white')
            ages = list(set(df['age (hours)'].to_list()))
            ages.sort()
            for a, hour in enumerate(ages):
                df0 = cell_df[cell_df['age (hours)']==hour].copy()
                axs[0].plot(df0['Current density'], df0['Voltage'], c=self.p_sequential[a], marker='o', linestyle='solid', label=f'{hour}hour')    
                axs[0].plot(df0['Current density'], df0['ir free'], c=self.p_sequential[a], linestyle='dashed')  
                axs[1].plot(df0['Current density'], df0['HFR'], c=self.p_sequential[a], linestyle='solid')                  
            axs[0].legend()                
            fig.savefig(os.path.join(dir_out,f'transience.png'), bbox_inches='tight', dpi=300)
            plt.close()

            all_df = self.cell_df.copy()

            #save_to = os.path.join(dir_out, f'{label} vba')
            #self.vba.analyses(df,'run',save_to)
            #loss_dict = self.vba.datastore()

        else:
            xover = self.crossover_data.copy()
            #this section plots the bol and eot comparison  
            to_plot = {'Voltage(V)': ['Voltage (V)', 'Voltage'],
                        'H2:O2 (%)': [r'$H_2 : O_2 (\%)$', 'H2inO2']} 
            for i, (k,v) in enumerate(to_plot.items()): 
                fig, ax = plt.subplots() #to plot flux / 
                ax.set_xlabel(r'Current density ($A/cm^2$)') 
                ax.set_ylabel(v[0])
                df00 = xover[0].copy()
                df00['P_SP'] = np.round(df00['cathode pressure (bar)'],0)
                grouped = df00.groupby('P_SP')
                dfs_list = [data for _, data in grouped]
                for j, df in enumerate(dfs_list):
                    p = df['P_SP'].iat[0]
                    label = f'{p}bar {self.temperature}C'                                              
                    ax.plot(df['j (A/cm2)'], df[k], marker='o', label=label)        
                ax.legend()
                fig.savefig(os.path.join(self.directoryOUT, f'BOL - {v[1]}.png'), dpi=300)
                plt.close() 

            #This section plots current density specific over time comparisons at 30 bar        
            if len(self.holds_df) > 1:
                all_df = pd.concat(xover)
                all_df = all_df[all_df['cathode pressure (bar)'].between(self.pressure-2,self.pressure+2)]

                #this section plots all plots compared with current density at the pressure testing is done at
                to_plot = [['Voltage(V)','Voltage (V)', 'Voltage over time'],
                            ['H2:O2 (%)',r'$H_2 : O_2 (\%)$', 'H2inO2 over time']]
                for p, plot in enumerate(to_plot):
                    fig, ax = plt.subplots() #to plot flux / 
                    ax.set_xlabel(r'Current density ($A/cm^2$)') 
                    ax.set_ylabel(to_plot[1])              
                    for i, df in enumerate(xover):
                        df = df[df['cathode pressure (bar)'].between(self.pressure-2,self.pressure+2)]
                        age = df['age (hours)'].iat[0]
                        ax.plot(df['j (A/cm2)'], df[to_plot[0]], marker=self.m[i], label=f'{age}hours', color=self.p_sequential[i])
                    ax.legend()
                    fig.savefig(os.path.join(self.directoryOUT, f'{to_plot[2]}.png'), dpi=300)
                    plt.close()

                #this will plot the remaining standard plots of the excel sheet tab 1 - all BoL data sets
                #to_plot dictionary follows format: {'column title': ['axis label', 'plot name']}
                to_plot = {'J_H2_xover(mA/cm2)': [r'H2 Crossover Flux ($mA_{eq}/cm^2$)', 'H2 flux over time'],
                            'H2:O2 (%)': [r'$H_2 : O_2 (\%)$', 'H2inO2 over time']}
                #this plots all plots related to time
                for i, (k,v) in enumerate(to_plot.items()): 
                    fig1, ax1 = plt.subplots() #to plot flux / 
                    ax1.set_xlabel('time (hours)') 
                    ax1.set_ylabel(v[0]) 
                    grouped = all_df.groupby('J_SP')
                    dfs_list = [data for _, data in grouped]
                    for j, df in enumerate(dfs_list):    
                        cd = df['J_SP'].iat[0]                
                        ax1.plot(df['age (hours)'], df[k], marker=self.m[j], label=f'{cd}$Acm^{{{{-2}}}}$', color=self.p_sequential[j])        
                    ax1.legend()
                    fig1.savefig(os.path.join(self.directoryOUT, f'{v[1]}.png'), dpi=300)
                    plt.close()
            
        all_df = all_df.groupby('age (hours)').mean()
        data = ['dashed', 'solid']
        for p, plot in enumerate(all_plot):
            fig, ax = plt.subplots()
            ax.set_ylabel(plot[0])
            ax.set_xlabel(r'Time (hours)')
            if p == 1:
                index = all_df[all_df['R_cl']==0].index
                all_df.drop(index, inplace=True)
            ax.plot(all_df.index, all_df[plot[1]], c=self.c, marker='o', linestyle=data[2])  
            fig.savefig(os.path.join(dir_out,f'{plot[1]} over time.png'), bbox_inches='tight', dpi=400)
            plt.close()

    
#%%the machine that runs it all

    def RunIt(self): 
        #Establish Cell Characteristics
        #Either read in or export out Cell MetaData.xlsx
        self.metadata(self.directoryIN.get())

        #create processed folder for all data to go into
        self.directoryOUT = os.path.join(self.directoryIN.get(), 'Processed Durability')
        os.mkdir(self.directoryOUT) if not os.path.exists(self.directoryOUT) else None
            
        self.mode = self.var.get()   
        self.test_stand_data()                       
                
        if self.mode == 'v(t) plots':
            self.durability_plot()
            print('\n','V(t) exported')
            
        if self.mode == 'hfr durability':
            #self.vba = vba(self.tafel,self.cell) 
            self.hfr_extraction()           
            self.characterization_plotting()
            self.durability_plot()
            print('\n','hfr exported')

        if self.mode == 'crossover durability':
            self.crossover_extraction()
            self.characterization_plotting()
            print('\n','crossover exported')
        
        self.metadata_export(self.directoryIN.get())
        

if __name__ == '__main__':

    root = Tk()
    App = Durability(root)  
    root.mainloop()