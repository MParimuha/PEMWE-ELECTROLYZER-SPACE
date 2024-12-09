###########################################
#this script was developed by Makenzie Parimuha for high pressure crossover analyses
#if you have any questions, email Makenzie.Parimuha@nrel.gov
###########################################

from tkinter import *
import tkinter as Tkinter
import tkinter.filedialog as tkFileDialog
#import matplotlib.dates as md
#from matplotlib.dates import AutoDateLocator, AutoDateFormatter
#matplotlib.use('TkAgg')
from datetime import timedelta, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

#%%
class cellcharacteristics:
    def __init__(self):
        #BB protocols is the list of MATRIX_COMMENT entries. add all matrix comments used
        #if test stand == green light, leave as be. it will not be used
        self.BB_protocols = ['25cm^2 -> 2A-cm2','Standard Protocol']

class xoveranalyses(cellcharacteristics):
    def __init__(self, master):
        super().__init__() 

        master.title("Crossover")
        self.master = master
        master.geometry("650x230")
        buttonFrame = Frame(master, pady=10)
        buttonFrame.grid(row=0,columnspan=2)  
        self.directoryIN = Entry()
        self.ModeSelectBox = Entry()
      
#%%to make the GUI
        self.var = StringVar(master)

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
        self.ccmarea = Entry(buttonFrame, width = 10)
        self.ccmarea.insert(0, '')
        self.ccmarea.grid(row=3, column=1, sticky =W)

        # String entry box for the name of the output file. 
        Label(buttonFrame, text = "Output file name:").grid(row = 4, column = 0)
        self.OFile = Entry(buttonFrame, width = 40)
        self.OFile.insert(0, "")
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
        
#%%this part is callable to get the desired file type ##do not mix file flags when naming files
    def file_collect(self, dtype):
        # get list of all requested files types
        name_keyword  ={'hfr':['hfr','HFR'], #these names are integral to the script detecting the correct files for use. do not mix naming conventions
                    'GC':['Samp', 'SAMP'], 
                    'test_stand':['Crossover','crossover', 'BB']}
        
        file_type = {'hfr':['.csv'], #these are native the software which creates the filetypes (should remain constant)
                    'GC':['.txt', '.TXT'], 
                    'test_stand':['.csv', '.xlsx']}
        paths = []
        for dirpath, dirname, files in os.walk(self.directoryOUT):            
            for x in files:
                for v in name_keyword[dtype]:
                    for end in file_type[dtype]:
                        if v in x and x.endswith(end):
                            paths.append(os.path.join(dirpath, x))                                       

        return(paths) #when called, this function yields list of all file paths which match keyword and end string

        
#%%this part creates a xover dataframe from gas chromatography files and determines which testing conditions are included in selected folder          
    def GC_data(self): 
        #run file_collect func to get appropriate file types / with keyword
        files = self.file_collect('GC')
        specific_rows= [2,3,4,5]
        
        GC_df = pd.DataFrame(columns=['path','date', 'H2 Area', 'O2 Area'])
        for i, file in enumerate(files):
            df = pd.read_csv(file, skiprows= lambda x: x not in specific_rows, delimiter='\t')
            if not df.empty:
                path, tail =os.path.split(file)
                dir_path = os.path.dirname(path)
                #extract date, and time from file name or path name (directory should be names appropriately)
                #this section handles GL gc crossover data
                if '.txt' in tail:
                    date = str(re.search(r'\d+(?=_)', path).group())
                    time = str(re.search(r'\d+(?=.txt)', path).group())
                    date = pd.to_datetime(date + ' ' + time, dayfirst=True)
                    elementdict = dict(zip(list(df['Component']), list(df['Area']))) 
                    areas =[elementdict['hydrogen'],elementdict['oxygen']]
                    self.H2coeff = 0.0000268737139
                    self.O2coeff = 0.0003007354501
                    self.gc = 'GL'
                #this section handles cart data    
                if '.TXT' in tail:
                    gcfile = open(file, "r+")
                    content = gcfile.readlines()
                    date = None
                    line = 11
                    while not date and line < 15:
                        if re.search('(\d+)-(\d+)-(\d+)', str(content[line])):
                            date = str(re.search('(\d+)-(\d+)-(\d+)', str(content[line])).group())
                            time = str(re.search('([0-1]?\d|2[0-3]):([0-5]?\d):([0-5]?\d)', str(content[line+1])).group())
                        line += 1
                    gcfile.close()
                    date = pd.to_datetime(date + ' ' + time, dayfirst=True)
                    date = date + timedelta(minutes=49)
                    areas = [df['Area'].iloc[0], df['Area'].iloc[1]] 
                    self.H2coeff_high = 0.0000164752
                    self.H2coeff_low = 0.0000146156
                    self.O2coeff = 0.0001622275
                    self.gc = 'cart'
                GC_df.loc[len(GC_df.index)] = [dir_path, date, areas[0],areas[1]]               

        #this runs function to merge BB data and GC data
        # #adds elapsed time column
        GC_df['time_diff'] = GC_df['date'].diff()
        total_min = GC_df["time_diff"].sum().total_seconds()/60
        GC_df['elapsed time (min)'] = [i*(total_min/len(GC_df)) for i in range(len(GC_df))]
        GC_df.sort_values('date', inplace=True)
                       
        merged_df = self.PS_data(GC_df) 
        mean_hfr = self.hfr_extraction()
        merged_df['hfr (ohm*cm2)'] = [mean_hfr] * len(merged_df)
        
        return(merged_df)


#%%this part combines PS station files data with the GC chromatography data
    def PS_data(self, GC_df):
        area = float(self.ccmarea.get())
        files = self.file_collect('test_stand')
            
        df_list = []
        for file in files:
            #to remove degree symbol which cant be parsed by pandas
            path, tail = os.path.split(file)
            if 'BB' in tail:
                #this will read BB files and rename the columns to the logger naming convention 
                # (GL came first so...)
                df0 = pd.read_csv(file,encoding = "cp1252")
                df1 = df0[df0['MATRIX_WORKSHEET'].isin(self.BB_protocols)].copy()
                df1 = df1[['Timestamp', 'CELL_V_FB','PS_CURRENT_DENSITY_FB','PS_CURRENT_DENSITY_SP','TEMP_CELL_FB']]
                try:
                    df1['cathode pressure (bar)'] = df0.round[{'cat_P_FB':0}]
                except:
                    df1['cathode pressure (bar)'] = [0]*len(df1)
                df1.rename(columns={"Timestamp": "Time stamp", 'CELL_V_FB':'Voltage (V)', 
                                    'PS_CURRENT_DENSITY_FB':'j (A/cm2)', 'PS_CURRENT_DENSITY_SP':'J_SP', 
                                    'TEMP_CELL_FB':'temperature (C)'}, inplace=True) 
                df1['Time stamp'] = pd.to_datetime(df1['Time stamp'])
            else:
                #this processes GL files and keeps the GL naming convention
                df1 = pd.read_csv(file, skiprows=10,encoding = "cp1252",
                                  usecols=['Time stamp','cell_voltage_001','current','current_set','pressure_cathode_outlet','temp_cathode_outlet'])
                df1['current_set'] = df1['current_set']/area #sets this in A/cm2
                df1['current'] = df1['current']/area
                df1.rename(columns={'cell_voltage_001':'Voltage (V)','current':'j (A/cm2)', 
                                    'current_set':'J_SP','pressure_cathode_outlet':'cathode pressure (bar)',
                                    'temp_cathode_outlet':'temperature (C)'}, inplace=True) 
                df1 = df1.round[{'cathode pressure (bar)':0}]
                #standardize timestamp column / correct time
                df1['Time stamp'] = pd.to_datetime(df1['Time stamp']) + timedelta(minutes=19, seconds=30)
            df_list.append(df1)

        PS_df = pd.concat(df_list)
        PS_df.dropna(inplace=True)
        PS_df.sort_values('Time stamp', inplace=True)
        
        if self.gc == 'GL':
            #included this for GL GC data and mixed logger type data
            pass
        #this section handles the weird GC start time for instances where the time delay is not understood. 
        #assumes identical start time (which is true +- a few seconds)
        elif abs(GC_df['date'].iloc[0] - PS_df['Time stamp'].iloc[0]) > timedelta(minutes=1):
            start_time = PS_df['Time stamp'].iloc[0]
            GC_df['time_diff'].iloc[0] = timedelta(minutes=0)
            GC_df['date'] = start_time + GC_df['time_diff'].cumsum()
            GC_df.sort_values('date', inplace=True)

        #This line makes it so that the Conditions + GC dataframes merge based on the end time of the GC measurement
        #picks up on current density at the END of the crosssover measurement
        GC_df['date'] = GC_df['date'] + timedelta(minutes=1)
        # Merge based on the closest timestamp (backward)  
        # **dreamboat status / the most useful piece of code HOT DAMN
        merged_df = pd.merge_asof(GC_df, PS_df, left_on='date', right_on='Time stamp', direction='backward').reset_index()  
        merged_df.drop(['Time stamp', 'time_diff'], axis=1, inplace=True) 
        
        return(merged_df)

#%%this part calculates average hfr 
    def hfr_extraction(self):
        file = self.file_collect('hfr')
        if file:
            df = pd.read_csv(*file, usecols=['HFR'])
            df.drop([0], inplace = True)   
            df = df.astype(float)            
            mean_hfr = df['HFR'].mean()
            return(mean_hfr) 
        else:
            print('HFR files not found. Add manually')
            return(0)
    

    def calculations(self, df):
        #establish constants        
        n_ad = 0.9 #adiabatic efficiency for energy calculations
        Faraday = 96485 #Coulombs
        kwh = 3.6 #kWh/MJ conversion
        kelvin = 273.15 + 80 #operation temperature
        R = 8.314 # gas constant

        #clean up H2 Area values in order to avoid them in the 'clean' data  
        df = df[(df['H2 Area'] > 0)].copy()                                                       
        #data processing for J_H2_xover, FE, kWh/kg (with and without W_ad)
        #this determines if the H2 Areas are more high H2 or low H2
        low_areas = df['H2 Area'].between(113, 5103).sum()
        high_areas = df['H2 Area'].between(3177, 542916).sum()
        if low_areas>high_areas:
            print('\n<<LOW>> H2 Area dataset detected. \n<<LOW>> H2 Area coefficient used.')
            df['H2:O2 (%)'] = (df['H2 Area']*self.H2coeff_low)*100/(df['O2 Area']*self.O2coeff)
        else:
            print('\n<<HIGH>> H2 Area dataset detected. \n<<HIGH>> H2 Area coefficient used.')
            df['H2:O2 (%)'] = (df['H2 Area']*self.H2coeff_high)*100/(df['O2 Area']*self.O2coeff)
        df['J_H2_xover(mA/cm2)'] = 0.5*df['j (A/cm2)']*(df['H2:O2 (%)']/100)*1000
        df['FE'] = 1 - df['J_H2_xover(mA/cm2)']/1000/df['j (A/cm2)']
        df['Produced (kWh/kg)'] = 2*Faraday*df['Voltage (V)']/df['FE']/0.002/1E6/kwh
        df['Prod + W_ad (kWh/kg)'] = df['Produced (kWh/kg)'] + (n_ad**(-1)*5.2*R*kelvin/2/0.002)*((3.1E6/1E6)**(2/7)-1)/1E6 

        #extract final test at each current density for further comparison
        self.final = df.groupby('J_SP').tail(1).copy() 
        print(self.final)

        #plotting H2O2 with J over time 
        fig, ax = plt.subplots() #to plot transience
        ax.set_xlabel(r'Time $(min)$')
        ax.set_ylabel('H2:O2 (%)') 
        ax1 = ax.twinx()
        ax1.set_ylabel(r'J $(Acm^{-2})$', color = 'm')
        ax1.spines['right'].set_color('m')
        ax1.tick_params(axis='y', colors='m')
        ax1.yaxis.label.set_color('m')
        ax1.scatter(df['elapsed time (min)'], df['j (A/cm2)'], color='m', marker='s')  
        ax.scatter(df['elapsed time (min)'], df['H2:O2 (%)'], marker='o', color='k', label='data')
        ax.plot(self.final['elapsed time (min)'],self.final['H2:O2 (%)'], marker='X', markersize=8, color='k', ls='dotted', label = 'final')  
        ax.legend(loc='upper center')
        fig.savefig(os.path.join(self.directoryOUT, f'{self.OFile.get()} transience.png'), dpi=300)
        plt.close() 
         
                
#%% this will export clean data set             
    def data_export(self):
        name = self.OFile.get() + '-xoverdata.csv'   
        desired_col_order = ['date', 'temperature (C)', 'cathode pressure (bar)', 'j (A/cm2)', 'Voltage (V)', 'hfr (ohm*cm2)',
                                'H2 Area','O2 Area', 'H2:O2 (%)', 'J_H2_xover(mA/cm2)', 'FE', 'Produced (kWh/kg)', 'Prod + W_ad (kWh/kg)']
        self.final=self.final[desired_col_order]
        self.final.to_csv(os.path.join(self.directoryOUT, name), index=False)

        
#%%the machine that runs it all
    def RunIt(self):
        self.directoryOUT = self.directoryIN.get()
        #run function to process hold data and create dictionary with hold start times and hold duration
        merged_df = self.GC_data()
        self.calculations(merged_df)
        self.data_export() 

        print(f'\n\n{self.OFile.get()} has been exported')       

if __name__ == '__main__':

    root = Tk()
    App = xoveranalyses(root)  
    root.mainloop()
