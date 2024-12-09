###########################################
#this script was developed by Makenzie Parimuha for Gamry EIS / Pol curve / CV analyses
#if you have any questions, email Makenzie.Parimuha@nrel.gov
#most recently updated November 2024
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

class GAMRY_EISPROC:
    def __init__(self, master):

#%%making the GUI
        master.title("GAMRY EIS Processing V2.5")
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
        self.ModeSelectBox = OptionMenu(buttonFrame, self.var, 'EIS Manual Fit', "EIS Summary", 'HFR Fit Check', 'CV')
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
        self.ccmarea = Entry(buttonFrame, width = 15)
        #self.ccmarea.insert(0, '5')
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
        self.eis_dfs = []
        paths = []
        for dirpath, dirname, files in os.walk(self.directoryIN.get()):
            for x in files:
                if 'EIS' in x and x.endswith('.DTA'):
                    paths.append(os.path.join(dirpath, x))
        paths.sort()

        #determine if .DTA file is from old or new gamry (there are differences)
        #determine necessary parameters to open a dataframe correctly based on which old v new gamry 
        file1 = open(paths[0], 'r')
        content = file1.readlines()
        if 'CAPACITY' in str(content[8]): #multichannel file
            df_start = 59
            self.pol_start = 64
        elif 'DRIFTCOR' in str(content[15]): #new gamry if statement
            df_start = 61
            self.pol_start = 62
        else: #old gamry
            df_start = 55
            self.pol_start = 56
        file1.close()

        channels = []
        for f, File in enumerate(paths):
            path, tail = os.path.split(File)
            ch = 0
            if 'AECH' in tail:
                name = [x for x in tail.split('CH')]
                ch = int(name[1].replace('.DTA',''))
            channels.append(ch) 
            #open up data lines into dataframe
            df=pd.read_csv(File,encoding = "cp1252", skiprows=df_start, sep='\t')
            df.drop([0],inplace=True)
            df = df.astype(float)
            df.rename(columns={'Freq':'Frequency (Hz)', 'Zreal':'Z\' (Ω)', 'Zimag':'-Z\'\' (Ω)',	
                               'Zmod':'Z (Ω)',	'Zphz':'-Phase (°)'}, inplace=True)
            df['-Z\'\' (Ω)'] = df['-Z\'\' (Ω)']*-1
            df['Zimag'] = df['-Z\'\' (Ω)']
            df['Filename'] = [tail]*len(df)
            df['Channel'] = [ch]*len(df)
            self.eis_dfs.append(df)
        self.channels = list(set(channels))
 
#%%Process impedence
    def Impedance(self):
        #clear all data storage lists
        self.eis_df = pd.DataFrame(columns=['Current','HFR','HFR-ohm','Channel'])      
        FileOut = self.OFile.get()
            
        fig, ax = plt.subplots()
        ax.set_xlabel('Z\' (ohm)')
        ax.set_ylabel('-Z\'\' (ohm)')
        ax.set_title(FileOut)
        ax.set_aspect('equal')
        
        for f, df in enumerate(self.eis_dfs):
            path, tail = os.path.split(df['Filename'].iloc[0])
            ch = df['Channel'].iloc[0]
            if not df.empty:
                df['Zimag'] = df['-Z\'\' (Ω)']
                df['Zreal'] = df['Z\' (Ω)']
                current = df['Idc'].mean()
                plot_title = np.mean(current) / self.ccm_area
                
                if 'POT' in tail:
                    self.CL_resistance(df,ch)
                    continue  
                if f != 0 and f%2 == 0:
                    ax.scatter(df['Zreal'], df['Zimag'], s=2, label=f'{plot_title/self.ccm_area:.3}')                
                if self.var.get() == 'EIS Manual Fit':                   
                    dir_out = os.path.join(self.directoryOUT, 'manual fit')
                    os.mkdir(dir_out) if not os.path.exists(dir_out) else None 
                    file_name = f'{plot_title:.3f}Acm2' if ch == 0 else f'Ch{ch}-{plot_title:.3f}Acm2'                     
                    completepath = os.path.join(dir_out, f'{file_name}.csv')        
                    df.to_csv(completepath, index=False)                    
                else:
                    #will run if EIS or summary is selected. fits nyquist data
                    try: 
                        intercept = self.full_EIS(df,ch)
                    except:
                        print(f'unable to fit {plot_title:.3f}A cm-2 ({tail}) EIS')
                        intercept = 0 
                    data =  [current,intercept*self.ccm_area,intercept,ch] 
                    self.eis_df.loc[len(self.eis_df.index)] = data  

        if not self.eis_df.empty:
            self.eis_df = self.eis_df.sort_values(by=['Channel','Current']).reset_index() 
            self.eis_df.drop(['Current'],axis=1,inplace=True)  
        if not df.empty:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
            fig.savefig(os.path.join(self.directoryOUT, 'Nyquist_series.png'), dpi=300, bbox_inches='tight') 
            plt.close()

#%%Process CL resistance   
    def CL_resistance(self, df0,channel):
        dir_out = os.path.join(self.directoryOUT, 'manual fit')
        os.mkdir(dir_out) if not os.path.exists(dir_out) else None        
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

        #CL resistance plots
        plot_title = df['Vdc'].mean() 
        file_name = f'Rcl-{plot_title}V' if channel == 0 else f'Ch{channel}Rcl-{plot_title}V'                      
        fig1, ax1 = plt.subplots(figsize=(5,10))
        ax1.set_xlabel('RealZ (ohm)')
        ax1.set_ylabel('-ImagZ (ohm)')
        ax1.hlines(xmin=0.4*lin_func(0), xmax=1.3*max(lin_func(linear_range)), y=0, color = 'black', linestyles='dashed')         
        ax1.plot(df['Zreal'], df['Zimag'], color='k',ls='none',marker='o', label='R_cl HFR')        
        ax1.plot(rlZ, imgZ, c='magenta',marker='o', mfc='none', mec='magenta',ls='none')
        ax1.plot(lin_func(linear_range), linear_range, c='magenta', ls='--', label=f'{lin_func(0):.3g}')  
        if not df_low.empty:
            cl_r=3*self.ccm_area*(lin_func(0)-b)
            ax1.plot(fit_y, fit_x,marker='o', c='orange',mfc='none',mec='orange',ls='none')              
            ax1.scatter(b,0, c='orange', label=f'{b:.3g}')   
        else:
            cl_r='manual calculation required'
        ax1.set_xlim(0.4*lin_func(0), 1.3*max(lin_func(linear_range)))
        ax1.set_ylim(-0.03, 1.3*max(linear_range))
        ax1.set_aspect('equal')
        ax1.legend()
        fig1.savefig(os.path.join(dir_out, f'{file_name}_macro.png'),bbox_inches='tight', dpi=300) 
        #zoom into curved region
        ax1.set_xlim(0.8*lin_func(0), 1.05*lin_func(0))
        ax1.set_ylim(-0.005, 0.05*max(linear_range))
        fig1.savefig(os.path.join(dir_out, f'{file_name}_ZOOM.png'),bbox_inches='tight', dpi=300) 
        plt.close()
        self.rcl_df.loc[len(self.rcl_df.index)] = [cl_r, df['Channel'].iloc[0]] 
        #export to a csv and save the calculated Rcl value 
        df.to_csv(os.path.join(dir_out, f'{file_name}.csv'), index=False)
                
#%%
    def full_EIS(self, df,channel):
        intercept = 0
        #this block of code was developped for instances where the hfr tail goes positive 
        # after the hfr intercept
        #finds first instance of negative value in first 20 rows
        df_working = df.copy()
        has_neg = (df['Zimag'][:20] < 0).any()
        if has_neg:
            first_index = df['Zimag'].iloc[:20].lt(0).idxmax()
            df_working = df.loc[first_index:].copy()
        # split points up around intercept
        df_pos = df_working[df_working['Zimag']>0].copy() 
        cut_off = df_pos['Frequency (Hz)'].iloc[0]
        df_neg = df_working[(df_working['Zimag']<0)&(df_working['Frequency (Hz)']>cut_off)].copy()
        #setting up a lower and upper limit based on the last Z'' before crossing y=0
        up_lim = df_pos['Zreal'].iloc[0]
        low_lim = 0.9*up_lim
        
        #check for inductance range / and also that tail is not curling to the right
        if has_neg:        
            fit_y = [df_neg['Zreal'].iloc[-1], df_pos['Zreal'].iloc[0]]
            fit_x = [df_neg['Zimag'].iloc[-1], df_pos['Zimag'].iloc[0]]
            fit = np.polyfit(fit_x, fit_y, 1)
            if low_lim<fit[-1]<up_lim:
                intercept = fit[-1]
                function = np.poly1d(fit)
                regime = 'linear interpolation'
            else:
                fit_y = pd.concat([df_neg['Zreal'].iloc[-4:],df_pos['Zreal'].iloc[:4]], ignore_index=True)
                fit_x = pd.concat([df_neg['Zimag'].iloc[-4:], df_pos['Zimag'].iloc[:4]], ignore_index = True)
                fit = np.polyfit(fit_x, fit_y, 1)
                intercept = fit[-1]
                function = np.poly1d(fit)
                regime = 'messy data interpolation'

        #if nyquist incomplete, i.e. does not pass y=0, script will make best guess  
        else:                             
            #df_wip excludes all data points after Nyquist peak
            df_wip = df_pos[:8] if not has_neg else pd.concat([df_neg[-4:],df_pos[:8]])
            df_wip.sort_values(by=['Frequency (Hz)'])
              
            bestfunc = {}
            for o in [2, 3, 4]:
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
                fit_y = df_wip['Zreal']
                fit_x = df_wip['Zimag']
                func = np.poly1d(np.polyfit(fit_x, fit_y, 1))
                R2 = r2_score(fit_y, func(fit_x))    
                if low_lim<func(0)<up_lim:
                    intercept = func(0)
                    function = func
                    regime ='linear extrapolation'

        if self.var.get() == 'HFR Fit Check':  
            plotsout = os.path.join(self.directoryOUT, 'fit check figs')
            os.mkdir(plotsout) if not os.path.exists(plotsout) else None
            j = df['Idc'].mean()/self.ccm_area            
            plot_title = f'{j:.3f}Acm2' if intercept != 0 else f'{j:.3f}Acm2_error'
            title = plot_title if channel == 0 else f'CH{channel:.0f}-{plot_title}'
            #plot the fitted nyquist plots to check fitting quality  
            fig, ax = plt.subplots(dpi=200)
            ax.set_xlabel('RealZ (ohm)')
            ax.set_ylabel('-ImagZ (ohm)')
            ax.plot(df['Zreal'], df['Zimag'],ls='dashed',marker='o', label=f'{j:.3f}Acm2 data')
            ax.set_aspect('equal', 'box')
            if intercept != 0:
                ax.plot(fit_y, fit_x, ls='none',marker='o',mfc='none',mec='orange',mew=2.5, label='fit points')
                ax.plot(intercept, 0, 'r', marker = 'x', markersize=8)
                data = np.arange(0,max(fit_x), 0.0001)
                ax.plot(function(data), data, 'r', marker='o', markersize = 2,alpha=0.3, label=regime)                 
            ax.legend()                 
            fig.savefig(os.path.join(plotsout, f'{title}.png'), dpi=200, bbox_inches='tight') 
            if (max(df['Zreal'])-min(df['Zreal'])) > 0.015:
                ax.set_xlim(1.05*low_lim, 1.1*up_lim)
                imZ_lim = df_neg['Zimag'].iloc[-1]
                ax.set_ylim(8*imZ_lim,-8*imZ_lim)
                fig.savefig(os.path.join(plotsout, f'{title}-HFRzoom.png'), dpi=200, bbox_inches='tight') 
            plt.close()
                        
        return(intercept)

#%%process pol curve files
    def Polarization(self):
        # Clears the lists every time the Import is called.
        self.pol_df = pd.DataFrame(columns=['Current density', 'Voltage','Channel'])
        pol_path = []
        #this is specific to the Gamry script
        for dirpath, dirname, files in os.walk(self.directoryIN.get()):
            for x in files:
                if 'polcurve' in x and x.endswith('.DTA'):
                    pol_path.append(os.path.join(dirpath, x))
        pol_path.sort()
        #checks if there is an extra polcurve file (i.e. pol27) and deletes it
        if len(pol_path) > len(self.eis_dfs):
            del pol_path[-1]

        for f, File in enumerate(pol_path):
            for channel in self.channels:
                path, tail = os.path.split(File)
                df=pd.read_csv(File, skiprows=self.pol_start,encoding = "cp1252", sep='\t')
                #for data export and pol curve plot
                volt_name = 'Vf' if channel == 0 else f'Vf{int(channel)}'
                df=df[['Im',volt_name]].copy()
                df.drop([0],inplace=True)
                df = df.astype(float)
                data =  [df['Im'].mean()/self.ccm_area,df[volt_name].mean(),channel] 
                self.pol_df.loc[len(self.pol_df.index)] = data 
                volt_sd = df[volt_name].std()  
                #this function checks if the pol curve is unstable over 60s (i.e variation > 10%)
                if volt_sd/df[volt_name].mean() > 0.1:
                    print(f'\nCheck {tail} file. Unstable Pol curve.')
        self.pol_df = self.pol_df.sort_values(by=['Channel','Current density']).reset_index()

        if self.var.get() == 'EIS Manual Fit':
            fig, ax = plt.subplots()
            ax.set_title(self.OFile.get())
            ax.set_ylabel('Potential (V)')
            ax.set_xlabel(r'J $A/cm^2$')
            for channel  in self.channels:
                curve = self.pol_df[self.pol_df['Channel']==channel].copy()
                ax.plot(curve['Current density'], curve['Voltage'], marker='o', label=f'Ch{channel}')
            ax.legend() if len(self.channels)>1 else None
            fig.savefig(os.path.join(self.directoryOUT, f'pol.png'), dpi=300)
            plt.close()
                        
#%%extract final CV cycles from benchmarking CV files
    def CVAnalyses(self):
        # Clears the lists every time the Import is called. 
        directoryIN = self.directoryIN.get()
        cv_path = []
        for dirpath, dirname, files in os.walk(directoryIN):
            for x in files:
                if 'mVps' in x and x.endswith('.DTA'):
                    cv_path.append(os.path.join(dirpath, x))      
        final_curves = []
        for i, File in enumerate(cv_path):
            path, tail = os.path.split(File)
            scanrate = int(re.search('\d{2,3}', tail).group())
            df=pd.read_csv(File, skiprows=63,usecols=['Vf','Im'],encoding = "cp1252", sep='\t')
            df = df.drop([0])
            indices = df[df.apply(lambda row: row.astype(str).str.contains('Vf').any(), axis=1)].index.tolist()
            indices.append(len(df))
            curves = []
            for i in range(len(indices) - 1):
                start_idx = indices[i]
                end_idx = indices[i + 1]
                header = df.iloc[start_idx].values
                dataframe = pd.DataFrame(df.iloc[start_idx+1:end_idx].values, columns=header)
                curves.append(dataframe)
            curve_l = [len(curve) for curve in curves]
            final = curves[-1] if curve_l[-1]>curve_l[-2] else curves[-2]
            final['scan rate (mVps)'] = [scanrate]*len(final)
            final_curves.append(final)
        export_df = pd.concat(final_curves)
        export_df.to_csv(os.path.join(self.directoryOUT, self.OFile.get()+'-CV.csv'),index=False)
                         
#%%Export combined Pol and EIS data
    def Summarize(self):
        directoryOUT = self.directoryOUT
        FileOut = self.OFile.get()+'-'+str(self.ccm_area)+'cm2'
        mc = True

        tafel_r = pd.DataFrame({'VBA range': ['(start,end)']})
        self.eis_df = self.eis_df.drop(['Channel'], axis=1)
        export_df = pd.concat([self.pol_df,self.eis_df,tafel_r],axis=1,sort=False)
        export_df['hfr free'] = export_df['Voltage']-export_df['HFR']*export_df['Current density']
        export_df = export_df[['Channel','Current density','Voltage','HFR','hfr free','HFR-ohm','VBA range']]
        if not self.rcl_df.empty:
            export_df = pd.concat([export_df,self.rcl_df], axis=1)
        if len(self.channels) == 1:
            export_df = export_df.drop(['Channel'],axis=1)
            mc = False
        
        completeName = os.path.join(directoryOUT, FileOut+'.csv')
        if os.path.exists(completeName) and self.var.get() == 'EIS Manual Fit':
            pass
        elif self.var.get() == 'EIS Summary':
            export_df.to_csv(completeName, index=False) 

            colors = ['#4477AA', '#EE6677', '#228833', '#66CCEE', '#AA3377']
            fig, axs = plt.subplots(2,1,sharex=True,figsize=(6,8),gridspec_kw={'height_ratios': [3,1]})
            axs[0].set_ylabel('Potential (V)')
            axs[0].hlines(xmin=3.2, xmax=3.5, y=export_df['Voltage'].iloc[0], color='black', linestyles="dashed")
            axs[0].text(x=3.75, y=export_df['Voltage'].iloc[0], s='hfr-free', ha='center', va='center', backgroundcolor='white')
            axs[1].set_xlabel(r'Current density $Acm^{-2}$')
            axs[1].set_ylabel(r'HFR ($m\Omega*cm^2$)')    
            axs[1].set_ylim(0.5*export_df['HFR'].mean()*1000, 1.5*export_df['HFR'].mean()*1000) 
            for ax in axs: 
                ax.grid(color='#7f7f7f', linestyle='--', linewidth=0.5)         
            for ch in self.channels:
                df = export_df.copy() if not mc else export_df[export_df['Channel']==ch].copy()
                axs[0].plot(df['Current density'], df['Voltage'], 'o', color=colors[ch], label=f'Ch{ch}')
                #remove the uncalclable lines from the ir-free and hfr plots
                df = df[df['HFR'] > 0].copy()
                axs[0].plot(df['Current density'], df['hfr free'], '--',color=colors[ch])
                lasty=max(df['Voltage'])
                lastx=max(df['Current density'])
                axs[0].annotate(f'{lasty:.2f}V', (lastx - 0.5, lasty), ha='center',color=colors[ch])
                axs[1].plot(df['Current density'], df['HFR']*1000, 'o',color=colors[ch])
            axs[0].legend() if mc else None
            plt.tight_layout = True
            fig.savefig(os.path.join(directoryOUT, 'eis summary.png'),bbox_inches='tight', dpi=300)
            plt.close()
            
#%%the machine that runs it all
    def RunIt(self):        
        self.rcl_df = pd.DataFrame(columns=['R_cl', 'Channel'])
        self.ccm_area = float(self.ccmarea.get())

        #create processed folder for all data to go into
        self.directoryOUT = os.path.join(self.directoryIN.get(), 'Processed EIS')
        os.mkdir(self.directoryOUT) if not os.path.exists(self.directoryOUT) else None 
            
        if self.var.get() == "CV":
            self.CVAnalyses()
            
        else: 
            self.file_clean_up()
            self.Impedance()
            if self.var.get() == "EIS Summary" or self.var.get() == "EIS Manual Fit":
                self.Polarization()
                self.Summarize()
        
        print ('\n script finished!')

if __name__ == '__main__':

    root = Tk()
    App = GAMRY_EISPROC(root)  
    root.mainloop()
