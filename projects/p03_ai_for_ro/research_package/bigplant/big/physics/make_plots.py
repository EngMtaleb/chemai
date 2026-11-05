from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/industrial-ro-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent

def make_plots(r,grid):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                        'axes.spines.right':False,'svg.fonttype':'none'})
    navy='#12324b';teal='#0b929c';amber='#d28a19';red='#b53d4a'
    p=r['nominal']['vessel_profile']
    x=[v['axial_fraction']*7 for v in p]
    fig,axs=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for ax,key,title,c in zip(axs,['pressure_bar','bulk_g_L','flux_LMH'],['Pressure (bar)','Bulk salinity (g/L)','Local flux (LMH)'],[navy,teal,amber]):
        ax.plot(x,[v[key] for v in p],color=c,lw=2.4)
        ax.set(xlabel='Position along seven elements',ylabel=title,xlim=(0,7));ax.grid(alpha=.17)
    fig.suptitle('A vessel is a sequence: pressure falls, salt concentrates, flux changes',color=navy)
    fig.savefig(HERE/'vessel_profile.svg');fig.savefig(HERE/'vessel_profile.png',dpi=180);plt.close(fig)
    selected=[r['scenarios'][i] for i in [0,2,3,4,5,6,7]]
    labels=['Clean','35°C','50 g/L','A × 0.8','ΔP law × 3','B × 2.5','Combined']
    fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,key,yl in zip(axs,['feed_pressure_bar','SEC_kWh_m3','permeate_salinity_mg_L'],['Pressure (bar)','Modeled electricity (kWh/m³)','Product salt (mg/L)']):
        ax.bar(np.arange(len(selected)),[v[key] for v in selected],color=[teal if v['feasible'] else red for v in selected])
        ax.set_xticks(range(len(labels)),labels,rotation=40,ha='right');ax.set_ylabel(yl);ax.grid(axis='y',alpha=.17)
        if key=='permeate_salinity_mg_L':ax.axhline(500,color=red,ls='--',lw=1,label='Study gate');ax.legend(frameon=False,fontsize=9)
    fig.suptitle('Scenario stress tests at equal gross production — simulated, not measured',color=navy)
    fig.savefig(HERE/'stress_cases.svg');fig.savefig(HERE/'stress_cases.png',dpi=180);plt.close(fig)
    g=[p for p in grid if p['state']=='clean' and p['inputs']['recovery']==.4 and p['inputs']['target_flux_LMH']==r['config']['target_flux_LMH']]
    T=sorted(set(p['inputs']['temperature_C'] for p in g));C=sorted(set(p['inputs']['feed_salinity_g_L'] for p in g))
    vals=np.array([[next(p['feed_pressure_bar'] for p in g if p['inputs']['temperature_C']==t and p['inputs']['feed_salinity_g_L']==c) for c in C] for t in T])
    fig,ax=plt.subplots(figsize=(6.4,4.5),layout='constrained')
    im=ax.imshow(vals,origin='lower',cmap='Blues',aspect='auto');ax.set_xticks(range(len(C)),C);ax.set_yticks(range(len(T)),T)
    for i,t in enumerate(T):
      for j,c in enumerate(C):
        pp=next(p for p in g if p['inputs']['temperature_C']==t and p['inputs']['feed_salinity_g_L']==c)
        ax.text(j,i,f'{vals[i,j]:.1f}'+(' ✕' if not pp['feasible'] else ''),ha='center',va='center',color='white' if vals[i,j]>73 else navy)
    # BIGPLANT: title text was hard-coded 'at 600 m³/day'; now taken from the nominal result (label only).
    ax.set(xlabel='Feed salinity (g/L)',ylabel='Feed temperature (°C)',title=f"Pressure demand at {r['nominal']['plant_q_m3_day']:,.0f} m³/day\n× indicates a failed study design gate")
    fig.colorbar(im,ax=ax,label='Feed pressure (bar)')
    fig.savefig(HERE/'operating_envelope.svg');fig.savefig(HERE/'operating_envelope.png',dpi=180);plt.close(fig)
