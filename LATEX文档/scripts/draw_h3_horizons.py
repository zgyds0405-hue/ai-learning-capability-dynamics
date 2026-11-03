"""Three capability indices and horizon-specific choice among constant policies."""
from pathlib import Path
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,FixedFormatter,NullLocator
BASE=Path(__file__).resolve().parents[1];ROOT=BASE.parent
for p in [ROOT/'code',BASE/'reproduction/code',ROOT/'code/figure_qa',BASE/'reproduction/code/figure_qa']:sys.path.insert(0,str(p))
from audit_panel_alignment import require_matplotlib_panel_alignment
from tick_display import apply_trimmed_tick_formatters
DATA=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'results/h3_horizon'
if not DATA.exists():DATA=BASE/'reproduction/results/h3_horizon'
z=np.load(DATA/'fixed_policy_trajectories.npz');t=z['time'];u=z['u']
# The log-horizon panel uses strictly positive saved times only; T=0 is tied.
assert np.all(t[1:]>0)
plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':7,'axes.labelsize':8,
 'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':8,'axes.linewidth':.6,
 'lines.linewidth':1.1,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white',
 'mathtext.fontset':'stix','mathtext.default':'it'})
fig,axes=plt.subplots(2,2,figsize=(159/25.4,116/25.4))
fig.subplots_adjust(left=.10,right=.97,bottom=.10,top=.91,wspace=.30,hspace=.43)
# A neutral independent-learning baseline and a restrained blue/teal/violet family.
# Line styles also distinguish allocations where trajectories nearly coincide.
colors=['#767676','#88AACB','#3775BA','#42949E','#9A4D8E']
styles=['--',':','-','-.','-']
for ax,key,letter in zip(axes.flat,['h1','h2','h3','choice'],'abcd'):
    ax.spines[['right','top']].set_visible(False)
    ax.tick_params(width=.6,length=2.5,pad=3)
    ax.text(-.18,1.10,letter,transform=ax.transAxes,fontsize=8,fontweight='bold',ha='left',va='top')
    if key!='choice':
        for j,color,ls in zip([0,50,100,150,200],colors,styles):
            ax.plot(t,z[key][:,j],color=color,linestyle=ls,linewidth=1.6 if j==200 else 1.3,label=rf'$u={u[j]:g}$')
        ax.set(xlim=(0,20),ylim=(0,1),xticks=[0,4,8,12,16,20],yticks=[0,.25,.5,.75,1],xlabel=r'$t$')
        ax.set_ylabel({'h1':r'$h_1$','h2':r'$h_2$','h3':r'$h_3$'}[key])
    else:
        for k,col,label,ls in [('h1','#3775BA',r'$h_1$','--'),('h2','#42949E',r'$h_2$',':'),('h3','#9A4D8E',r'$h_3$','-')]:
            best=u[np.argmax(z[k],axis=1)]
            ax.plot(t[1:],best[1:],color=col,label=label,linestyle=ls,linewidth=1.2)
        ax.set(xscale='log',xlim=(.01,20),ylim=(-.04,1.07),yticks=[0,.25,.5,.75,1],xlabel=r'$T$',ylabel=r'$u$')
        ticks=[.01,.1,.5,1,4,20]
        ax.xaxis.set_major_locator(FixedLocator(ticks));ax.xaxis.set_major_formatter(FixedFormatter(['0.01','0.1','0.5','1','4','20']))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.legend(loc='center left',bbox_to_anchor=(.02,.51),frameon=False,handlelength=2.5,labelspacing=.7)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,ncol=5,loc='upper center',bbox_to_anchor=(.54,1.003),frameon=False,handlelength=2.3,columnspacing=1.8)
qa=BASE/'output/qa';qa.mkdir(parents=True,exist_ok=True)
apply_trimmed_tick_formatters(fig)
alignment=require_matplotlib_panel_alignment(fig,axes=list(axes.flat),panel_ids=list('abcd'),
    require_panel_labels=True,json_out=qa/'h3_horizons_alignment.json',overlay_svg=qa/'h3_horizons_alignment.svg')
fig.savefig(BASE/'figures/09_h3_horizons.pdf')
fig.savefig(BASE/'figures/09_h3_horizons.svg')
fig.savefig(BASE/'figures/09_h3_horizons.png',dpi=600)
fig.savefig(BASE/'figures/09_h3_horizons.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
manifest=dict(claim='The best constant proportion depends on both evaluation horizon and which capability is valued.',
 evidence={'a':'h1 time course exposes the early opportunity cost and late internalization benefit.',
           'b':'h2 time course shows accumulation or decay of collaborative skills.',
           'c':'h3 evaluates the two concurrent states with equal weights.',
           'd':'Each point selects one constant u for a separate run to that horizon; it is not a switching schedule.'},
 archetype='quantitative grid',backend='python-matplotlib',width_mm=159,height_mm=116,font_floor_pt=5.0,
 palette=colors,line_styles=styles,
 source_data=str(DATA),source_shape=[2001,201,3],display='Five prespecified proportions shown in a-c; all 201 evaluated in d.',
 exclusions='No source data removed. T=0 is omitted only in d because every policy is tied and log(0) is undefined.',
 uncertainty='Deterministic simulation, no sampling or confidence intervals.',alignment=alignment['verdict'],
 style_note='159 mm matches the course manuscript text width; it is not an official Nature column width.')
(qa/'h3_horizons_figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('Exported h3 horizon figure; panel alignment:',alignment['verdict'])
