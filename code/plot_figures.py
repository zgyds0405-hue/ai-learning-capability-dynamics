"""Nature-inspired quantitative figures; deterministic simulations only.

Each panel is traceable to the complete CSV files in results/data. There are
no observations, replicates, estimated confidence intervals or p values.
"""
from pathlib import Path
import sys
import json
import subprocess
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.text import Text
from tick_display import apply_trimmed_tick_formatters
from model import (PARAM,REP_U,BETA_SCENARIOS,H20_SCENARIOS,HORIZONS,GRID,
                   INTERACTION_BETA,JOINT_HORIZONS,independent)

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'/'figure_qa'))
if (ROOT/'.runtime').is_dir():
    sys.path.insert(0,str(ROOT/'.runtime'))
from audit_panel_alignment import require_matplotlib_panel_alignment
from audit_pdf_text import audit_pdf as audit_text
from audit_figure_collisions import audit_pdf as audit_collisions, write_overlay_pdf

IMAGES=ROOT/'images'
QA=ROOT/'results'/'figure_qa'
COLORS=['#868B90','#A7C5D9','#557FAD','#387E87','#9B657D']
STYLES=['-','--','-','-.',':']
CAP_COLORS=['#437DA8','#B47559']
P_SYMBOLS={'a':'a','beta':r'\beta','c':'c','eps':r'\varepsilon',
           'delta1':r'\delta_1','delta2':r'\delta_2'}
P_LABELS={key:f'${symbol}$' for key,symbol in P_SYMBOLS.items()}
RECORDS=[]

def setup():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7.5,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
        'legend.fontsize':8,'axes.titlesize':8,'axes.titleweight':'regular',
        'mathtext.fontset':'stix','mathtext.default':'it','text.usetex':False,
        'axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False,
        'xtick.major.width':.65,'ytick.major.width':.65,'xtick.major.size':2.5,
        'ytick.major.size':2.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'legend.frameon':False,'lines.linewidth':1.35,'savefig.facecolor':'white'})

def canvas(rows=1,cols=3,height=78,*,right=.975,top=.80,bottom=.20):
    fig,axes=plt.subplots(rows,cols,figsize=(180/25.4,height/25.4),squeeze=False)
    fig.subplots_adjust(left=.09,right=right,bottom=bottom,top=top,wspace=.42,hspace=.56)
    for j,ax in enumerate(axes.flat):
        ax.text(-.19,1.07,chr(97+j),transform=ax.transAxes,fontsize=9,fontweight='bold',
                ha='left',va='bottom')
        ax.tick_params(pad=2)
    return fig,axes

def time_axis(ax):
    ax.set(xlim=(0,20),xticks=[0,5,10,15,20],xlabel=r'$t$')

def ability_axis(ax,which):
    ax.set(ylim=(0,1),yticks=[0,.5,1],ylabel=rf'$h_{{{which}}}$')

def legend(fig,handles,labels,*,ncol=5,y=.97):
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,y),ncol=ncol,
               handlelength=2.3,columnspacing=1.4,handletextpad=.5)

def save(fig,name,axes):
    IMAGES.mkdir(exist_ok=True)
    QA.mkdir(parents=True,exist_ok=True)
    apply_trimmed_tick_formatters(fig)
    labels=[]
    for artist in fig.findobj(match=Text):
        value=artist.get_text()
        if not value:
            continue
        if value.count('$')%2:
            raise RuntimeError(f'Unbalanced math delimiters: {name}: {value}')
        plain=re.sub(r'\$[^$]*\$','',value)
        if len(value)>1 and re.search(r'\bh[12]\b|[βδε]|\b(?:u|T|t)\b',plain):
            raise RuntimeError(f'Mathematical symbol outside dollar delimiters: {name}: {value}')
        if len(value)>1 and plain.strip() and not re.fullmatch(r'[\d\s.,+\-−eE]+',plain):
            raise RuntimeError(f'Explanatory text in symbol-only figure: {name}: {value}')
        labels.append(value)
    (QA/f'{name}.labels.json').write_text(json.dumps(dict(
        math_fontset=plt.rcParams['mathtext.fontset'],math_default=plt.rcParams['mathtext.default'],
        all_symbol_labels_in_math_mode=True,symbol_only_labels=True,labels=labels),ensure_ascii=False,indent=2),encoding='utf-8')
    # Reject accidental data clipping independently of text-collision auditing.
    # Reference lines using mixed coordinate transforms are excluded here.
    for ax in axes.flat:
        xlo,xhi=sorted(ax.get_xlim())
        ylo,yhi=sorted(ax.get_ylim())
        for line in ax.lines:
            if line.get_transform()!=ax.transData:
                continue
            xx=np.asarray(line.get_xdata(),dtype=float)
            yy=np.asarray(line.get_ydata(),dtype=float)
            if np.any(xx<xlo-1e-10) or np.any(xx>xhi+1e-10) or np.any(yy<ylo-1e-10) or np.any(yy>yhi+1e-10):
                raise RuntimeError(f'Curve outside display range: {name}')
    # The final drawn data rectangles, including panel-label anchors, are audited.
    alignment=require_matplotlib_panel_alignment(fig,axes=list(axes.flat),
        json_out=QA/f'{name}.alignment.json',overlay_svg=QA/f'{name}.alignment.svg',
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=True,strict=True)
    # Fixed physical canvas: no tight bounding-box rescaling.
    fig.savefig(IMAGES/f'{name}.pdf',dpi=600)
    fig.savefig(IMAGES/f'{name}.svg',dpi=600)
    fig.savefig(IMAGES/f'{name}.png',dpi=600)
    pdf=IMAGES/f'{name}.pdf'
    text=audit_text(pdf.read_bytes(),minimum_pt=5.)
    collision=audit_collisions(pdf)
    (QA/f'{name}.text.json').write_text(json.dumps(text,indent=2),encoding='utf-8')
    (QA/f'{name}.collision.json').write_text(json.dumps(collision,indent=2),encoding='utf-8')
    write_overlay_pdf(pdf,QA/f'{name}.collision.pdf',collision['findings'])
    RECORDS.append(dict(figure=name,alignment=alignment.get('verdict'),
                        minimum_text_pt=text['minimum_found_pt'],collision=collision['verdict'],
                        findings=collision['summary'],data_clipping='PASS',formula_labels='PASS',
                        symbol_only_labels='PASS'))
    plt.close(fig)
    if not text['auditable'] or text['below_minimum_count']:
        raise RuntimeError(f'PDF glyph floor failed: {name}')
    if not collision['auditable'] or collision['summary']['fail']:
        raise RuntimeError(f'PDF collision gate failed: {name}')
    print(f'Figure {name}: {collision["verdict"]}',flush=True)

def line_triptych(data,name,values,label,*,difference=False):
    t,x=data
    fig,aa=canvas()
    axes=aa[0]
    handles=[]
    for j,v in enumerate(values):
        color=COLORS[j]
        ls=STYLES[j]
        hero=3 if difference else 2
        line,=axes[0].plot(t,x[:,j,0],color=color,ls=ls,lw=1.7 if j==hero else 1.35)
        axes[1].plot(t,x[:,j,1],color=color,ls=ls)
        if difference:
            axes[2].plot(t,x[:,j,0]-x[:,0,0],color=color,ls=ls)
        else:
            axes[2].plot(x[:,j,0],x[:,j,1],color=color,ls=ls)
            axes[2].plot(*x[-1,j],marker='o',ms=3.1,color=color)
        handles.append(line)
    for j in range(2):
        time_axis(axes[j])
        ability_axis(axes[j],j+1)
    if difference:
        time_axis(axes[2])
        maxdiff=float(np.max(x[...,0]-x[:,0:1,0]))
        axes[2].set(ylabel=r'$h_1(t)-h_1(t;\beta=0)$',ylim=(-.005,maxdiff*1.08),yticks=[0,.1,.2])
    else:
        axes[2].set(xlabel=r'$h_1$',ylabel=r'$h_2$',
                    xlim=(0,1),ylim=(0,1),xticks=[0,.5,1],yticks=[0,.5,1])
    legend(fig,handles,[rf'${label}={v:g}$' for v in values])
    save(fig,name,aa)

def mechanisms(data):
    t,x=data
    fig,aa=canvas(2,3,150,top=.82,bottom=.10)
    perturb_colors=['#638EAF','#888D92','#B77960']
    for j,(key,ax) in enumerate(zip(PARAM,aa.flat)):
        for idx,col,marker in zip([1+2*j,0,2+2*j],perturb_colors,['o',None,'^']):
            ax.plot(t,x[:,idx,0],color=col,ls='-',marker=marker,ms=2.1,markevery=(200,400))
            ax.plot(t,x[:,idx,1],color=col,ls='--',lw=1.05,marker=marker,ms=2.1,markevery=(200,400))
        ax.set_title(rf'${P_SYMBOLS[key]}={PARAM[key]:g}$',pad=8)
        time_axis(ax)
        ax.set(ylim=(0,1),yticks=[0,.5,1],ylabel=r'$h_i$')
    handles=[Line2D([],[],color=c,marker=m,ms=2.5) for c,m in zip(perturb_colors,['o',None,'^'])]
    legend(fig,handles,[r'$p/p_0=0.8$',r'$p/p_0=1$',r'$p/p_0=1.2$'],ncol=3,y=.985)
    legend(fig,[Line2D([],[],color='#4A4E52'),Line2D([],[],color='#4A4E52',ls='--')],
           [r'$h_1$',r'$h_2$'],ncol=2,y=.937)
    save(fig,'03_mechanism_trajectories',aa)

def coupling(data):
    t,x=data
    fig,aa=canvas(2,2,132,top=.83,bottom=.11)
    cc=['#3F7B8C','#9A759C','#B58C5D','#9A9E9F']
    ss=['-','--','-.',':']
    handles=[]
    for j in range(4):
        l,=aa[0,0].plot(t,x[:,j,0],color=cc[j],ls=ss[j],lw=1.7 if j==0 else 1.3)
        aa[0,1].plot(t,x[:,j,1],color=cc[j],ls=ss[j])
        aa[1,0].plot(x[:,j,0],x[:,j,1],color=cc[j],ls=ss[j])
        aa[1,0].plot(*x[-1,j],marker='o',ms=3,color=cc[j])
        handles.append(l)
    for j in range(2):
        time_axis(aa[0,j]); ability_axis(aa[0,j],j+1)
    aa[1,0].set(xlabel=r'$h_1$',ylabel=r'$h_2$',
                  xlim=(0,1),ylim=(0,1),xticks=[0,.5,1],yticks=[0,.5,1])
    diff=x[:,0]-x[:,3]
    for j in range(2):
        aa[1,1].plot(t,diff[:,j],color=CAP_COLORS[j],ls=['-','--'][j],label=rf'$h_{{{j+1}}}$')
    time_axis(aa[1,1])
    aa[1,1].set(ylabel=r'$\Delta h_i(t)$',
               ylim=(0,float(diff.max())*1.08),yticks=[0,.1,.2,.3])
    aa[1,1].legend(loc='upper left',fontsize=8)
    legend(fig,handles,[r'$\mathcal{C}_0$',r'$\mathcal{C}_2$',r'$\mathcal{C}_1$',r'$\mathcal{C}_{12}$'],ncol=4,y=.985)
    save(fig,'04_dynamic_coupling',aa)

def horizons(data):
    endpoints,paths=data
    fig,aa=canvas(2,2,134,top=.82,bottom=.11)
    colors=['#AEB4BA','#8199AB','#A4B9CF','#477EB0','#588C82','#B48B64','#906D8C']
    styles=[':', '--','-.','-','--','-.','-']
    handles=[]
    for j,T in enumerate(HORIZONS):
        t,path=paths[j]
        l,=aa[0,0].plot(t/T,path[:,0],color=colors[j],ls=styles[j])
        aa[0,1].plot(t/T,path[:,1],color=colors[j],ls=styles[j])
        aa[1,0].plot(GRID,endpoints[j,:,0],color=colors[j],ls=styles[j])
        aa[1,1].plot(GRID,endpoints[j,:,0]-independent(T)[0],color=colors[j],ls=styles[j])
        handles.append(l)
    for j in range(2):
        aa[0,j].set(xlim=(0,1),xticks=[0,.5,1],xlabel=r'$t/T$')
        ability_axis(aa[0,j],j+1)
        aa[1,j].set(xlim=(0,1),xticks=[0,.25,.5,.75,1],xlabel=r'$u$')
    aa[1,0].set(ylabel=r'$h_1(T)$',ylim=(0,1),yticks=[0,.5,1])
    aa[1,1].axhline(0,color='#B8BEC2',lw=.7,zorder=0)
    aa[1,1].set(ylabel=r'$h_1(T;u)-h_1(T;u=0)$',ylim=(-.17,.06),yticks=[-.15,-.10,-.05,0,.05])
    legend(fig,handles,[rf'$T={v:g}$' for v in HORIZONS],ncol=4,y=.985)
    save(fig,'05_observation_horizons',aa)

def initial(data):
    t,x=data
    fig,aa=canvas()
    handles=[]
    for j,v in enumerate(H20_SCENARIOS):
        l,=aa[0,0].plot(t,x[:,j,0],color=COLORS[j],ls=STYLES[j])
        aa[0,1].plot(t,x[:,j,1],color=COLORS[j],ls=STYLES[j])
        handles.append(l)
    for j in range(2):
        time_axis(aa[0,j]); ability_axis(aa[0,j],j+1)
        aa[0,2].plot(t,np.ptp(x[...,j],axis=1),color=CAP_COLORS[j],ls=['-','--'][j],label=rf'$h_{{{j+1}}}$')
    time_axis(aa[0,2])
    aa[0,2].set(ylabel=r'$R_i(t)$',ylim=(0,1),yticks=[0,.5,1])
    aa[0,2].legend(loc='upper right')
    legend(fig,handles,[rf'$h_2(0)={v:g}$' for v in H20_SCENARIOS],ncol=4)
    save(fig,'06_initial_collaboration',aa)

def sensitivity(summary):
    fig,aa=canvas(1,2,90,top=.78,bottom=.20)
    for j,ax in enumerate(aa[0]):
        rows=sorted(summary['parameter_effects'],key=lambda r:r[f'span_h{j+1}'],reverse=True)
        for k,r in enumerate(rows):
            low,high=r[f'dh{j+1}_minus'],r[f'dh{j+1}_plus']
            ax.plot([low,high],[k,k],color='#B8BFC5',lw=1)
            ax.plot(low,k,marker='o',color='#648DB0',ms=4)
            ax.plot(high,k,marker='s',color='#B87960',ms=3.7)
        ax.axvline(0,color='#C1C6CA',lw=.7,zorder=0)
        ax.set(yticks=np.arange(6),yticklabels=[P_LABELS[r['parameter']] for r in rows],
                xlabel=rf'$\Delta h_{{{j+1}}}(20)$',ylim=(5.5,-.5))
        ax.tick_params(axis='y',labelsize=8)
        limits=[.013,.034][j]
        ax.set_xlim(-limits,limits)
        ax.set_xticks([-.01,0,.01] if j==0 else [-.03,0,.03])
    legend(fig,[Line2D([],[],color='#648DB0',marker='o',ls=''),
                Line2D([],[],color='#B87960',marker='s',ls='')],
           [r'$p/p_0=0.8$',r'$p/p_0=1.2$'],ncol=2)
    save(fig,'07_local_sensitivity',aa)

def joint(data):
    if len(data)!=len(JOINT_HORIZONS):
        raise ValueError('Joint surfaces must cover all four configured horizons')
    fig,aa=plt.subplots(2,4,figsize=(159/25.4,119/25.4),squeeze=False)
    fig.subplots_adjust(left=.075,right=.905,bottom=.12,top=.85,wspace=.57,hspace=.72)
    for j,ax in enumerate(aa.flat):
        ax.text(-.27,1.17,chr(97+j),transform=ax.transAxes,fontsize=9,fontweight='bold',
                ha='left',va='bottom')
        ax.tick_params(pad=2,labelsize=6.5)
    # Explicit user request (2026-11-02): jet, matching the supplied example.
    # Four time slices use the same saved mesh and fixed [0,1] colour scale.
    contour_effects=[path_effects.Stroke(linewidth=1.8,foreground='#28313D'),path_effects.Normal()]
    ug,bg=np.meshgrid(GRID,INTERACTION_BETA)
    for j,T in enumerate(JOINT_HORIZONS):
        for i in range(2):
            ax=aa[i,j]
            image=ax.pcolormesh(ug,bg,data[j][...,i],cmap='jet',vmin=0,vmax=1,
                                shading='nearest',rasterized=True)
            ax.set(xlabel=r'$u$',ylabel=r'$\beta$',
                   xlim=(0,1),ylim=(0,.8),xticks=[0,.5,1],yticks=[0,.4,.8])
            ax.set_title(rf'$h_{{{i+1}}}(T),\ T={T}$',pad=6)
            if i==0:
                d=data[j][...,0]-independent(T)[0]
                # Exclude the trivial u=0 equality from the interior contour.
                contour=ax.contour(ug[:,1:],bg[:,1:],d[:,1:],levels=[0],colors=['white'],
                                   linewidths=1.,linestyles='--')
                contour.set_path_effects(contour_effects)
    cax=fig.add_axes([.94,.28,.014,.40])
    cbar=fig.colorbar(image,cax=cax,ticks=[0,.5,1])
    cbar.ax.tick_params(labelsize=6.5,pad=2)
    cbar.ax.set_title(r'$h_i(T)$',fontsize=8,pad=7)
    legend(fig,[Line2D([],[],color='white',ls='--',path_effects=contour_effects)],
           [r'$h_1(T;u,\beta)=h_1(T;0)$'],ncol=1,y=.985)
    save(fig,'08_joint_parameter_surface',aa)

def draw_all(out,summary):
    setup()
    QA.mkdir(parents=True,exist_ok=True)
    source_check=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'code'/'figure_qa'/'validate_figure.py'),
        str(Path(__file__).resolve()),'--json'],capture_output=True,text=True,encoding='utf-8')
    (QA/'source_preflight.json').write_text(source_check.stdout,encoding='utf-8')
    if source_check.returncode:
        audit=json.loads(source_check.stdout)
        failures=[v for v in audit['findings'] if v['level']=='FAIL']
        # The user's supplied cmap='jet' overrides the style-only rainbow rule.
        # Preserve the raw finding; no other preflight failure is exempted.
        if not failures or any(v['check_id']!='COLOR-MAP' for v in failures):
            raise RuntimeError('Figure source preflight failed: '+source_check.stdout)
        (QA/'jet_user_style_override.json').write_text(json.dumps(dict(
            authority='Explicit user request and supplied matplotlib cmap=jet example, 2026-11-02',
            scope='Figure 10 colour mapping only',exempted_check='COLOR-MAP',raw_findings=failures),
            ensure_ascii=False,indent=2),encoding='utf-8')
    RECORDS.clear()
    line_triptych(out['allocation'],'01_time_allocation',REP_U,'u')
    line_triptych(out['beta'],'02_internalization_strength',BETA_SCENARIOS,r'\beta',difference=True)
    mechanisms(out['mechanisms'])
    coupling(out['coupling'])
    horizons(out['horizons'])
    initial(out['initial'])
    sensitivity(summary)
    joint(out['joint'])
    (QA/'summary.json').write_text(json.dumps(RECORDS,indent=2),encoding='utf-8')
