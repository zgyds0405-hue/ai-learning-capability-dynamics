"""Six preview panels only; never modifies the LaTeX manuscript or old results."""
from pathlib import Path
from hashlib import sha256
import csv,json,sys,shutil
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
for p in [BASE/'reproduction',ROOT/'code',ROOT/'code/figure_qa',ROOT/'.runtime']:sys.path.insert(0,str(p))
import numpy as np
from scipy.integrate import solve_ivp
import model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from audit_panel_alignment import require_matplotlib_panel_alignment

WEIGHTS=np.array([0.,.25,.5,.75,1.])
COLORS=['#9574AD','#ED944F','#4BA45B','#438CB5','#DE6966']
CASES=[('a','beta',0.,.1),('b','beta',.48,.1),('c','beta',.8,.1),
       ('d','epsilon',.48,0.),('e','epsilon',.48,.1),('f','epsilon',.48,1.)]
OUT=BASE/'figures';DATA=BASE/'data';QA=BASE/'qa'
for p in [OUT,DATA,QA,BASE/'panels']:p.mkdir(parents=True,exist_ok=True)

def digest(p):return sha256(p.read_bytes()).hexdigest()
def json_write(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def document_guard():
    paths=[p for p in (ROOT/'LATEX文档').rglob('*') if p.is_file()]
    paths += [ROOT/'LATEX文档.zip',ROOT/'code/model.py',*ROOT.glob('*.md')]
    return {str(p.relative_to(ROOT)):digest(p) for p in paths if p.exists()}

def compute():
    cache={};records=[]
    for letter,family,beta,epsilon in CASES:
        key=(beta,epsilon)
        if key in cache:continue
        t,x=model.simulate(u=.5,p={'beta':beta,'eps':epsilon})
        tf,xf=model.simulate(u=.5,p={'beta':beta,'eps':epsilon},max_step=.005)
        error=float(np.abs(x-xf[::2]).max())
        assert np.allclose(t,tf[::2],rtol=0,atol=1e-12) and error<1e-7
        # An independently written RHS and different solver verify the trajectory.
        def rhs(t,y):
            h1,h2=y;p=model.PARAM
            return [(1-h1)*(p['a']*.5+beta*.5*h2)-p['delta1']*h1,
                    p['c']*.5*(epsilon+(1-epsilon)*h1)*(1-h2)-p['delta2']*h2]
        sol=solve_ivp(rhs,(0,20),[model.H10,model.H20],method='DOP853',t_eval=t,rtol=1e-12,atol=1e-14)
        independent_error=float(np.abs(x-sol.y.T).max())
        assert sol.success and independent_error<1e-7
        records.append(dict(beta=beta,epsilon=epsilon,step_halving_max_error=error,DOP853_max_error=independent_error))
        cache[key]=(t,x)
    t=next(iter(cache.values()))[0]
    states=np.stack([cache[(b,e)][1] for _,_,b,e in CASES],axis=1)
    h3=states[:,:,0,None]*WEIGHTS+states[:,:,1,None]*(1-WEIGHTS)
    assert np.isfinite(h3).all() and states.min()>=0 and states.max()<=1
    assert np.array_equal(h3[:,:,0],states[:,:,1]) and np.array_equal(h3[:,:,-1],states[:,:,0])
    assert np.array_equal(h3[:,1,:],h3[:,4,:])
    # Two limiting cases are available in closed form.
    p=model.PARAM;r=p['a']*.5+p['delta1'];limit=p['a']*.5/r
    beta0_error=float(np.abs(states[:,0,0]-(limit+(model.H10-limit)*np.exp(-r*t))).max())
    r=p['c']*.5+p['delta2'];limit=p['c']*.5/r
    epsilon1_error=float(np.abs(states[:,5,1]-(limit+(model.H20-limit)*np.exp(-r*t))).max())
    assert max(beta0_error,epsilon1_error)<1e-9
    old=ROOT/'results/h3_horizon/fixed_policy_trajectories.npz'
    saved_error=None
    if old.exists():
        z=np.load(old);index=int(np.flatnonzero(np.isclose(z['u'],.5))[0])
        saved_error=float(max(np.abs(states[:,1,0]-z['h1'][:,index]).max(),np.abs(states[:,1,1]-z['h2'][:,index]).max()))
        assert saved_error<1e-12
    np.savez_compressed(DATA/'trajectories.npz',T=t,w=WEIGHTS,h1=states[:,:,0],h2=states[:,:,1],h3=h3,
                        beta=np.array([c[2] for c in CASES]),epsilon=np.array([c[3] for c in CASES]))
    fields=['panel','factor','beta','epsilon','u','T','w','h1','h2','h3']
    with (DATA/'all_trajectories.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(fields)
        for j,(letter,family,beta,epsilon) in enumerate(CASES):
            for i,T in enumerate(t):
                for k,w in enumerate(WEIGHTS):writer.writerow([letter,family,beta,epsilon,.5,float(T),float(w),float(states[i,j,0]),float(states[i,j,1]),float(h3[i,j,k])])
    endpoints=[]
    for j,(letter,family,beta,epsilon) in enumerate(CASES):
        endpoints.append(dict(panel=letter,factor=family,beta=beta,epsilon=epsilon,h1_T20=float(states[-1,j,0]),h2_T20=float(states[-1,j,1]),
                              **{f'h3_w{w:g}':float(h3[-1,j,k]) for k,w in enumerate(WEIGHTS)}))
    with (DATA/'endpoints_T20.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(endpoints[0]));writer.writeheader();writer.writerows(endpoints)
    json_write(QA/'numerical_checks.json',dict(all_checks_passed=True,unique_dynamic_conditions=5,displayed_panels=6,
       weight_curves_per_panel=5,total_displayed_curves=30,saved_time_points=len(t),csv_rows=6*5*len(t),integration_step=.01,
       baseline_parameters=p,fixed_u=.5,initial=[model.H10,model.H20],checks=records,beta0_analytic_error=beta0_error,
       epsilon1_analytic_error=epsilon1_error,baseline_saved_data_error=saved_error,state_bounds=[float(states.min()),float(states.max())],
       clipping=False,duplicate_baseline_panels=['b','e'],weight_role='Evaluation only: h3=w*h1+(1-w)*h2; w is not AI time u.',endpoints=endpoints))
    return t,states,h3

def style_axes(ax):
    ax.view_init(elev=24,azim=-57)
    ax.set_proj_type('ortho')
    ax.set_box_aspect((1.40,.85,1.02))
    ax.set(xlim=(0,20),ylim=(-.035,1.035),zlim=(0,1),xticks=[0,5,10,15,20],yticks=WEIGHTS,zticks=[0,.25,.5,.75,1])
    ax.set_yticklabels(['0','0.25','0.5','0.75','1'])
    ax.set_zticklabels(['0','0.25','0.5','0.75','1'])
    ax.set_xlabel('T',labelpad=4,fontsize=10)
    ax.set_ylabel('w',labelpad=5,fontsize=10)
    ax.set_zlabel('h₃',labelpad=4,fontsize=10)
    ax.tick_params(axis='both',labelsize=8,pad=0)
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
        axis.pane.set_facecolor((.97,.98,.99,1));axis.pane.set_edgecolor((.83,.85,.88,1))
        axis._axinfo['grid'].update(color=(.78,.81,.85,.5),linewidth=.45)
        axis.line.set_color('#75808A');axis.line.set_linewidth(.6)

def draw_panel(ax,t,values,j):
    style_axes(ax)
    for k,(w,color) in enumerate(zip(WEIGHTS,COLORS)):
        z=values[:,j,k]
        verts=[(float(t[0]),0.),*list(zip(t,z)),(float(t[-1]),0.)]
        fill=PolyCollection([verts],facecolors=[color],edgecolors='none',alpha=.13)
        ax.add_collection3d(fill,zs=float(w),zdir='y')
        # Every saved time contributes to the line and curtain. Markers are sparse
        # time guides only, not a subsample of the underlying simulated trajectory.
        ax.plot(t,np.full_like(t,w),z,color=color,lw=1.25,marker='o',markersize=2.1,markevery=100,
                markeredgewidth=0,label=f'w = {w:g}')
    letter,family,beta,epsilon=CASES[j]
    title=f'β = {beta:.2f}' if family=='beta' else f'ε = {epsilon:.2f}'
    ax.set_title(title,fontsize=11,pad=3,color='#26333D')
    ax.text2D(.01,1.025,letter,transform=ax.transAxes,fontsize=10,fontweight='bold',va='top')

def static_figures(t,states,h3):
    plt.rcParams.update({'font.family':['Microsoft YaHei','Arial','DejaVu Sans'],'font.size':9,
        'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','axes.unicode_minus':False})
    fig=plt.figure(figsize=(310/25.4,220/25.4))
    grid=fig.add_gridspec(2,3,left=.015,right=.985,bottom=.065,top=.85,wspace=.065,hspace=.28)
    axes=[]
    for j in range(6):
        ax=fig.add_subplot(grid[j//3,j%3],projection='3d');axes.append(ax);draw_panel(ax,t,h3,j)
    fig.text(.5,.985,'耦合能力随时间、目标权重及机制参数的演化',ha='center',va='top',fontsize=13)
    fig.text(.5,.947,'h₃(T) = w h₁(T) + (1 − w) h₂(T)    ·    w 为独立能力权重',ha='center',fontsize=10)
    handles=[Line2D([],[],color=c,lw=1.5,marker='o',markersize=3,label=f'w = {w:g}') for w,c in zip(WEIGHTS,COLORS)]
    fig.legend(handles=handles,ncol=5,loc='upper center',bbox_to_anchor=(.5,.929),frameon=False,fontsize=9,handlelength=2.6,columnspacing=2.3)
    fig.text(.035,.887,'内化效率对照：固定 ε = 0.10',fontsize=10)
    fig.text(.035,.449,'基础协作支持对照：固定 β = 0.48',fontsize=10)
    fig.text(.5,.053,'固定 u = 0.50；初值 h₁(0) = 0.20，h₂(0) = 0.10；T 为未标定的有效学习时间。',ha='center',fontsize=8.5)
    fig.text(.5,.033,'其余参数：a = 0.25，c = 0.55，δ₁ = 0.02，δ₂ = 0.03；β 的单位为模型时间⁻¹，ε 无量纲。',ha='center',fontsize=8)
    fig.text(.5,.013,'半透明填充仅辅助区分权重切片；b 与 e 为相同基准。曲线为数值轨迹，圆点为每 1 个时间单位的标记。',ha='center',fontsize=8)
    alignment=require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=list('abcdef'),require_panel_labels=True,
        json_out=QA/'panel_alignment.json',overlay_svg=QA/'panel_alignment.svg')
    fig.savefig(OUT/'h3_weights_beta_epsilon_3d.png',dpi=300)
    fig.savefig(OUT/'h3_weights_beta_epsilon_3d.pdf')
    fig.savefig(OUT/'h3_weights_beta_epsilon_3d.svg')
    fig.savefig(OUT/'h3_weights_beta_epsilon_3d.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    for j,(letter,family,beta,epsilon) in enumerate(CASES):
        f=plt.figure(figsize=(105/25.4,92/25.4));ax=f.add_subplot(111,projection='3d')
        f.subplots_adjust(left=.02,right=.95,bottom=.08,top=.88)
        draw_panel(ax,t,h3,j)
        f.text(.5,.035,f'u = 0.50; β = {beta:g}; ε = {epsilon:g}',ha='center',fontsize=8)
        f.savefig(BASE/f'panels/{letter}_{family}.png',dpi=300)
        f.savefig(BASE/f'panels/{letter}_{family}.pdf')
        plt.close(f)
    return alignment['verdict']

def interactive(t,states,h3):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    titles=[f'{a}　β = {b:.2f}（ε=0.10）' if family=='beta' else f'{a}　ε = {e:.2f}（β=0.48）' for a,family,b,e in CASES]
    f=make_subplots(rows=2,cols=3,specs=[[{'type':'scene'}]*3]*2,subplot_titles=titles,horizontal_spacing=.035,vertical_spacing=.10)
    n=len(t)
    for j,(letter,family,beta,epsilon) in enumerate(CASES):
        for k,(w,color) in enumerate(zip(WEIGHTS,COLORS)):
            x=np.r_[t,t];y=np.full(2*n,w);z=np.r_[h3[:,j,k],np.zeros(n)]
            i=np.r_[np.arange(n-1),np.arange(n-1)];jj=np.r_[np.arange(1,n),np.arange(1,n)+n];kk=np.r_[np.arange(n-1)+n,np.arange(n-1)+n]
            f.add_trace(go.Mesh3d(x=x,y=y,z=z,i=i,j=jj,k=kk,color=color,opacity=.12,flatshading=True,
                                  showscale=False,showlegend=False,legendgroup=f'w{k}',hoverinfo='skip'),row=j//3+1,col=j%3+1)
            custom=np.c_[states[:,j,0],states[:,j,1]]
            f.add_trace(go.Scatter3d(x=t,y=np.full(n,w),z=h3[:,j,k],mode='lines',line=dict(color=color,width=4),
                    name=f'w={w:g}',legendgroup=f'w{k}',showlegend=j==0,customdata=custom,
                    hovertemplate=f'面板 {letter} · β={beta:g}, ε={epsilon:g}<br>w={w:g}<br>T=%{{x:.2f}}<br>h₃=%{{z:.6f}}<br>h₁=%{{customdata[0]:.6f}}<br>h₂=%{{customdata[1]:.6f}}<extra></extra>'),row=j//3+1,col=j%3+1)
    scene=dict(xaxis=dict(title='T',range=[0,20],tickvals=[0,5,10,15,20]),
               yaxis=dict(title='w',range=[-.035,1.035],tickvals=WEIGHTS),zaxis=dict(title='h₃',range=[0,1],tickvals=[0,.25,.5,.75,1]),
               aspectmode='manual',aspectratio=dict(x=1.4,y=.85,z=1.02),camera=dict(eye=dict(x=1.6,y=-2.15,z=1.3),projection=dict(type='orthographic')))
    f.update_scenes(**scene)
    f.update_layout(title=dict(text='h₃ = w h₁ + (1−w) h₂ · u=0.50 · 五种权重的六联图',x=.5),height=970,
                    margin=dict(l=20,r=20,t=100,b=25),font=dict(family='Arial,Microsoft YaHei,sans-serif',size=12),
                    template='plotly_white',legend=dict(orientation='h',x=.5,xanchor='center',y=1.065,groupclick='togglegroup'))
    # All data are embedded; the file opens offline. One rotated camera is shared
    # with the other scenes to make comparisons retain the same projection.
    post=r"""var gd=document.getElementById('{plot_id}');var busy=false;
gd.on('plotly_relayout',function(e){if(busy)return;var keys=Object.keys(e);var key=keys.find(k=>/^scene[2-6]?\.camera$/.test(k));if(!key)return;
var cam=e[key],src=key.split('.')[0],patch={};['scene','scene2','scene3','scene4','scene5','scene6'].forEach(s=>{if(s!==src)patch[s+'.camera']=cam;});
busy=true;Plotly.relayout(gd,patch).then(()=>{busy=false;});});"""
    f.write_html(BASE/'h3_3d_interactive.html',include_plotlyjs=True,full_html=True,post_script=post,config=dict(responsive=True,displaylogo=False))

def main():
    guard=document_guard();json_write(QA/'protected_document_hashes_before.json',guard)
    t,states,h3=compute();alignment=static_figures(t,states,h3);interactive(t,states,h3)
    changed=[name for name,value in guard.items() if digest(ROOT/name)!=value]
    assert not changed,changed
    json_write(QA/'document_unchanged.json',dict(all_unchanged=True,checked_files=len(guard),manuscript_and_zip_untouched=True,changed_files=changed))
    json_write(QA/'figure_contract.json',dict(question='How do weighting and two mechanism parameters jointly change h3 time courses at the same fixed AI proportion?',
        claim='The same weight can emphasize different time-dependent capability mixtures; beta changes internalization while epsilon changes the knowledge condition.',
        archetype='quantitative grid; six equal 3D curtain panels',panel_roles={'a':'zero internalization','b':'baseline for beta row','c':'higher internalization','d':'knowledge-only condition','e':'matched baseline for epsilon row','f':'full external support; h2 formation no longer depends on h1'},
        backend='Python: Matplotlib static and Plotly interactive',width_mm=310,height_mm=220,
        palette=dict(zip(map(str,WEIGHTS),COLORS)),alignment=alignment,displayed_rows=60030,
        exclusions='None. Every saved point is used in lines and fill boundaries. Static dots mark every 1 time unit; they are not replicates.',
        filled_regions='Visual weight slices to z=0; no additional measured response or integral is encoded.',
        preview_only=True,manuscript_modified=False,reference_image='User-supplied palette and curtain layout only; no reference-image data or discontinuities reused.'))
    rep=BASE/'reproduction';rep.mkdir(exist_ok=True)
    source=Path(model.__file__).resolve();target=(rep/'model.py').resolve()
    if source!=target:shutil.copy2(source,target)
    helper=ROOT/'code/figure_qa/audit_panel_alignment.py'
    if helper.exists():shutil.copy2(helper,rep/'audit_panel_alignment.py')
    print(json.dumps(dict(output=str(BASE),alignment=alignment,manuscript_unchanged=True,unique_conditions=5,panels=6,curves=30),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
