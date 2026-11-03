"""Extend only the joint scan; verify against old CSVs and an independent solver."""
from pathlib import Path
import csv, json, sys
import numpy as np
from scipy.integrate import solve_ivp
from threadpoolctl import threadpool_limits
from model import PARAM, H10, H20, DT, FINE_DT, TOL, GRID, INTERACTION_BETA, JOINT_HORIZONS, simulate, independent

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/data'

def run():
    ug,bg=np.meshgrid(GRID,INTERACTION_BETA)
    print('Joint scan: RK4 time courses, 2091 parameter pairs',flush=True)
    t,coarse=simulate(u=ug,p={'beta':bg},keep=True)
    tf,fine=simulate(u=ug,p={'beta':bg},max_step=FINE_DT,keep=True)
    step_error=float(np.abs(coarse-fine[::2]).max())
    assert step_error<TOL
    del fine
    a,c,eps,d1,d2=[PARAM[k] for k in ['a','c','eps','delta1','delta2']]
    def independent_rhs(t,y):
        h1,h2=y.reshape(-1,2).T
        return np.column_stack(((1-h1)*(a*(1-ug.ravel())+bg.ravel()*ug.ravel()*h2)-d1*h1,
            c*ug.ravel()*(eps+(1-eps)*h1)*(1-h2)-d2*h2)).ravel()
    y0=np.tile([H10,H20],(ug.size,1)).ravel()
    print('Joint scan: independent DOP853 comparison at all four horizons',flush=True)
    with threadpool_limits(limits=1):
        ref=solve_ivp(independent_rhs,(0,20),y0,method='DOP853',t_eval=JOINT_HORIZONS,rtol=1e-11,atol=1e-13)
    assert ref.success
    selected=np.stack([coarse[round(T/DT)] for T in JOINT_HORIZONS])
    independent_error=float(np.abs(selected-ref.y.T.reshape(selected.shape)).max())
    assert independent_error<TOL
    metrics=[];old_errors={}
    for k,T in enumerate(JOINT_HORIZONS):
        x=selected[k];base=float(independent(T)[0]);path=DATA/f'08_joint_T{T}.csv'
        if path.exists():
            with path.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
            old=np.array([[float(v['h1']),float(v['h2'])] for v in rows]).reshape(x.shape)
            old_errors[str(T)]=float(np.abs(x-old).max())
            assert old_errors[str(T)]<1e-12
        else:
            with path.open('w',encoding='utf-8-sig',newline='') as f:
                w=csv.writer(f);w.writerow(['T','u','beta','h1','h2','D_h1'])
                w.writerows([T,u,b,*v,v[0]-base] for u,b,v in zip(ug.ravel(),bg.ravel(),x.reshape(-1,2)))
        bracket=[]
        d=x[:,25,0]-base
        for j in range(len(d)-1):
            if d[j]<=0<d[j+1]:bracket=[float(INTERACTION_BETA[j]),float(INTERACTION_BETA[j+1])];break
        metrics.append(dict(T=T,h1_pure_independent=base,
            h1_u05_beta048=float(x[24,25,0]),D_h1_u05_beta048=float(x[24,25,0]-base),
            beta_equality_bracket_at_u05=bracket,
            h2_beta0_u08=float(x[0,40,1]),h2_beta0_u1=float(x[0,50,1]),
            h2_difference_u08_minus_u1=float(x[0,40,1]-x[0,50,1])))
    checks=dict(horizons=list(JOINT_HORIZONS),parameter_pairs_per_horizon=int(ug.size),total_conditions=int(ug.size*len(JOINT_HORIZONS)),
        step_halving_max_error=step_error,independent_dop853_max_error=independent_error,
        original_surfaces_max_errors=old_errors,state_min=float(coarse.min()),state_max=float(coarse.max()),
        tolerance=TOL,all_passed=True,metrics=metrics)
    path=ROOT/'results/joint_four_horizons_checks.json'
    path.write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    settings=ROOT/'results/settings.json';s=json.loads(settings.read_text(encoding='utf-8'))
    s['joint_horizons']=list(JOINT_HORIZONS);s['joint_conditions']=ug.size*len(JOINT_HORIZONS)
    settings.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(checks,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':run()
