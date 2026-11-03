"""Experiments, source data and independent numerical verification."""
from pathlib import Path
import csv
import json
import platform
import numpy as np
from scipy.integrate import solve_ivp
from model import (PARAM,H10,H20,PROCESS_U,PROCESS_HORIZON,REP_U,BETA_SCENARIOS,
    H20_SCENARIOS,HORIZONS,GRID,INTERACTION_BETA,DT,FINE_DT,TOL,CHECKS,STATE_AUDIT,
    MODES,MODEL_VERSION,JOINT_HORIZONS,collaboration_condition,rhs,simulate,verified,independent)

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results'/'data'

def write_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def csv_write(name,header,rows):
    DATA.mkdir(parents=True,exist_ok=True)
    with (DATA/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def export_process(name,t,x,labels,*,u=PROCESS_U,p=None,h20=H20,modes=None):
    n=len(labels)
    us=np.broadcast_to(u,(n,))
    starts2=np.broadcast_to(h20,(n,))
    params=[dict(PARAM) for _ in range(n)] if p is None else p
    modes=[(True,True)]*n if modes is None else modes
    rates=np.stack([rhs(x[:,j],us[j],params[j],dynamic=modes[j],anchors=(H10,starts2[j]))
                    for j in range(n)],axis=1)
    header=['case','t','u','h10','h20',*PARAM,'dynamic_h2_to_h1','dynamic_h1_to_h2',
            'h1','h2','dh1_dt','dh2_dt']
    def row(i,j):
        return [labels[j],t[i],us[j],H10,starts2[j],*params[j].values(),
                *map(int,modes[j]),*x[i,j],*rates[i,j]]
    csv_write(f'{name}_trajectories.csv',header,
              (row(i,j) for i in range(len(t)) for j in range(n)))
    csv_write(f'{name}_endpoints.csv',header,(row(-1,j) for j in range(n)))
    return rates

def special_checks():
    errors={}
    # New normalization endpoints and degenerate support/coupling cases.
    for eps in [0.,.1,1.]:
        errors[f'g0_eps{eps:g}']=abs(float(collaboration_condition(0.,eps))-eps)
        errors[f'g1_eps{eps:g}']=abs(float(collaboration_condition(1.,eps))-1.)
    t,x=simulate(.5,p={'eps':1.},h10=np.array([0.,.2,1.]))
    errors['eps1_h2_independent_of_h1_initial']=float(np.ptp(x[...,1],axis=1).max())
    rate=PARAM['c']*.5
    exact=rate/(rate+PARAM['delta2'])+(H20-rate/(rate+PARAM['delta2']))*np.exp(-(rate+PARAM['delta2'])*t)
    errors['eps1_h2_analytic']=float(np.abs(x[...,1]-exact[:,None]).max())
    _,x=simulate(1.,h10=0.,h20=0.,p={'eps':0.})
    errors['eps0_u1_zero_state_trap']=float(np.abs(x).max())
    try:
        simulate(p={'eps':1.01})
    except ValueError:
        pass
    else:
        raise AssertionError('eps above one was accepted')
    t,x=simulate(0.)
    errors['u0_analytic_full_trajectory']=float(np.abs(x-independent(t)).max())
    for u in REP_U:
        t,x=simulate(u,p={'beta':0})
        f=PARAM['a']*(1-u)
        r=f+PARAM['delta1']
        exact=f/r+(H10-f/r)*np.exp(-r*t)
        errors[f'beta0_h1_analytic_u{u:g}']=float(np.abs(x[:,0]-exact).max())
    t,x=simulate(0.,p={'beta':BETA_SCENARIOS})
    errors['u0_beta_invariance']=float(np.ptp(x,axis=1).max())
    _,x=simulate(.5,p={'beta':0.},h20=H20_SCENARIOS)
    errors['beta0_h1_h20_invariance']=float(np.ptp(x[...,0],axis=1).max())
    start=np.array([H10,H20])
    errors['frozen_controls_same_initial_vector_field']=max(
        float(np.abs(rhs(start,PROCESS_U,dynamic=flags)-rhs(start,PROCESS_U)).max())
        for flags in MODES.values())
    t,x=simulate(REP_U)
    initial=np.broadcast_to(start,(len(REP_U),2)).copy()
    ref=solve_ivp(lambda t,y:rhs(y.reshape(-1,2),REP_U).ravel(),(0,PROCESS_HORIZON),
                  initial.ravel(),t_eval=t,method='DOP853',rtol=1e-11,atol=1e-13)
    if not ref.success:
        raise AssertionError(ref.message)
    errors['rk4_vs_dop853_full_trajectories']=float(np.abs(x-ref.y.T.reshape(x.shape)).max())
    long=simulate(.5)[1]
    errors['deadline_prefix_invariance']=max(float(np.abs(
        simulate(.5,horizon=float(T))[1]-long[:round(T/DT)+1]).max()) for T in HORIZONS)
    for eps in [0.,.1,1.]:
        for u in [0.,.5,1.]:
            for h1 in [0.,.5,1.]:
                for h2 in [0.,.5,1.]:
                    d=rhs(np.array([h1,h2]),u,{**PARAM,'eps':eps})
                    if (h1==0 and d[0]<0) or (h1==1 and d[0]>0):
                        raise AssertionError('H1 boundary vector points outward')
                    if (h2==0 and d[1]<0) or (h2==1 and d[1]>0):
                        raise AssertionError('H2 boundary vector points outward')
    if max(errors.values())>=TOL:
        raise AssertionError(errors)
    return errors

def run_experiments():
    DATA.mkdir(parents=True,exist_ok=True)
    CHECKS.clear()
    STATE_AUDIT.update(min=float('inf'),max=-float('inf'))
    out,summary={},{}
    print('1/8 Time allocation',flush=True)
    t,x=verified('allocation',u=REP_U)
    export_process('01_allocation',t,x,[f'u={u:g}' for u in REP_U],u=REP_U)
    out['allocation']=t,x
    summary['allocation']=[dict(u=float(u),h1=float(v[0]),h2=float(v[1])) for u,v in zip(REP_U,x[-1])]
    print('2/8 Internalization strength',flush=True)
    t,x=verified('beta',p={'beta':BETA_SCENARIOS})
    pp=[{**PARAM,'beta':float(b)} for b in BETA_SCENARIOS]
    export_process('02_beta',t,x,[f'beta={b:g}' for b in BETA_SCENARIOS],p=pp)
    out['beta']=t,x
    summary['beta']=[dict(beta=float(b),h1=float(v[0]),h2=float(v[1])) for b,v in zip(BETA_SCENARIOS,x[-1])]
    print('3/8 Six parameter trajectory perturbations',flush=True)
    cases=[('Baseline',dict(PARAM))]
    for key in PARAM:
        cases.extend((f'{key} x{factor:g}',{**PARAM,key:PARAM[key]*factor}) for factor in [.8,1.2])
    pv={key:np.array([p[key] for _,p in cases]) for key in PARAM}
    t,x=verified('mechanisms',p=pv)
    export_process('03_mechanisms',t,x,[label for label,_ in cases],p=[p for _,p in cases])
    out['mechanisms']=t,x
    effects=[]
    for j,key in enumerate(PARAM):
        minus,plus=x[-1,1+2*j],x[-1,2+2*j]
        base=x[-1,0]
        effects.append(dict(parameter=key,baseline_value=PARAM[key],
            h1_minus=float(minus[0]),h1_plus=float(plus[0]),h2_minus=float(minus[1]),h2_plus=float(plus[1]),
            dh1_minus=float(minus[0]-base[0]),dh1_plus=float(plus[0]-base[0]),
            dh2_minus=float(minus[1]-base[1]),dh2_plus=float(plus[1]-base[1]),
            span_h1=float(abs(plus[0]-minus[0])),span_h2=float(abs(plus[1]-minus[1]))))
    csv_write('07_parameter_effects.csv',list(effects[0]),(r.values() for r in effects))
    summary['parameter_effects']=effects
    print('4/8 Dynamic coupling controls',flush=True)
    paths=[verified(f'coupling_{label}',dynamic=flags) for label,flags in MODES.items()]
    t=paths[0][0]
    x=np.stack([v[1] for v in paths],axis=1)
    export_process('04_coupling',t,x,list(MODES),modes=list(MODES.values()))
    out['coupling']=t,x
    inter=x[:,0]-x[:,1]-x[:,2]+x[:,3]
    csv_write('04_coupling_interaction.csv',['t','I_h1','I_h2'],([tt,*v] for tt,v in zip(t,inter)))
    summary['coupling']=[dict(mode=k,h1=float(v[0]),h2=float(v[1])) for k,v in zip(MODES,x[-1])]
    summary['coupling_interaction_T']=inter[-1].tolist()
    print('5/8 Observation deadlines',flush=True)
    endpoints,rows,normrows,paths=[],[],[],[]
    for T in HORIZONS:
        xx=verified(f'deadline_{T:g}',u=GRID,horizon=float(T),keep=False)
        endpoints.append(xx)
        base=independent(T)
        rows.extend([T,u,*v,*base,v[0]-base[0]] for u,v in zip(GRID,xx))
        tt,path=simulate(PROCESS_U,horizon=float(T))
        paths.append((tt,path))
        normrows.extend([T,ti,ti/T,PROCESS_U,*v] for ti,v in zip(tt,path))
    csv_write('05_horizon_endpoints.csv',['T','u','h1','h2','h1_u0','h2_u0','D_h1'],rows)
    csv_write('05_horizon_trajectories.csv',['T','t','t_over_T','u','h1','h2'],normrows)
    out['horizons']=np.stack(endpoints),paths
    exact=np.stack([simulate(REP_U,horizon=float(T),keep=False) for T in HORIZONS])
    summary['horizons']=[dict(T=float(T),h1=v[:,0].tolist(),h2=v[:,1].tolist(),D=(v[:,0]-base).tolist())
                         for T,v,base in zip(HORIZONS,exact,independent(HORIZONS)[:,0])]
    print('6/8 Initial collaborative capability',flush=True)
    t,x=verified('initial_h20',h20=H20_SCENARIOS)
    export_process('06_initial',t,x,[f'h20={v:g}' for v in H20_SCENARIOS],h20=H20_SCENARIOS)
    out['initial']=t,x
    summary['initial']=[dict(h20=float(v),h1=float(y[0]),h2=float(y[1])) for v,y in zip(H20_SCENARIOS,x[-1])]
    print('7/8 Local sensitivity reuses the shared 13 cases',flush=True)
    print('8/8 Joint beta-allocation surfaces',flush=True)
    ug,bg=np.meshgrid(GRID,INTERACTION_BETA)
    joint=[]
    for T in JOINT_HORIZONS:
        xx=verified(f'joint_{T:g}',u=ug,p={'beta':bg},horizon=T,keep=False)
        joint.append(xx)
        ref=independent(T)[0]
        csv_write(f'08_joint_T{T:g}.csv',['T','u','beta','h1','h2','D_h1'],
             ([T,u,b,*v,v[0]-ref] for u,b,v in zip(ug.ravel(),bg.ravel(),xx.reshape(-1,2))))
    out['joint']=joint
    errors=special_checks()
    summary['baseline']=summary['allocation'][2]
    summary['initial_AI_marginal_h1_rate']=(1-H10)*(PARAM['beta']*H20-PARAM['a'])
    summary['instantaneous_h2_threshold']=PARAM['a']/PARAM['beta']
    summary['special_checks']=errors
    write_json(ROOT/'results'/'numerical_checks.json',dict(all_checks_passed=True,tolerance=TOL,
       method='RK4',max_step=DT,fine_step=FINE_DT,step_halving=CHECKS,special_checks=errors,
       state_range=STATE_AUDIT,max_step_halving_error=max(r['step_halving_max_error'] for r in CHECKS),
       dop853_rtol=1e-11,dop853_atol=1e-13,boundary_checks_passed=True,state_clipping_used=False))
    write_json(ROOT/'results'/'settings.json',dict(parameters=PARAM,h10=H10,h20=H20,
       process_horizon=PROCESS_HORIZON,process_u=PROCESS_U,u_scenarios=REP_U.tolist(),
       beta_scenarios=BETA_SCENARIOS.tolist(),h20_scenarios=H20_SCENARIOS.tolist(),horizons=HORIZONS.tolist(),
       allocation_grid=GRID.tolist(),interaction_beta_grid=INTERACTION_BETA.tolist(),joint_horizons=list(JOINT_HORIZONS),
       joint_conditions=len(JOINT_HORIZONS)*len(GRID)*len(INTERACTION_BETA),
       coupling_modes={k:list(v) for k,v in MODES.items()},perturbation_factors=[.8,1.2],unique_parameter_cases=13,
       model='autonomous; fixed u; two capability stocks; illustrative parameters',
       model_version=MODEL_VERSION,collaboration_condition='eps+(1-eps)*h1',eps_range=[0,1],
       runtime=dict(python=platform.python_version(),numpy=np.__version__)))
    write_json(ROOT/'results'/'summary.json',summary)
    return out,summary
