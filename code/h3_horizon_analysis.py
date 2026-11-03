"""Fixed-policy trajectories and horizon-dependent evaluation of h3.

h3 is an index derived from two states, never a third differential equation.
No switching-policy search is performed. Units of time are uncalibrated.
"""
from pathlib import Path
import csv, json
import numpy as np
from scipy.integrate import solve_ivp
import model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/h3_horizon'

def write_csv(name, fields, rows):
    with (OUT/name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    u = np.linspace(0, 1, 201)
    time, x = model.simulate(u=u, horizon=20.)
    tf, xf = model.simulate(u=u, horizon=20., max_step=.005)
    error = float(np.abs(x-xf[::2]).max())
    assert np.allclose(time,tf[::2],atol=1e-12,rtol=0) and error < 1e-7
    rep_indices = np.array([0,50,100,150,200])
    independent_error = 0.
    for j in rep_indices:
        sol = solve_ivp(lambda t,y: model.rhs(y,float(u[j])),(0,20),[model.H10,model.H20],
                        method='DOP853',t_eval=time,rtol=1e-12,atol=1e-14)
        assert sol.success
        independent_error=max(independent_error,float(np.abs(sol.y.T-x[:,j,:]).max()))
    assert independent_error < 1e-7
    h3=x.mean(axis=2)
    np.savez_compressed(OUT/'fixed_policy_trajectories.npz',time=time,u=u,h1=x[:,:,0],h2=x[:,:,1],h3=h3)
    write_csv('representative_trajectories.csv',['T','u','h1','h2','h3'],
              (dict(T=float(t),u=float(u[j]),h1=float(x[i,j,0]),h2=float(x[i,j,1]),h3=float(h3[i,j]))
               for i,t in enumerate(time) for j in rep_indices))
    best = np.stack([np.argmax(x[:,:,0],axis=1),np.argmax(x[:,:,1],axis=1),np.argmax(h3,axis=1)],axis=1)
    write_csv('best_fixed_u_by_horizon.csv',['T','best_u_h1','best_u_h2','best_u_h3','h1_max','h2_max','h3_max','all_tied'],
              (dict(T=float(t),best_u_h1=float(u[best[i,0]]) if i else '',
                    best_u_h2=float(u[best[i,1]]) if i else '',best_u_h3=float(u[best[i,2]]) if i else '',
                    h1_max=float(x[i,best[i,0],0]),h2_max=float(x[i,best[i,1],1]),
                    h3_max=float(h3[i,best[i,2]]),all_tied=int(i==0)) for i,t in enumerate(time)))
    selected=[]
    for T in [.1,.25,.5,1,2,4,8,12,20]:
        i=int(round(T/.01)); k=best[i,2]
        rp=[rep_indices[np.argmax(x[i,rep_indices,j])] for j in [0,1]]
        rp.append(rep_indices[np.argmax(h3[i,rep_indices])])
        selected.append(dict(T=T,best_preset_u_h1=float(u[rp[0]]),best_preset_u_h2=float(u[rp[1]]),
                             best_preset_u_h3=float(u[rp[2]]),best_grid_u_h3=float(u[k]),
                             h1_at_best_h3=float(x[i,k,0]),h2_at_best_h3=float(x[i,k,1]),h3_max=float(h3[i,k])))
    write_csv('selected_horizons.csv',list(selected[0]),selected)
    # Reweight the same trajectories; weights change the evaluation, not dynamics.
    weights=np.linspace(0,1,101)
    weight_best=np.empty((len(time),len(weights)));weight_max=np.empty_like(weight_best)
    for j,w in enumerate(weights):
        value=w*x[:,:,0]+(1-w)*x[:,:,1]
        ix=np.argmax(value,axis=1)
        weight_best[:,j]=u[ix];weight_max[:,j]=value[np.arange(len(time)),ix]
    weight_best[0,:]=np.nan  # all policies share the initial state
    np.savez_compressed(OUT/'weight_sensitivity.npz',time=time,w1=weights,w2=1-weights,best_u=weight_best,h3_max=weight_max)
    weight_rows=[dict(T=T,w1=float(w),w2=float(1-w),best_u=float(weight_best[int(round(T/.01)),j]),
                      h3_max=float(weight_max[int(round(T/.01)),j]))
                 for T in [.5,1,4,20] for j,w in enumerate(weights)]
    write_csv('weight_sensitivity_selected.csv',list(weight_rows[0]),weight_rows)
    # Boundary check reuses original joint scans instead of refitting parameters.
    boundary=[]
    joint_root=ROOT/'results/data'
    if joint_root.exists():
        for T in [4,20]:
            with (joint_root/f'08_joint_T{T}.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
            for beta in sorted({float(v['beta']) for v in rows}):
                subset=[v for v in rows if float(v['beta'])==beta]
                v=max(subset,key=lambda r:(float(r['h1'])+float(r['h2']))/2)
                boundary.append(dict(T=T,beta=beta,best_u=float(v['u']),h3_max=(float(v['h1'])+float(v['h2']))/2))
        write_csv('internalization_boundary.csv',list(boundary[0]),boundary)
    # Independent u=0 analytic special case, checked at every saved time.
    analytic_error=float(np.abs(x[:,0,:]-model.independent(time)).max())
    assert analytic_error < 1e-7
    summary=dict(description='A distinct constant u is selected ex ante for each horizon, never a switching trajectory.',
                 parameters=model.PARAM,initial=[model.H10,model.H20],w1=.5,w2=.5,
                 u_grid_step=.005,n_u=len(u),n_time=len(time),dt=.01,max_T=20.,
                 weight_step=.01,selected_horizons=selected,
                 weight_examples=[v for v in weight_rows if round(v['w1'],2) in [0.,.25,.5,.75,1.]],
                 boundary_examples=[v for v in boundary if v['beta'] in [0.,.48]],
                 numerical_checks=dict(step_halving_max_error=error,DOP853_max_error=independent_error,
                                       independent_analytic_max_error=analytic_error,state_min=float(x.min()),state_max=float(x.max()),
                                       clipping=False,tolerance=1e-7),
                 tie_rule='At T=0 all u are tied; optimal-u arrays use NaN or a blank. Otherwise first grid maximizer is retained.',
                 limitation='Grid maxima among fixed proportions; neither continuous nor arbitrary time-varying global optimality is asserted.')
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
