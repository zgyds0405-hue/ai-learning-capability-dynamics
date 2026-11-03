"""Redraw saved trajectories with trimmed ticks; no ODE solver is called."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
import plot_figures as plots
from model import HORIZONS,JOINT_HORIZONS

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/data'

def process(name):
    p=DATA/f'{name}_trajectories.csv'
    header=p.open(encoding='utf-8-sig').readline().strip().split(',')
    data=np.loadtxt(p,delimiter=',',skiprows=1,encoding='utf-8-sig',
                    usecols=[header.index(v) for v in ['t','h1','h2']])
    t=np.unique(data[:,0]);blocks=data.reshape(len(t),-1,3)
    assert np.array_equal(blocks[...,0],np.broadcast_to(t[:,None],blocks[...,0].shape))
    return t,blocks[...,1:]

def main():
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob('*.csv')}
    out={key:process(name) for key,name in [
        ('allocation','01_allocation'),('beta','02_beta'),('mechanisms','03_mechanisms'),
        ('coupling','04_coupling'),('initial','06_initial')]}
    end=np.loadtxt(DATA/'05_horizon_endpoints.csv',delimiter=',',skiprows=1,encoding='utf-8-sig',usecols=[0,2,3])
    path=np.loadtxt(DATA/'05_horizon_trajectories.csv',delimiter=',',skiprows=1,encoding='utf-8-sig',usecols=[0,1,4,5])
    out['horizons']=(np.stack([end[end[:,0]==T,1:] for T in HORIZONS]),
                     [(path[path[:,0]==T,1],path[path[:,0]==T,2:]) for T in HORIZONS])
    out['joint']=[np.loadtxt(DATA/f'08_joint_T{T}.csv',delimiter=',',skiprows=1,encoding='utf-8-sig',usecols=[3,4]).reshape(41,51,2) for T in JOINT_HORIZONS]
    summary=json.loads((ROOT/'results/summary.json').read_text(encoding='utf-8'))
    standalone=(ROOT.parent/'main.tex').exists() and ROOT.name=='reproduction'
    if standalone:
        plots.IMAGES=ROOT.parent/'figures';plots.QA=ROOT.parent/'output/qa/trimmed_ticks'
    plots.draw_all(out,summary)
    assert all(hashlib.sha256((DATA/name).read_bytes()).hexdigest()==h for name,h in before.items())
    if not standalone:
        base=ROOT/'LATEX文档'
        for entry in plots.RECORDS:
            for ext in ['pdf','png','svg']:
                shutil.copy2(plots.IMAGES/f"{entry['figure']}.{ext}",base/f"figures/{entry['figure']}.{ext}")
        for name in ['plot_figures.py','tick_display.py','redraw_saved_figures.py']:
            shutil.copy2(ROOT/'code'/name,base/'reproduction/code'/name)
    print('Saved CSV hashes unchanged; figures redrawn without calling the ODE solver.')

if __name__=='__main__':main()
