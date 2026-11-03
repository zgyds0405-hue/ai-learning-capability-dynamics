"""Two capability dynamics; illustrative rates and uncalibrated time."""
from __future__ import annotations
import math
import numpy as np

PARAM = dict(a=.25, beta=.48, c=.55, eps=.10, delta1=.02, delta2=.03)
MODEL_VERSION = 'normalized-support-v2'


def collaboration_condition(h1, eps):
    """External baseline support and knowledge foundation; eps in [0,1]."""
    return eps + (1-eps)*h1
H10, H20 = .20, .10
PROCESS_HORIZON, PROCESS_U = 20., .5
REP_U = np.array([0., .25, .5, .75, 1.])
BETA_SCENARIOS = np.array([0., .12, .30, .48, .60])
H20_SCENARIOS = np.array([0., .1, .5, 1.])
HORIZONS = np.array([.5, 1., 2., 4., 8., 12., 20.])
JOINT_HORIZONS = (4, 8, 12, 20)
GRID = np.linspace(0., 1., 51)
INTERACTION_BETA = np.linspace(0., .8, 41)
DT, FINE_DT, TOL = .01, .005, 1e-7
CHECKS = []
STATE_AUDIT = dict(min=math.inf, max=-math.inf)
MODES = {'Full': (True, True), 'Freeze H2 input': (False, True),
         'Freeze H1 input': (True, False), 'Freeze both inputs': (False, False)}


def rhs(x, u, p=None, *, dynamic=(True, True), anchors=(H10, H20)):
    """Diagnostic flags enable H2 -> H1 and H1 -> H2 dynamic inputs.

    Frozen controls retain initial cross-channel inputs, own-state saturation
    and decay. These controls are not additional fitted model parameters.
    """
    p = PARAM if p is None else p
    h1, h2 = x[..., 0], x[..., 1]
    source2 = h2 if dynamic[0] else np.asarray(anchors[1])
    source1 = h1 if dynamic[1] else np.asarray(anchors[0])
    return np.stack(((1-h1)*(p['a']*(1-u)+p['beta']*u*source2)-p['delta1']*h1,
                     p['c']*u*collaboration_condition(source1,p['eps'])*(1-h2)-p['delta2']*h2), axis=-1)


def observe(x):
    if not np.isfinite(x).all():
        raise AssertionError('Non-finite state')
    STATE_AUDIT['min'] = min(STATE_AUDIT['min'], float(x.min()))
    STATE_AUDIT['max'] = max(STATE_AUDIT['max'], float(x.max()))
    if x.min() < -1e-12 or x.max() > 1+1e-12:
        raise AssertionError('State outside [0,1]; no clipping is used')


def simulate(u=PROCESS_U, horizon=PROCESS_HORIZON, h10=H10, h20=H20,
             p=None, max_step=DT, dynamic=(True, True), keep=True):
    supplied = {} if p is None else dict(p)
    if set(supplied)-set(PARAM):
        raise ValueError(f'Unknown model parameters: {set(supplied)-set(PARAM)}')
    p = {**PARAM, **supplied}
    arrays = np.broadcast_arrays(*[np.asarray(v, dtype=float) for v in [u, h10, h20, *p.values()]])
    u, h10, h20 = arrays[:3]
    p = dict(zip(PARAM, arrays[3:]))
    if any(not np.isfinite(v).all() for v in arrays):
        raise ValueError('All inputs must be finite')
    if any(np.any((v<0)|(v>1)) for v in [u, h10, h20]):
        raise ValueError('u and initial capabilities must lie in [0,1]')
    if any(np.any(v<0) for v in p.values()):
        raise ValueError('Rate parameters and eps must be nonnegative')
    if np.any(p['eps']>1):
        raise ValueError('eps must lie in [0,1]')
    if not math.isfinite(horizon) or horizon<0 or not math.isfinite(max_step) or max_step<=0:
        raise ValueError('Invalid horizon or integration step')
    if len(dynamic)!=2 or any(type(flag) is not bool for flag in dynamic):
        raise ValueError('dynamic must contain two boolean diagnostic flags')
    n = max(1, math.ceil(horizon/max_step))
    dt = horizon/n
    times = np.linspace(0, horizon, n+1)
    x = np.stack((h10, h20), axis=-1).copy()
    observe(x)
    states = np.empty((n+1, *x.shape)) if keep else None
    if keep:
        states[0] = x
    for i in range(n):
        args = dict(p=p, dynamic=dynamic, anchors=(h10,h20))
        k1 = rhs(x, u, **args)
        y2 = x+dt*k1/2
        observe(y2)
        k2 = rhs(y2, u, **args)
        y3 = x+dt*k2/2
        observe(y3)
        k3 = rhs(y3, u, **args)
        y4 = x+dt*k3
        observe(y4)
        k4 = rhs(y4, u, **args)
        x += dt*(k1+2*k2+2*k3+k4)/6
        observe(x)
        if keep:
            states[i+1] = x
    return (times,states) if keep else x


def verified(name, **kwargs):
    keep = kwargs.pop('keep',True)
    coarse = simulate(keep=keep,**kwargs)
    fine = simulate(keep=keep,max_step=FINE_DT,**kwargs)
    if keep:
        tc,xc = coarse
        tf,xf = fine
        if not np.allclose(tc,tf[::2],rtol=0,atol=1e-12):
            raise AssertionError('Coarse/fine grids do not align')
        error = float(np.abs(xc-xf[::2]).max())
    else:
        error = float(np.abs(coarse-fine).max())
    CHECKS.append(dict(name=name,step_halving_max_error=error,passed=error<TOL))
    if error>=TOL:
        raise AssertionError(f'{name}: step-halving error {error}')
    return coarse


def independent(t,h10=H10,h20=H20):
    t = np.asarray(t)
    r = PARAM['a']+PARAM['delta1']
    h1 = PARAM['a']/r+(h10-PARAM['a']/r)*np.exp(-r*t)
    return np.stack((h1,h20*np.exp(-PARAM['delta2']*t)),axis=-1)
