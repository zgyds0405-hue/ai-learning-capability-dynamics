"""Readable numeric ticks with no redundant trailing zeros."""
import math
from matplotlib.ticker import FuncFormatter, ScalarFormatter

def trimmed_tick(value, position=None):
    if abs(value)<1e-12:
        return '0'
    if abs(value)<.001:
        exponent=math.floor(math.log10(abs(value)))
        coefficient=f'{value/10**exponent:.3f}'.rstrip('0').rstrip('.')
        return '$'+coefficient+r'\times10^{'+str(exponent)+'}$'
    return f'{value:.3f}'.rstrip('0').rstrip('.').replace('-', '\N{MINUS SIGN}')

def apply_trimmed_tick_formatters(fig):
    for ax in fig.axes:
        for axis in [ax.xaxis,ax.yaxis]:
            if isinstance(axis.get_major_formatter(),ScalarFormatter):
                axis.set_major_formatter(FuncFormatter(trimmed_tick))
    fig.canvas.draw()
