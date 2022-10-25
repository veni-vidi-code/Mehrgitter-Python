from Utils.components import snipping_switch
from implementations.dirichlect import get_jacobi_generator, N_l

import numpy as np
import dash
from dash import html, dcc, callback, Input, Output, State, ctx

import plotly.graph_objects as go

from pages.cache import cache

dash.register_page(__name__, name="Iterationen Dämpfung Jacobi")

layout = html.Div(children=[
    html.H1(children='Iterationen relaxiertes Jacobi Verfahren'),
    html.Div([
        snipping_switch,
        "Gitter (l): ",
        dcc.Slider(1, 5, 1, value=3, id="l-4-8"),
        "Dämpfung (w): ",
        dcc.Slider(0, 0.5, step=1e-6, marks={
            1: '1',
            0.5: '1/2',
            1 / 3: '1/3',
            1 / 4: '1/4',
            1 / 8: '1/8'
        }, value=0.5, id="w-4-8", tooltip={"placement": "bottom"}),
        html.Button(id='submit-button-4-8', children='Hinzufügen', n_clicks=0),
    ]),
    html.Br(),
    dcc.Graph(id='iter-graph-4-8'),
])


def find_needed_iters(stufenindex_l, j, w: float = 0.5, limit: float = 1e-2):
    generator = get_jacobi_generator(stufenindex_l, j, w)
    e = next(generator)
    eps = np.linalg.norm(e)
    i = 0
    while eps > limit:
        e = next(generator)
        eps = np.linalg.norm(e)
        i += 1
        if i > 500:
            return 0
    return i


@cache.memoize()
def _iter_trace(stufenindex_l, w):
    x = list(range(1, N_l(stufenindex_l) + 1))
    y = [find_needed_iters(stufenindex_l, i, w) for i in x]
    return x, y


def _add_iters_trace(stufenindex_l, w, fig):
    x, y = _iter_trace(stufenindex_l, w)
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=f'w={w}'))


@callback(Output('iter-graph-4-8', 'figure'), Output('w-4-8', 'value'),
          Input('submit-button-4-8', 'n_clicks'), Input('l-4-8', 'value'),
          State('w-4-8', 'value'), State('iter-graph-4-8', 'figure'), Input('tabs-jacobi-gausseidel-switch', 'value'))
def add_traces(n_clicks, stufenindex_l, w, fig, mode):
    if ctx.triggered_id is not None and ctx.triggered_id.startswith('submit-button-4-8'):
        fig = go.Figure(fig)
        _add_iters_trace(stufenindex_l, w, fig)
        return fig, w
    else:
        fig = go.Figure()
        _add_iters_trace(stufenindex_l, 0.5, fig)
        return fig, 0.5


@callback(Output('w-4-8', 'step'), Input('snapping', 'on'))
def snapping(value):
    if value:
        return None
    else:
        return 1e-6
