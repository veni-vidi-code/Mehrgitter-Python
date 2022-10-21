from implementations.mehrgitterhelper import fourier_mode, dirichlect_randwert_a_l
from implementations.jacobi import jacobi_steps

import numpy as np
import dash
from dash import html, dcc, callback, Input, Output, State, ctx

import plotly.graph_objects as go

from pages.cache import cache

dash.register_page(__name__, name="4-8")

layout = html.Div(children=[
    html.H1(children='Iterationen relaxiertes Jacobi Verfahren'),
    html.Div([
        dcc.Checklist(
            options=[
                {'label': 'Easy Snapping', 'value': 'y'}
            ],
            value=['y'],
            id="snapping-4-8"
        ),
        "Gitter (l): ",
        dcc.Slider(1, 5, 1, value=3, id="l-4-8"),
        "Dämpfung (w): ",
        dcc.Slider(0, 0.5, step=0.000001, marks={
            1: '1',
            0.5: '1/2',
            1 / 3: '1/3',
            1 / 4: '1/4',
            1 / 8: '1/8',
            0: '0'
        }, value=0.5, id="w-4-8", tooltip={"placement": "bottom"}),
        html.Button(id='submit-button-4-8', children='Hinzufügen', n_clicks=0),
    ]),
    html.Br(),
    dcc.Graph(id='iter-graph-4-8'),
])


def find_needed_iters(stufenindex_l, j, w: float = 0.5, limit: float = 1e-2):
    e = fourier_mode(stufenindex_l, j)
    a = dirichlect_randwert_a_l(stufenindex_l)
    h = 1 / (2 ** (stufenindex_l + 1))
    m = np.identity(a.shape[0], dtype=a.dtype) - w * h * h * a
    eps = np.linalg.norm(e)
    i = 0
    while eps > limit:
        e = np.dot(m, e)
        eps = np.linalg.norm(e)
        i += 1
        if i > 500:
            return 0
    return i


@cache.memoize()
def _iter_trace(stufenindex_l, w):
    x = list(range(1, (2 ** (stufenindex_l + 1))))
    y = [find_needed_iters(stufenindex_l, i, w) for i in x]
    return x, y


def _add_iters_trace(stufenindex_l, w, fig):
    x, y = _iter_trace(stufenindex_l, w)
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=f'w={w}'))


@callback(Output('iter-graph-4-8', 'figure'), Output('w-4-8', 'value'),
          Input('submit-button-4-8', 'n_clicks'), Input('l-4-8', 'value'),
          State('w-4-8', 'value'), State('iter-graph-4-8', 'figure'))
def add_traces(n_clicks, stufenindex_l, w, fig):
    if ctx.triggered_id is not None and ctx.triggered_id.startswith('submit-button-4-8'):
        fig = go.Figure(fig)
        _add_iters_trace(stufenindex_l, w, fig)
        return fig, w
    else:
        fig = go.Figure()
        _add_iters_trace(stufenindex_l, 0.5, fig)
        return fig, 0.5


@callback(Output('w-4-8', 'step'), Input('snapping-4-8', 'value'))
def snapping(value):
    if value:
        return None
    else:
        return 0.000001
