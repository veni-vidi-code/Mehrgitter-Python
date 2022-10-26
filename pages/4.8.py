import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output, State, ctx

from Utils.components import snipping_switch
from implementations.dirichlect import get_dirichlect_generator
from implementations.helpers import N_l
from pages.cache import cache

dash.register_page(__name__, name="Anzahl Iterationen", order=2)

layout = html.Div(children=[
    html.H1(children='Anzahl Iterationen'),
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
        dbc.Button(
            "Hinzufügen",
            id="submit-button-4-8",
            color="info",
            outline=True,
            n_clicks=0,
        ),
    ]),
    html.Br(),
    dcc.Graph(id='iter-graph-4-8', mathjax=True),
])


def find_needed_iters(stufenindex_l, j, w: float = 0.5, limit: float = 1e-2, mode=""):
    generator = get_dirichlect_generator(stufenindex_l, j, w, iterator=mode)
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
def _iter_trace(stufenindex_l, w, mode):
    x = list(range(1, N_l(stufenindex_l) + 1))
    y = [find_needed_iters(stufenindex_l, i, w, mode=mode) for i in x]
    return x, y


def _add_iters_trace(stufenindex_l, w, fig, mode=""):
    x, y = _iter_trace(stufenindex_l, w, mode)
    fig.add_trace(
        go.Scatter(x=x, y=y, mode='lines+markers', name=f'w={w}, {"Jacobi" if mode == "jacobi" else "Gauss-Seidel"}'))
    x, y = _iter_trace(stufenindex_l, w, "zweigitter-" + mode)
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers',
                             name=f'w={w}, Zweigitter {"Jacobi" if mode == "jacobi" else "Gauss-Seidel"}'))


@callback(Output('iter-graph-4-8', 'figure'), Output('w-4-8', 'value'),
          Input('submit-button-4-8', 'n_clicks'), Input('l-4-8', 'value'),
          State('w-4-8', 'value'), State('iter-graph-4-8', 'figure'), Input('tabs-jacobi-gaussseidel-switch', 'value'))
def add_traces(n_clicks, stufenindex_l, w, fig, mode):
    if ctx.triggered_id is not None and ctx.triggered_id.startswith('submit-button-4-8'):
        fig = go.Figure(fig)
        _add_iters_trace(stufenindex_l, w, fig, mode)
        return fig, w
    else:
        fig = go.Figure(layout=go.Layout(
            yaxis={"title": 'Iterationen'},
            xaxis={"title": "$$j$$"}))
        _add_iters_trace(stufenindex_l, 0.5, fig, mode)
        return fig, 0.5


@callback(Output('w-4-8', 'step'), Input('snapping', 'on'))
def snapping(value):
    if value:
        return None
    else:
        return 1e-6
