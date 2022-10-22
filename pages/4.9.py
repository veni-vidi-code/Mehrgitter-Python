from implementations.dirichlect import get_jacobi_generator, N_l

import numpy as np
import dash
from dash import html, dcc, callback, Input, Output, State, ALL

import plotly.graph_objects as go

from pages.cache import cache

dash.register_page(__name__, name="Fehler Dämpfung Jacobi")

max_l = 4

example_startfault = np.array([0.75, 0.2, 0.6, 0.45, 0.9, 0.6, 0.8,
                               0.85, 0.55, 0.7, 0.9, 0.5, 0.6, 0.3, 0.2])  # This is the one in the book
startfaults = [(np.random.rand(N_l(l)) if l != 3 else example_startfault) for l in range(1, max_l + 1)]

default_array_div = []
for l in range(1, max_l + 1):
    x = []
    vec = startfaults[l - 1]
    for i in range(len(vec)):
        x.append(dcc.Input(placeholder=f"Start Fehler {i}", type="number", value=vec[i],
                           id={'type': 'startfault-4-9',
                               'index': i},
                           inputMode="numeric"))
        x.append(html.Br())
    default_array_div.append(x)

layout = html.Div(children=[
    html.H1(children='Fehler relaxiertes Jacobi Verfahren'),
    html.Div([
        dcc.Checklist(
            options=[
                {'label': 'Easy Snapping', 'value': 'y'}
            ],
            value=['y'],
            id="snapping-4-9"
        ),
        "Gitter (l): ",
        dcc.Slider(1, max_l, 1, value=3, id="l-4-9"),
        "Dämpfung (w): ",
        dcc.Slider(0, 0.5, step=1e-6, marks={
            0.5: '1/2',
            1 / 3: '1/3',
            1 / 4: '1/4',
            1 / 8: '1/8'
        }, value=1 / 4, id="w-4-9", tooltip={"placement": "bottom"}),
        html.Details(children=[
            html.Summary(children="Start Fehler"),
            html.Div(id="fault_vector_div-4-9"),
            html.Button(id='submit-button-4-9', children='Update', n_clicks=0)
        ]),
    ]),
    html.Br(),
    dcc.Graph(id='iter-graph-4-9'),
])


def fault_after_steps(stufenindex_l, w: float, start: np.ndarray, steps: int = 2):
    if start is None:
        start = example_startfault
    generator = get_jacobi_generator(stufenindex_l, 0, w, start)
    return [next(generator) for _ in range(steps + 1)]


@cache.memoize()
def _generate_fig(stufenindex_l, w, start: np.ndarray):
    assert start.size == N_l(stufenindex_l)
    faults = fault_after_steps(stufenindex_l, w, start, 2)
    fig = go.Figure()
    x = np.arange(1, N_l(stufenindex_l) + 1)
    for i in range(len(faults)):
        fig.add_trace(go.Scatter(x=x, y=faults[i], name=f"Schritt {i}"))
    return fig


@callback(Output('iter-graph-4-9', 'figure'), Output('fault_vector_div-4-9', 'children'),
          Input('l-4-9', 'value'),
          Input('w-4-9', 'value'),
          State({'type': 'startfault-4-9', 'index': ALL}, 'value'),
          Input('submit-button-4-9', 'n_clicks'))
def add_traces(stufenindex_l, w, vector, n_clicks):
    if dash.ctx.triggered_id is None or dash.ctx.triggered_id.startswith("l-4-9"):
        vector = startfaults[stufenindex_l - 1]
        return _generate_fig(stufenindex_l, w, np.array(vector)), default_array_div[stufenindex_l - 1]
    else:
        return _generate_fig(stufenindex_l, w, np.array(vector)), dash.no_update


@callback(Output('w-4-9', 'step'), Input('snapping-4-9', 'value'))
def snapping(value):
    if value:
        return None
    else:
        return 1e-6
