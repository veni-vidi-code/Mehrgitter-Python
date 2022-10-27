import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output, State, ALL

from Utils.components import snipping_switch
from implementations.dirichlect import get_dirichlect_generator
from implementations.gaussseidel import gauss_seidel_matrices
from implementations.helpers import N_l
from implementations.jacobi import jacobi_matrices
from implementations.zweigitter import zweigitter_step
from implementations.merhgitterverfahren import mehrgitterverfahren_rekursiv
from pages.cache import cache

dash.register_page(__name__, name="Fehler Dämpfung", order=3)

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
    html.H1(children='Fehler Dämpfung'),
    html.Div([
        snipping_switch,
        "Gitter (l): ",
        dcc.Slider(1, max_l, 1, value=3, id="l-4-9"),
        "Dämpfung (w): ",
        dcc.Slider(0, 0.5, step=1e-6, marks={
            0.5: '1/2',
            1 / 3: '1/3',
            1 / 4: '1/4',
            1 / 8: '1/8'
        }, value=1 / 4, id="w-4-9", tooltip={"placement": "bottom"}),
        dcc.Dropdown(
            options=[
                {'label': dcc.Markdown("$$ZGM$$", mathjax=True), 'value': 'zgm'},
                {'label': dcc.Markdown("$$MGM (\\gamma = 1)$$", mathjax=True), 'value': 'mgm1'},
                {'label': dcc.Markdown("$$MGM (\\gamma = 2)$$", mathjax=True), 'value': 'mgm2'},
            ],
            value=[],
            clearable=True,
            multi=True,
            id="additional-taces-4-9",
            placeholder="Andere Verfahren",
        ),
        html.Br(),
        dbc.Button(
            "Start Fehler",
            id="collapse-button-4-9",
            color="info",
            outline=True,
            n_clicks=0,
        ),
        dbc.Collapse(
            dbc.Card(dbc.CardBody([html.Div(id="fault_vector_div-4-9"),
                                   dbc.Button(
                                       "Update",
                                       id="submit-button-4-9",
                                       color="primary",
                                       n_clicks=0,
                                   )
                                   ])),
            id="collapse-4-9",
            is_open=False,
        ),

    ]),
    html.Br(),
    dcc.Graph(id='iter-graph-4-9', mathjax=True),
])


@cache.memoize()
def fault_after_steps(stufenindex_l, w: float, start: np.ndarray, steps: int = 2, mode="jacobi"):
    if start is None:
        start = example_startfault
    generator = get_dirichlect_generator(stufenindex_l, 0, w, start, mode)
    return [next(generator) for _ in range(steps + 1)]


@cache.memoize()
def _generate_fig(stufenindex_l, w, start: np.ndarray, mode, additional_traces):
    assert start.size == N_l(stufenindex_l)
    faults = fault_after_steps(stufenindex_l, w, start, 2, mode)

    # for performance improvements this uses the already known result from faults
    matrices = jacobi_matrices if mode == "jacobi" else gauss_seidel_matrices

    fig = go.Figure(layout=go.Layout(
        yaxis={"title": "Fehler"},
        xaxis={"title": "$$k$$"}))
    x = np.arange(1, N_l(stufenindex_l) + 1)
    i = 0
    for i in range(len(faults)):
        fig.add_trace(go.Scatter(x=x, y=faults[i], name=f"$$e_{{{i}}}^{{{stufenindex_l}}}$$"))
    if "zgm" in additional_traces:
        y = zweigitter_step(stufenindex_l, 0, 2, faults[2], np.zeros_like(start), psi_vor_matrice=matrices, w1=2 * w,
                            w2=2 * w)
        fig.add_trace(go.Scatter(x=x, y=y, name=f"$$e_{{{i}}}^{{{stufenindex_l}, ZGM}}$$"))
    if "mgm1" in additional_traces:
        y = mehrgitterverfahren_rekursiv(stufenindex_l, 0, 2, faults[2], np.zeros_like(start), psi_vor_matrice=matrices,
                                         w1=2 * w, w2=2 * w, gamma=1)
        fig.add_trace(go.Scatter(x=x, y=y, name=f"$$e_{{{i}}}^{{{stufenindex_l}, MGM}}, \\gamma = 1$$"))
    if "mgm2" in additional_traces:
        y = mehrgitterverfahren_rekursiv(stufenindex_l, 0, 2, faults[2], np.zeros_like(start), psi_vor_matrice=matrices,
                                         w1=2 * w, w2=2 * w, gamma=2)
        fig.add_trace(go.Scatter(x=x, y=y, name=f"$$e_{{{i}}}^{{{stufenindex_l}, MGM}}, \\gamma = 2$$"))
    return fig


@callback(Output('iter-graph-4-9', 'figure'), Output('fault_vector_div-4-9', 'children'),
          Input('l-4-9', 'value'),
          Input('w-4-9', 'value'),
          State({'type': 'startfault-4-9', 'index': ALL}, 'value'),
          Input('submit-button-4-9', 'n_clicks'), Input('tabs-jacobi-gaussseidel-switch', 'value'),
          Input('additional-taces-4-9', 'value'))
def add_traces(stufenindex_l, w, vector, n_clicks, mode, additional_traces):
    if dash.ctx.triggered_id is None or dash.ctx.triggered_id.startswith("l-4-9"):
        vector = startfaults[stufenindex_l - 1]
        return _generate_fig(stufenindex_l, w, np.array(vector),
                             mode, additional_traces), default_array_div[stufenindex_l - 1]
    else:
        return _generate_fig(stufenindex_l, w, np.array(vector), mode, additional_traces), dash.no_update


@callback(Output('w-4-9', 'step'), Input('snapping', 'on'))
def snapping(value):
    if value:
        return None
    else:
        return 1e-6


@callback(
    Output("collapse-4-9", "is_open"),
    [Input("collapse-button-4-9", "n_clicks")],
    [State("collapse-4-9", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open
