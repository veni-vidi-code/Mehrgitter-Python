import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output, State, ALL

from Utils.components import snipping_switch
from implementations.Gitter import standard_schrittweitenfolge
from implementations.dirichlect_ndarrays import dirichlect_randwert_a_l
from implementations.gaussseidel import gauss_seidel_matrices
from implementations.helpers import N_l
from implementations.jacobi import jacobi_matrices
from implementations.merhgitterverfahren import mehrgitterverfahren_rekursiv
from pages.cache import cache

dash.register_page(__name__, name="Lösungsentwicklung Mehrgitter", order=6)

max_l = 5

example_startfault = np.array([0.75, 0.2, 0.6, 0.45, 0.9, 0.6, 0.8,
                               0.85, 0.55, 0.7, 0.9, 0.5, 0.6, 0.3, 0.2])  # This is the one in the book
startfaults = [(np.random.rand(N_l(l)) if l != 3 else example_startfault) for l in range(2, max_l + 1)]

default_array_div = []
for l in range(1, max_l + 1):
    x = []
    vec = startfaults[l - 2]
    for i in range(len(vec)):
        x.append(dcc.Input(placeholder=f"Start Fehler {i}", type="number", value=vec[i],
                           id={'type': 'startfault-4-22',
                               'index': i},
                           inputMode="numeric"))
        x.append(html.Br())
    default_array_div.append(x)

layout = html.Div(children=[
    html.H1(children='Fehler Dämpfung'),
    html.Div([
        snipping_switch,
        "Gitter (l): ",
        dcc.Slider(2, max_l, 1, value=3, id="l-4-22"),
        "Dämpfung (w): ",
        dcc.Slider(0, 0.5, step=1e-6, marks={
            0.5: '1/2',
            1 / 3: '1/3',
            1 / 4: '1/4',
            1 / 8: '1/8'
        }, value=1 / 4, id="w-4-22", tooltip={"placement": "bottom"}),
        dcc.Markdown("$$\\gamma$$:", mathjax=True),
        dcc.Slider(1, 3, step=1, value=1, id="gamma-4-22"),
        dbc.Button(
            "Start Fehler",
            id="collapse-button-4-22",
            color="info",
            outline=True,
            n_clicks=0,
        ),
        dbc.Collapse(
            dbc.Card(dbc.CardBody([html.Div(id="fault_vector_div-4-22"),
                                   dbc.Button(
                                       "Update",
                                       id="submit-button-4-22",
                                       color="primary",
                                       n_clicks=0,
                                   )
                                   ])),
            id="collapse-4-22",
            is_open=False,
        ),

    ]),
    html.Br(),
    dcc.Graph(id='graph-4-22', mathjax=True),
])


def f_x(x):
    return np.pi * np.pi / 8. * (9. * np.sin(3. * np.pi * x / 2.) + 25 * np.sin(5 * np.pi * x / 2.))


def u_x(x):  # Lösung von f_x
    return np.sin(2. * np.pi * x) * np.cos(np.pi * x / 2.)


@cache.memoize()
def generate_figure(stufenindex_l, w, start: np.ndarray, mode, *, v1: int=2, v2: int=2, gamma: int = 1):
    x = standard_schrittweitenfolge(stufenindex_l)
    f = f_x(x)
    u_star = u_x(x)
    u_0 = u_star + start
    matrices = jacobi_matrices if mode == "jacobi" else gauss_seidel_matrices
    u_mgm = mehrgitterverfahren_rekursiv(stufenindex_l, v1, v2, u_0, f, matrices, w1=2 * w, w2=2 * w, gamma=gamma)
    fig = go.Figure(layout=go.Layout(
        yaxis={"title": "$$u$$"},
        xaxis={"title": "$$x$$"}))
    fig.add_trace(go.Scatter(x=x, y=u_star, name=f"$$u^{{{stufenindex_l},\\star}}$$"))
    fig.add_trace(go.Scatter(x=x, y=u_mgm, name=f"$$u^{{{stufenindex_l},MGM}}$$"))
    fig.add_trace(go.Scatter(x=x, y=u_0, name=f"$$u_0^{{{stufenindex_l}}}$$"))
    fig.add_trace(go.Scatter(x=x, y=(dirichlect_randwert_a_l(stufenindex_l) @ u_mgm) - f, name=f"$$abw$$"))
    return fig


@callback(Output('graph-4-22', 'figure'), Output('fault_vector_div-4-22', 'children'),
          Input('l-4-22', 'value'),
          Input('w-4-22', 'value'),
          State({'type': 'startfault-4-22', 'index': ALL}, 'value'),
          Input('submit-button-4-22', 'n_clicks'), Input('tabs-jacobi-gaussseidel-switch', 'value'),
          Input('gamma-4-22', 'value'))
def add_traces(stufenindex_l, w, vector, n_clicks, mode, gamma):
    if dash.ctx.triggered_id is None or dash.ctx.triggered_id.startswith("l-4-22"):
        vector = startfaults[stufenindex_l - 2]
        return generate_figure(stufenindex_l, w,
                               np.array(vector), mode, gamma=int(gamma)), default_array_div[stufenindex_l - 1]
    else:
        return generate_figure(stufenindex_l, w, np.array(vector), mode, gamma=int(gamma)), dash.no_update


@callback(Output('w-4-22', 'step'), Input('snapping', 'on'))
def snapping(value):
    if value:
        return None
    else:
        return 1e-6


@callback(
    Output("collapse-4-22", "is_open"),
    [Input("collapse-button-4-22", "n_clicks")],
    [State("collapse-4-22", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open
