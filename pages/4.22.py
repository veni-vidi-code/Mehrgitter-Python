import dash
import dash_bootstrap_components as dbc
import dash_daq as daq
import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output, State, ALL

from Utils.components import snipping_switch, stufenindex_l_check, w_check, vec_check
from implementations.Gitter import standard_schrittweitenfolge
from implementations.gaussseidel import gauss_seidel_matrices
from implementations.helpers import N_l
from implementations.jacobi import jacobi_matrices
from implementations.merhgitterverfahren import mehrgitterverfahren_rekursiv, get_start_vector
from implementations.zweigitter import zweigitter_step
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
        dcc.Dropdown(['MGM', 'Vollst. MGM', 'ZGM'], 'MGM', id='methode-4-22'),
        "Gitter (l): ",
        dcc.Slider(2, max_l, 1, value=3, id="l-4-22"),
        "Dämpfung (w): ",
        dcc.Slider(0, 0.5, step=1e-6, marks={
            0.5: '1/2',
            1 / 3: '1/3',
            1 / 4: '1/4',
            1 / 8: '1/8'
        }, value=1 / 4, id="w-4-22", tooltip={"placement": "bottom"}),
        html.Div([
            dcc.Markdown("$$\\gamma$$:", mathjax=True),
            dcc.Slider(1, 3, step=1, value=1, id="gamma-4-22")
        ], id="gamma-div-4-22", hidden=True),
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
        dbc.Button(
            "Start Vektor",
            id="collapse-start-vector-button-4-22",
            color="info",
            outline=True,
            n_clicks=0,
        ),
        dbc.Collapse(
            dbc.Card(dbc.CardBody([], id="start_vector_div-4-22")),
            id="collapse-start-vector-4-22",
            is_open=False,
        ),

    ]),
    html.Br(),
    dcc.Graph(id='graph-4-22', mathjax=True),
])


@cache.memoize()
def f_x(x):
    return np.pi * np.pi / 8. * (9. * np.sin(3. * np.pi * x / 2.) + 25 * np.sin(5 * np.pi * x / 2.))


@cache.memoize()
def u_x(x):  # Lösung von f_x
    return np.sin(2. * np.pi * x) * np.cos(np.pi * x / 2.)


@cache.memoize()
def generate_figure(stufenindex_l, w, u_0: np.ndarray, mode, *, v1: int = 2, v2: int = 2, gamma: int = 1,
                    methode="MGM"):
    x = standard_schrittweitenfolge(stufenindex_l)
    f = f_x(x)
    u_star = u_x(x)
    matrices = jacobi_matrices if mode == "jacobi" else gauss_seidel_matrices
    fig = go.Figure(layout=go.Layout(
        yaxis={"title": "$$u$$"},
        xaxis={"title": "$$x$$"}))
    fig.add_trace(go.Scatter(x=x, y=u_0, name=f"$$u_0^{{{stufenindex_l}}}$$"))
    if methode == "ZGM":
        u_zgm = zweigitter_step(stufenindex_l, v1, v2, u_0, f, psi_vor_matrice=matrices,
                                w1=2 * w, w2=2 * w)
        fig.add_trace(go.Scatter(x=x, y=u_zgm, name=f"$$u^{{{stufenindex_l},ZGM}}$$"))
    else:
        u_mgm = mehrgitterverfahren_rekursiv(stufenindex_l, v1, v2, u_0, f, matrices, w1=2 * w, w2=2 * w, gamma=gamma)
        fig.add_trace(go.Scatter(x=x, y=u_mgm, name=f"$$u^{{{stufenindex_l},MGM}}$$"))
    fig.add_trace(go.Scatter(x=x, y=u_star, name=f"$$u^{{{stufenindex_l},\\ast}}$$"))
    return fig


@callback(Output('graph-4-22', 'figure'),
          Input('l-4-22', 'value'),
          Input('w-4-22', 'value'),
          Input({'type': 'start-vector-4-22', 'index': ALL}, 'children'),
          Input('tabs-jacobi-gaussseidel-switch', 'value'),
          Input('gamma-4-22', 'value'),
          Input('methode-4-22', 'value'))
def add_traces(stufenindex_l, w, vector, mode, gamma, methode):
    stufenindex_l_check(stufenindex_l, 2, max_l)
    w_check(w, -1e-6, 0.5)
    vec_check(vector)
    stufenindex_l_check(gamma, 1, 3)

    if dash.ctx.triggered_id is None or \
            (isinstance(dash.ctx.triggered_id, str) and dash.ctx.triggered_id.startswith("l-4-22")):
        vector = startfaults[stufenindex_l - 2]
        return generate_figure(stufenindex_l, w,
                               np.array(vector), mode, gamma=int(gamma), methode=methode)
    else:
        return generate_figure(stufenindex_l, w, np.array(vector).astype(np.float64), mode, gamma=int(gamma),
                               methode=methode)


@callback(Output('fault_vector_div-4-22', 'children'),
          Input('l-4-22', 'value'), )
def update_fault_vector_div(stufenindex_l):
    return default_array_div[stufenindex_l - 1]


@callback(Output('start_vector_div-4-22', 'children'),
          Input('l-4-22', 'value'),
          Input('w-4-22', 'value'),
          Input('methode-4-22', 'value'),
          State({'type': 'startfault-4-22', 'index': ALL}, 'value'),
          Input('submit-button-4-22', 'n_clicks'),
          Input('tabs-jacobi-gaussseidel-switch', 'value'))
@cache.memoize()
def update_start_vector_div(stufenindex_l, w, methode, vector, n_clicks, mode):
    vollstaendig = methode == "Vollst. MGM"
    stufenindex_l_check(stufenindex_l, 2, max_l)
    w_check(w, -1e-6, 0.5)
    vec_check(vector)

    x = standard_schrittweitenfolge(stufenindex_l)
    if vollstaendig:
        f = f_x(x)
        matrices = jacobi_matrices if mode == "jacobi" else gauss_seidel_matrices
        u_0 = get_start_vector(stufenindex_l, 2, f, matrices, 2 * w)
    else:
        u_star = u_x(x)
        if dash.ctx.triggered_id is None or dash.ctx.triggered_id.startswith("l-4-22"):
            vector = startfaults[stufenindex_l - 2]
        else:
            vector = np.array(vector).astype(np.float64)
        u_0 = u_star + vector
    list_of_elems = []
    for i in range(len(u_0)):
        list_of_elems.append(dbc.ListGroupItem(str(u_0[i]),
                                               id={'type': 'start-vector-4-22',
                                                   'index': i}))
    return dbc.ListGroup(list_of_elems)


# Clientside callbacks

dash.clientside_callback("function (value) {if (value) {return null} else {return 1e-6}}",
                         Output('w-4-22', 'step'),
                         Input('snapping', 'on'))

dash.clientside_callback("function (n, methode, is_open) {"
                         "const triggered = dash_clientside.callback_context.triggered.map(t => t.prop_id);"
                         "if (triggered.includes('methode-4-22.value')) {"
                         "return false;"
                         "}"
                         "if (methode == 'Vollst. MGM') {return false;} "
                         "if (n) {return !is_open;} else {return is_open;}}",
                         Output("collapse-4-22", "is_open"),
                         Input("collapse-button-4-22", "n_clicks"),
                         Input("methode-4-22", 'value'),
                         State("collapse-4-22", "is_open"))

dash.clientside_callback("function (n, is_open) {if (n) {return !is_open;} else {return is_open;}}",
                         Output("collapse-start-vector-4-22", "is_open"),
                         Input("collapse-start-vector-button-4-22", "n_clicks"),
                         State("collapse-start-vector-4-22", "is_open"))

dash.clientside_callback("function (methode) {return methode == 'Vollst. MGM';}",
                         Output("collapse-button-4-22", "disabled"),
                         Input('methode-4-22', 'value'))

dash.clientside_callback("function (methode) {return methode == 'ZGM';}",
                         Output("gamma-div-4-22", "hidden"),
                         Input('methode-4-22', 'value'))
