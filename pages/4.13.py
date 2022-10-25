import dash
from dash import html, dcc, callback, Input, Output, State
from implementations import dirichlect
import plotly.graph_objects as go

from implementations.Gitter import LINEAR_GITTERHIERACHIE, TRIVIAL_GITTERHIERACHIE
from implementations.dirichlect import N_l
from pages.cache import cache
import dash_bootstrap_components as dbc
import dash_daq as daq

dash.register_page(__name__, name="Fourier Moden")

layout = html.Div(children=[
    html.H1(children='Fourier Moden'),
    html.Div([
        "Gitter (l): ",
        dcc.Slider(1, 10, 1, value=3, id="l"),
        "Wellenzahl (j): ",
        dcc.Slider(1, 20, 1, value=1, id="j"),
        dbc.Row(
            [dbc.Col("Skalieren (für Prolongation empfohlen!)", width="auto"),
             dbc.Col(daq.BooleanSwitch(
                 on=True,
                 id="skalieren"
             ), width="auto")]),
        "Richtung: ",
        dcc.Dropdown(
            options=[
                {'label': 'Restriktion', 'value': 'r'},
                {'label': 'Prolongation', 'value': 'p'}
            ],
            value='r',
            id="richtung"
        ),
    ]),
    html.Br(),
    dcc.Graph(id='fourier-modes'),
])


@callback(Output('fourier-modes', 'figure'), Input('l', 'value'), Input('j', 'value'), Input('skalieren', 'on'),
          Input('richtung', 'value'))
@cache.memoize()
def change_gitter(stufenindex_l, j, scale, direction):
    fig = go.Figure()
    linear_gitter = LINEAR_GITTERHIERACHIE
    x = linear_gitter.get_gitterfolge(stufenindex_l)
    e_l_j = dirichlect.fourier_mode(stufenindex_l, j, scale)
    fig.add_trace(go.Scatter(x=x, y=e_l_j, mode='lines+markers', name='e_l,j (original)'))
    trivial_gitter = TRIVIAL_GITTERHIERACHIE
    if direction == 'r':
        x2 = linear_gitter.get_gitterfolge(stufenindex_l - 1)
        y_linear = linear_gitter.get_restriktionsmatrix(stufenindex_l) @ e_l_j
        y_trivial = trivial_gitter.get_restriktionsmatrix(stufenindex_l) @ e_l_j
        fig.add_trace(go.Scatter(x=x2, y=y_trivial, mode='lines+markers', name='trivial'))
        fig.add_trace(go.Scatter(x=x2, y=y_linear, mode='lines+markers', name='linear'))
    else:
        e_l_minus_1_j = dirichlect.fourier_mode(stufenindex_l - 1, j, scale)
        y_linear = linear_gitter.get_prolongationsmatrix(stufenindex_l) @ e_l_minus_1_j
        fig.add_trace(go.Scatter(x=x, y=y_linear, mode='lines+markers', name='P*e_l-1,j'))

    return fig


@callback(Output('j', 'max'), Output('j', 'value'), Output('j', "tooltip"), Output('j', "marks"),
          Input('l', 'value'), State('j', 'value'), Input('richtung', 'value'), Input('tabs-jacobi-gausseidel-switch', 'value'))
def change_j_max(stufenindex_l, j, direction, mode):
    if direction == 'r':
        n = N_l(stufenindex_l)
    else:
        n = (2 ** stufenindex_l) - 1
    if stufenindex_l > 5:
        return n, min(n, j), {"placement": "bottom", "always_visible": True}, None
    else:
        return n, min(n, j), None, {}
