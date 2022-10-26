import dash
import dash_bootstrap_components as dbc
import dash_daq as daq
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output, State

from implementations import dirichlect_ndarrays
from implementations.Gitter import LINEAR_GITTERHIERACHIE, TRIVIAL_GITTERHIERACHIE
from implementations.helpers import N_l
from pages.cache import cache

dash.register_page(__name__, name="Restriktion/Prolongation", order=4)

layout = html.Div(children=[
    html.H1(children='Restriktion/Prolongation Fourier Moden'),
    html.Div([
        "Gitter (l): ",
        dcc.Slider(1, 10, 1, value=3, id="l"),
        "Wellenzahl (j): ",
        dcc.Slider(1, 20, 1, value=1, id="j"),
        dbc.Row(
            [dbc.Col("Skalieren (für Prolongation empfohlen!)", width="auto"),
             dbc.Col(daq.BooleanSwitch(
                 on=False,
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
    dcc.Graph(id='fourier-modes', mathjax=True)
])


@callback(Output('fourier-modes', 'figure'), Input('l', 'value'), Input('j', 'value'), Input('skalieren', 'on'),
          Input('richtung', 'value'))
@cache.memoize()
def change_gitter(stufenindex_l, j, scale, direction):
    fig = go.Figure(layout=go.Layout(xaxis={"title": "$$k$$"}))
    linear_gitter = LINEAR_GITTERHIERACHIE
    e_l_j = dirichlect_ndarrays.fourier_mode(stufenindex_l, j, scale)
    fig.add_trace(go.Scatter(x=list(range(1, N_l(stufenindex_l) + 1)), y=e_l_j, mode='lines+markers',
                             name=f"$$e_{{k}}^{{{stufenindex_l},{j}}}$$ (original)"))
    trivial_gitter = TRIVIAL_GITTERHIERACHIE
    if direction == 'r':
        x2 = list(range(2, N_l(stufenindex_l), 2))
        y_linear = linear_gitter.get_restriktionsmatrix(stufenindex_l) @ e_l_j
        y_trivial = trivial_gitter.get_restriktionsmatrix(stufenindex_l) @ e_l_j
        fig.add_trace(go.Scatter(x=x2, y=y_trivial, mode='lines+markers',
                                 name=f"$$(R_{stufenindex_l}^{stufenindex_l - 1}"
                                      f"e^{{{stufenindex_l},{j}}})_{{2k}} (triveal)$$"))
        fig.add_trace(go.Scatter(x=x2, y=y_linear, mode='lines+markers',
                                 name=f"$$(R_{stufenindex_l}^{stufenindex_l - 1}"
                                      f"e^{{{stufenindex_l},{j}}})_{{2k}} (linear)$$"))
    else:
        e_l_minus_1_j = dirichlect_ndarrays.fourier_mode(stufenindex_l - 1, j, scale)
        y_linear = linear_gitter.get_prolongationsmatrix(stufenindex_l) @ e_l_minus_1_j
        fig.add_trace(go.Scatter(x=list(range(1, N_l(stufenindex_l) + 1)), y=y_linear, mode='lines+markers',
                                 name=f"$$(P_{stufenindex_l - 1}^{stufenindex_l}"
                                      f"e^{{{stufenindex_l - 1},{j}}})_{{k}} (linear)$$"))

    return fig


@callback(Output('j', 'max'), Output('j', 'value'), Output('j', "tooltip"), Output('j', "marks"),
          Input('l', 'value'), State('j', 'value'), Input('richtung', 'value'))
def change_j_max(stufenindex_l, j, direction):
    if direction == 'r':
        n = N_l(stufenindex_l)
    else:
        n = N_l(stufenindex_l - 1)
    if stufenindex_l > 5:
        return n, min(n, j), {"placement": "bottom", "always_visible": True}, None
    else:
        return n, min(n, j), None, {}
