import dash
from dash import html, dcc, callback, Input, Output, State
from implementations import mehrgitterhelper
import plotly.graph_objects as go

from implementations.Gitter import LINEAR_GITTERHIERACHIE, TRIVIAL_GITTERHIERACHIE
from pages.cache import cache

dash.register_page(__name__, name="Fourier Moden")

layout = html.Div(children=[
    html.H1(children='Fourier Moden'),
    html.Div([
        "Gitter (l): ",
        dcc.Slider(1, 5, 1, value=3, id="l"),
        "Wellenzahl (j): ",
        dcc.Slider(1, 20, 1, value=1, id="j"),
        "Skalieren (für Prolongation empfohlen!)",
        dcc.Checklist(
            options=[
                {'label': 'Skalieren', 'value': 's'}
            ],
            value=[],
            id="skalieren"
        ),
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


@callback(Output('fourier-modes', 'figure'), Input('l', 'value'), Input('j', 'value'), Input('skalieren', 'value'),
          Input('richtung', 'value'))
@cache.memoize()
def change_gitter(stufenindex_l, j, scale, direction):
    if scale == ['s']:
        scale = True
    else:
        scale = False
    fig = go.Figure()
    linear_gitter = LINEAR_GITTERHIERACHIE
    x = linear_gitter.get_gitterfolge(stufenindex_l)
    e_l_j = mehrgitterhelper.fourier_mode(stufenindex_l, j, scale)
    fig.add_trace(go.Scatter(x=x, y=e_l_j, mode='lines+markers', name='e_l,j (original)'))
    trivial_gitter = TRIVIAL_GITTERHIERACHIE
    if direction == 'r':
        x2 = linear_gitter.get_gitterfolge(stufenindex_l - 1)
        y_linear = linear_gitter.get_restriktionsmatrix(stufenindex_l) @ e_l_j
        y_trivial = trivial_gitter.get_restriktionsmatrix(stufenindex_l) @ e_l_j
        fig.add_trace(go.Scatter(x=x2, y=y_trivial, mode='lines+markers', name='trivial'))
        fig.add_trace(go.Scatter(x=x2, y=y_linear, mode='lines+markers', name='linear'))
    else:
        e_l_minus_1_j = mehrgitterhelper.fourier_mode(stufenindex_l - 1, j, scale)
        y_linear = linear_gitter.get_prolongationsmatrix(stufenindex_l) @ e_l_minus_1_j
        fig.add_trace(go.Scatter(x=x, y=y_linear, mode='lines+markers', name='P*e_l-1,j'))

    return fig


@callback(Output('j', 'max'), Output('j', 'value'), Input('l', 'value'), State('j', 'value'),
          Input('richtung', 'value'))
def change_j_max(stufenindex_l, j, direction):
    if direction == 'r':
        n = (2 ** (stufenindex_l + 1)) - 1
    else:
        n = (2 ** stufenindex_l) - 1
    return n, min(n, j)


