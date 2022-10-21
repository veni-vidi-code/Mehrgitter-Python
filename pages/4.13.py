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
    ]),
    html.Br(),
    dcc.Graph(id='fourier-modes'),
])


@callback(Output('fourier-modes', 'figure'), Input('l', 'value'), Input('j', 'value'))
@cache.memoize()
def change_gitter(stufenindex_l, j):
    y = mehrgitterhelper.fourier_mode(stufenindex_l, j)
    linear_gitter = LINEAR_GITTERHIERACHIE
    trivial_gitter = TRIVIAL_GITTERHIERACHIE
    x = linear_gitter.get_gitterfolge(stufenindex_l)
    x2 = linear_gitter.get_gitterfolge(stufenindex_l - 1)
    y_linear = linear_gitter.get_restriktionsmatrix(stufenindex_l) @ y
    y_trivial = trivial_gitter.get_restriktionsmatrix(stufenindex_l) @ y
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='original'))
    fig.add_trace(go.Scatter(x=x2, y=y_trivial, mode='lines+markers', name='trivial'))
    fig.add_trace(go.Scatter(x=x2, y=y_linear, mode='lines+markers', name='linear'))
    return fig


@callback(Output('j', 'max'), Output('j', 'value'), Input('l', 'value'), State('j', 'value'))
def change_j_max(stufenindex_l, j):
    n = (2 ** (stufenindex_l + 1)) - 1
    return n, min(n, j)
