import dash
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output, State

import implementations.dirichlect_ndarrays as dirichlect
from implementations.helpers import N_l
from implementations.ggk import ggk_Psi_l
from pages.cache import cache

dash.register_page(__name__, name="Fourier Moden mit Grobgitterkorrektur", order=5)

layout = html.Div(children=[
    html.H1(children='Fourier Moden GGK'),
    html.Div([
        "Gitter (l): ",
        dcc.Slider(2, 10, 1, value=3, id="l-4-16"),
        "Wellenzahl (j): ",
        dcc.Slider(1, 20, 1, value=1, id="j-4-16")
    ]),
    html.Br(),
    dcc.Graph(id='fourier-modes-4-16', mathjax=True),
])


@callback(Output('fourier-modes-4-16', 'figure'), Input('l-4-16', 'value'), Input('j-4-16', 'value'))
@cache.memoize()
def change_gitter(stufenindex_l, j):
    # Dies würde sich auch effizienter mit Satz 4.53 berechnen lassen, aber zur Demonstration reicht das hier.
    fig = go.Figure(layout=go.Layout(xaxis={"title": "$$k$$"}))
    e_l_j = dirichlect.fourier_mode(stufenindex_l, j, False)
    y = ggk_Psi_l(stufenindex_l, e_l_j)
    x = list(range(1, N_l(stufenindex_l) + 1))
    fig.add_trace(go.Scatter(x=x, y=e_l_j, mode='lines+markers',
                             name=f"$$e_{{k}}^{{{stufenindex_l},{j}}}$$"))
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers',
                             name=f"$$(\\Psi_{stufenindex_l}^{{GGK}}(e^{{{stufenindex_l},{j}}}))_{{k}}$$"))
    return fig


@callback(Output('j-4-16', 'max'), Output('j-4-16', 'value'), Output('j-4-16', "tooltip"), Output('j-4-16', "marks"),
          Input('l-4-16', 'value'), State('j-4-16', 'value'))
def change_j_max(stufenindex_l, j):
    n = N_l(stufenindex_l)

    if stufenindex_l > 5:
        return n, min(n, j), {"placement": "bottom", "always_visible": True}, None
    else:
        return n, min(n, j), None, {}
