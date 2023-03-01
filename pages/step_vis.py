import dash
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output

from Utils.components import stufenindex_l_check
from Utils.export import save_image
from implementations.step_visualization import mehrgitterverfahren_visualization, \
    vollstaendiges_mehrgitterverfahren_visualization
from pages.cache import cache

import dash_bootstrap_components as dbc
import dash_daq as daq

dash.register_page(__name__, name="Stufenentwicklung Mehrgitterverfahren", order=8)
"""
Visualisiert die Stufenentwicklung des Mehrgitterverfahrens sowie des vollständigen Mehrgitterverfahrens.
"""

max_l = 4
max_gamma = 3

layout = html.Div(children=[
    html.H1(children='Stufenentwicklung Mehrgitterverfahren'),
    html.Div([
        "Gitter (l): ",
        dcc.Slider(1, max_l, 1, value=2, id="l-s-vis"),
        dcc.Markdown("$$\\gamma$$:", mathjax=True),
        dcc.Slider(1, max_gamma, step=1, value=1, id="gamma-s-vis"),
        dbc.Row(
            [dbc.Col("vollständig", width="auto"),
             dbc.Col(daq.BooleanSwitch(
                 on=False,
                 id="s-vis-vst"
             ), width="auto")])
    ]),
    html.Br(),
    dcc.Graph(id='s-vis-graph', mathjax=True),
])


@cache.memoize()
def cached_mehrgitterverfahren_visualization(l, gamma):
    return mehrgitterverfahren_visualization(l, gamma, cached_mehrgitterverfahren_visualization)


@callback(Output('s-vis-graph', 'figure'),
          Input('l-s-vis', 'value'),
          Input('gamma-s-vis', 'value'),
          Input('s-vis-vst', 'on'))
@cache.memoize()
def update_figure(l, gamma, vst):
    stufenindex_l_check(l, 1, max_l)
    if vst:
        a, b = vollstaendiges_mehrgitterverfahren_visualization(l, gamma, cached_mehrgitterverfahren_visualization)
    else:
        a, b = cached_mehrgitterverfahren_visualization(l, gamma)
    fig = go.Figure()
    fig.update_yaxes(tickprefix="$$\\Omega_", ticksuffix="$$", tick0=0, dtick=1)
    fig.update_xaxes(visible=False)
    fig.add_trace(go.Scatter(x=list(range(len(a))), y=a, mode='lines+markers'))
    for i in range(len(b)):
        fig.add_annotation(x=i + 1 / 2, y=(a[i] + a[i + 1]) / 2, text=str(b[i]), showarrow=False, yshift=10)
    return fig


@callback(Output('s-vis-graph', 'className'), Input('s-vis-graph', 'figure'))
def save_figure(fig):
    return save_image(fig, "step_vis")
