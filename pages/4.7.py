import dash
from dash import html, dcc, callback, Input, Output, State, ctx

from Utils.components import snipping_switch
from implementations import dirichlect
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from implementations.dirichlect import N_l

dash.register_page(__name__, name="Eigenwerte", order=1)

layout = html.Div(children=[
    html.H1(children='Eigenwerte Fourier Moden'),
    html.Div([
        snipping_switch,
        "Gitter (l): ",
        dcc.Slider(1, 5, 1, value=3, id="l"),
        "Dämpfung (w): ",
        dcc.Slider(0, 1, step=1e-6, marks={
            1: '1',
            0.5: '1/2',
            1 / 3: '1/3',
            1 / 4: '1/4',
            1 / 8: '1/8',
            0: '0'
        }, value=0.5, id="w", tooltip={"placement": "bottom"}),
        dbc.Button(
            "Hinzufügen",
            id="submit-button",
            color="info",
            outline=True,
            n_clicks=0,
        ),
    ]),
    html.Br(),
    dcc.Graph(id='eigenvalues'),
])


def _add_eigenvalues_trace(stufenindex_l, w, fig):
    x = list(range(1, N_l(stufenindex_l) + 1))
    y = [dirichlect.eigenvalues(stufenindex_l, i, w) for i in x]
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=f'w={w}'))


@callback(Output('eigenvalues', 'figure'), Output('w', 'value'),
          Input('submit-button', 'n_clicks'), Input('l', 'value'),
          State('w', 'value'), State('eigenvalues', 'figure'), Input('tabs-jacobi-gausseidel-switch', 'value'))
def add_eigenvalues(n_clicks, stufenindex_l, w, fig, mode):
    if ctx.triggered_id is not None and ctx.triggered_id.startswith('submit-button'):
        fig = go.Figure(fig)
        _add_eigenvalues_trace(stufenindex_l, w, fig)
        return fig, w
    else:
        fig = go.Figure()
        _add_eigenvalues_trace(stufenindex_l, 0.5, fig)
        return fig, 0.5


@callback(Output('w', 'step'), Input('snapping', 'on'))
def snapping(value):
    if value:
        return None
    else:
        return 1e-6
