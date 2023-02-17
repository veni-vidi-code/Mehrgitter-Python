import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output, State, ctx

from Utils.components import snipping_switch, w_check, stufenindex_l_check, fig_check
from implementations import dirichlect_ndarrays
from implementations.helpers import N_l

dash.register_page(__name__, name="Eigenwerte", order=1)
"""
Eigenwerte der Iterationsmatrix des gedämpften Jacobi-Verfahrens für
die Dirichlet-Randbedingung
"""

layout = html.Div(children=[
    html.H1(children='Eigenwerte Iterationsmatrix'),
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
    dcc.Graph(id='eigenvalues', mathjax=True)
])


def _add_eigenvalues_trace(stufenindex_l, w, fig):
    x = list(range(1, N_l(stufenindex_l) + 1))
    y = [dirichlect_ndarrays.eigenvalues(stufenindex_l, i, w) for i in x]
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=f'w={w}'))


@callback(Output('eigenvalues', 'figure'), Output('w', 'value'),
          Input('submit-button', 'n_clicks'), Input('l', 'value'),
          State('w', 'value'), State('eigenvalues', 'figure'))
def add_eigenvalues(n_clicks, stufenindex_l, w, fig):
    w_check(w, -1e-6, 1)
    stufenindex_l_check(stufenindex_l, 1, 5)
    fig_check(fig)

    if ctx.triggered_id is not None and ctx.triggered_id.startswith('submit-button'):
        fig = go.Figure(fig)
        _add_eigenvalues_trace(stufenindex_l, w, fig)
        return fig, w
    else:
        fig = go.Figure(layout=go.Layout(
            yaxis={"title": f'$${{\\lambda}}^{{{stufenindex_l},j}}(w)$$'},
            xaxis={"title": "$$j$$"}), )
        _add_eigenvalues_trace(stufenindex_l, 0.5, fig)
        return fig, 0.5


# Clientside callbacks

dash.clientside_callback("function (value) {if (value) {return null} else {return 1e-6}}",
                         Output('w', 'step'),
                         Input('snapping', 'on'))
