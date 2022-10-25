import dash_daq as daq
import dash_bootstrap_components as dbc
from dash import dcc

snipping_switch = dbc.Row(
    [dbc.Col("Easy Snapping", width="auto"),
     dbc.Col(daq.BooleanSwitch(
         on=True,
         id="snapping"
     ), width="auto")])

_tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'backgroundColor': '#abe2fb',
    "font-weight": "bold"
}

jacobi_gausseidel_switch = dcc.Tabs(id="tabs-jacobi-gausseidel-switch", value='jacobi', children=[
    dcc.Tab(label='Jacobi', value='jacobi', selected_style=_tab_selected_style),
    dcc.Tab(label='Gauss Seidel', value='gausseidel', selected_style=_tab_selected_style),
])
