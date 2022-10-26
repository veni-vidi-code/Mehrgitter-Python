import dash_bootstrap_components as dbc
import dash_daq as daq
from dash import dcc, html, Output, Input, State
from os.path import exists

from pages.cache import cache

snipping_switch = dbc.Row(
    [dbc.Col("Easy Snapping", width="auto"),
     dbc.Col(daq.BooleanSwitch(
         on=True,
         id="snapping"
     ), width="auto")])

_tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'backgroundColor': '#abe2fb',
    "fontWeight": "bold"
}

jacobi_gausseidel_switch = html.Div(dcc.Tabs(id="tabs-jacobi-gausseidel-switch", value='jacobi', children=[
    dcc.Tab(label='Jacobi', value='jacobi', selected_style=_tab_selected_style),
    dcc.Tab(label='Gauss Seidel', value='gausseidel', selected_style=_tab_selected_style),
]), hidden=True, id="div-jacobi-gausseidel-switch")

footer = html.Footer(
    dbc.Container([
        dbc.Row([dbc.Col("© Tom Mucke", className="ml-auto pull-left"),
                 dbc.Col(dbc.Button("Info", n_clicks=0, color="info", id="btn-info"),
                         className="pull-right d-grid gap-2 col-6 mx-aut")],
                className="justify-content-between")],
        fluid=True),
    className="fixed-bottom mb-2")

canvas = dbc.Offcanvas("", id="offcanvas", is_open=False, title="")


def add_callbacks(app):
    @app.callback(Output('offcanvas', 'is_open'),
                  Input('btn-info', 'n_clicks'),
                  State('offcanvas', 'is_open'), prevent_initial_call=True)
    def toggle_offcanvas(n, is_open):
        return not is_open

    @app.callback(Output('div-jacobi-gausseidel-switch', 'hidden'),
                  Input('url', 'pathname'))
    def hide_jacobi_gausseidel_switch(pathname):
        if pathname in ["/", "/4/7", "/4/13", "/4/16"]:  # A few pages don't need the switch
            return True
        else:
            return False

    @app.callback(Output('offcanvas', 'title'),
                  Output('offcanvas', 'children'),
                  Input('url', 'pathname'))
    @cache.memoize()
    def update_offcanvas(pathname):
        # read markdown file from assets/mardownpagesexplanaition
        # ensures no relative paths are used
        pathname = pathname.replace(".", "")
        pathname = pathname.replace("~", "")

        # replaces / with _ to get the correct filename
        filename = pathname.replace("/", "_")[1:] + ".md"

        if filename == ".md":
            filename = "index.md"
        path = "assets/markdownpagesexplanation/" + filename
        if exists(path):
            with open(path, "r", encoding="utf-8") as f:
                markdown = f.read()
            # set title to first line of markdown file
            title = markdown.split("\n")[0]
            # remove first line of markdown file by slicing its amount of charcter
            markdown = markdown[len(title) + 1:]

            if title == "":
                title = "Info"
                for page in app.page_registry.values():
                    if page["path"] == pathname:
                        title = page["name"]
                        break

            return title, dcc.Markdown(markdown, mathjax=True)
        else:
            return "Info", dcc.Markdown("Zu dieser Seite gibt es keine Info")
