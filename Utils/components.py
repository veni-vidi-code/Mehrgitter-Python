import dash_bootstrap_components as dbc
import dash_daq as daq
from dash import dcc, html, Output, Input, State
from os.path import exists

from dash.exceptions import PreventUpdate

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

jacobi_gaussseidel_switch = html.Div(dcc.Tabs(id="tabs-jacobi-gaussseidel-switch", value='jacobi', children=[
    dcc.Tab(label='Jacobi', value='jacobi', selected_style=_tab_selected_style),
    dcc.Tab(label='Gauss Seidel', value='gaussseidel', selected_style=_tab_selected_style),
]), hidden=True, id="div-jacobi-gaussseidel-switch")

footer = html.Footer(
    dbc.Container([
        dbc.Row([dbc.Col(
            dbc.Button("Github", color="info", outline=True, href="https://github.com/veni-vidi-code/Mehrgitter-Python",
                       target="_blank"),
            className="ml-auto pull-left"),
            dbc.Col(dbc.Button("Info", n_clicks=0, color="info", id="btn-info", outline=True),
                    className="pull-right d-grid gap-2 col-6 mx-aut")],
            className="justify-content-between")],
        fluid=True),
    className="fixed-bottom mb-2")

canvas = dbc.Offcanvas("", id="offcanvas", is_open=False, title="")


def add_callbacks(app):
    app.clientside_callback("function(n, is_open) {return !is_open;}",
                            Output('offcanvas', 'is_open'),
                            Input('btn-info', 'n_clicks'),
                            State('offcanvas', 'is_open'), prevent_initial_call=True)

    app.clientside_callback(
        'function(pathname) {return ["/", "/4/7", "/4/13", "/4/16", "/step-vis", "/read-more"].includes(pathname);}',
        Output('div-jacobi-gaussseidel-switch', 'hidden'),
        Input('url', 'pathname'))

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


def w_check(w, infimum=0, maximum=1):
    if not ((isinstance(w, float) or isinstance(w, int)) and infimum < w <= maximum):
        print("w", w)
        raise PreventUpdate


def stufenindex_l_check(stufenindex_l, minimum=0, maximum=10):
    if not (isinstance(stufenindex_l, int) and minimum <= stufenindex_l <= maximum):
        print("stufenindex_l", stufenindex_l)
        raise PreventUpdate


def fig_check(fig):
    if fig is not None and not (isinstance(fig, dict) and 'data' in fig and 'layout' in fig):
        print("fig", fig)
        raise PreventUpdate


def vec_check(vector):
    if not (isinstance(vector, list)):
        print("vector", vector)
        raise PreventUpdate
