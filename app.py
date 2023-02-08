"""
Diese Datei dient als Einstiegspunkt für die Anwendung.

Hier werden die einzelnen Seiten registriert und die Dash-App
initialisiert. Wird die Datei direkt ausgeführt, wird die App
im Debug-Modus gestartet.
"""

import dash
import dash_bootstrap_components as dbc
import flask
from dash import Dash, html, dcc

from Utils.components import jacobi_gaussseidel_switch, footer, canvas, add_callbacks
from pages.cache import cache

server = flask.Flask(__name__)
app = server
dashapp = Dash(__name__, use_pages=True, server=server, serve_locally=False,
               external_stylesheets=[dbc.themes.BOOTSTRAP])
cache.init_app(server)

navbar = dbc.NavbarSimple(
    dbc.DropdownMenu(
        [
            dbc.DropdownMenuItem(page["name"], href=page["path"])
            for page in dash.page_registry.values()
            if page["module"] != "pages.not_found_404"
        ],
        nav=True,
        in_navbar=True,
        label="Seite wechseln",
        direction="start",
    ),
    brand="Demo Mehrkörperverfahren",
    color="#abccfb",
    className="main-navbar",
    fluid=True,
    brand_href="/",
)

dashapp.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dbc.Container([
        navbar,
        jacobi_gaussseidel_switch,
        dash.page_container,
        footer,
        canvas],
        fluid=True),
])

add_callbacks(dashapp)

if __name__ == '__main__':
    dashapp.run(debug=True)
