import flask
from dash import Dash, html, dcc, Output, Input
import dash

from Utils.components import jacobi_gausseidel_switch
from pages.cache import cache
import dash_bootstrap_components as dbc

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
)

dashapp.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dbc.Container([
        navbar,
        jacobi_gausseidel_switch,
        dash.page_container], fluid=True),
])


@dashapp.callback(Output('div-jacobi-gausseidel-switch', 'hidden'),
                  Input('url', 'pathname'))
def hide_jacobi_gausseidel_switch(pathname):
    if pathname in ["/"]:  # A few pages don't need the switch
        return True
    else:
        return False


if __name__ == '__main__':
    dashapp.run(debug=True)
