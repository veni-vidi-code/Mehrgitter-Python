import flask
from dash import Dash, html, dcc, Output, Input
import dash
from pages.cache import cache


server = flask.Flask(__name__)
app = server
dashapp = Dash(__name__, use_pages=True, server=server, serve_locally=False)
cache.init_app(server)


dashapp.layout = html.Div([
    dcc.Location(id='url', refresh=True),
    html.H1('Demo Mehrkörperverfahren'),

    html.Div(
        [
            dcc.Dropdown(id='page_dd',
                         options=[{'label': page['name'], 'value': page["relative_path"]}
                                  for page in dash.page_registry.values() if page["relative_path"] != "/"]),
        ]
    ),

    dash.page_container
])


@dashapp.callback(Output("url", "pathname"), Input("page_dd", "value"))
def update_url_on_dropdown_change(dropdown_value):
    return dropdown_value


if __name__ == '__main__':
    dashapp.run(debug=True)
