import flask
from dash import Dash, html, dcc
import dash
#import dash_auth

allowed_users = {
    'numaseminar': 'MehrGitterVerfahren2022'
}

server = flask.Flask(__name__)
app = Dash(__name__, use_pages=True, server=server)

"""auth = dash_auth.BasicAuth(
    app,
    allowed_users
)"""

app.layout = html.Div([
    html.H1('Demo Mehrkörperverfahren'),

    html.Div(
        [
            html.Div(
                dcc.Link(
                    f"{page['name']}", href=page["relative_path"]
                )
            )
            for page in dash.page_registry.values()
        ]
    ),

    dash.page_container
])

if __name__ == '__main__':
    app.run_server(debug=True)
