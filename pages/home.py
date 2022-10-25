import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/')

layout = html.Div(children=[html.Br(), html.Div([dbc.Button(page["name"], color="info", href=page["path"])
                                                 for page in dash.page_registry.values()
                                                 if page["module"] != "pages.not_found_404" and page["path"] != "/"],
                                                className="d-grid gap-2 col-6 mx-auto align-items-center")])
