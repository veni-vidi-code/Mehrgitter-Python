import pandas as pd
import dash
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

import plotly.express as px

from Utils.export import save_image
from implementations.helpers import N_l
from pages.cache import cache

dash.register_page(__name__, name="Laufzeitvergleich", order=7)
"""
Lädt Benchmarkergebnisse und stellt diese in einem Vergleich dar.
"""

# results from book (ISBN 978-3-658-07200-1)
book_results = {
    2: "117%",
    4: "838%",
    6: "9 255%",
    8: "128 161%"
}

# read results from file
raw_results = pd.read_json('Utils/benchmark-results.json', orient='records')

layout = html.Div(children=[
    html.H1(children='Laufzeitvergleich'),
    dbc.Container([
        dbc.Row([
            dbc.Col(dcc.Graph(id='table-laufzeitvergleich', mathjax=True, style={"height": "85vh"}), width=5),
            dbc.Col(dcc.Graph(id='bar-laufzeitvergleich', mathjax=True, style={"height": "85vh"}), width=7),
        ])], fluid=True)
])


@cache.memoize(timeout=900)
def get_results():
    results = raw_results[raw_results["w"] == 0.25]
    results = results.drop(columns=["w", "time_mgm", "time_normal"], axis=1)
    results["var_count"] = results["l"].apply(N_l)
    results["book_results"] = results["l"].map(book_results)
    results["book_results"] = results["book_results"].fillna("")
    results = results.rename(columns={"percentage": "%"})
    results["runs"] = results["runs"].apply(lambda x: f"{x:,}".replace(",", " "))
    results["classic"] = results["%"].apply(lambda x: f'{x:,.2f}%'.replace(",", " "))
    results["multigrid"] = "100%"
    results["Gitter"] = results["l"].apply(lambda x: f'$$\\Omega_{x}$$')
    return results[["l", "Gitter", "var_count", "runs", "multigrid", "classic", "book_results", "mode", "%"]]


@cache.memoize()
def jacobi_fig():
    results = get_results()
    jacobi = results.loc[results['mode'] == 'jacobi']
    jacobi = jacobi.drop(columns=["mode"], axis=1)
    jacobi = jacobi.sort_values(by=['l'])
    jacobi = jacobi.set_index('l')
    table_df = jacobi.drop(columns=["%"], axis=1)

    jacobi_table = go.Figure(go.Table(
        header=dict(values=["Gitter", "Anzahl der Unbekannten", "Stichprobe",
                            "MG Verfahren", "Jacobi Verfahren", "Buch"]),
        cells=dict(values=table_df.transpose().values.tolist(), align='right'),
    ))

    jacobi["Verfahren"] = "Jacobi"
    multigrid = jacobi.copy()
    multigrid["Verfahren"] = "Mehrgitter"
    multigrid["%"] = 100
    bookvalues = jacobi.loc[jacobi['book_results'] != ''].copy()
    bookvalues["Verfahren"] = "Buch"

    def modify_row(row):
        row["%"] = int(row["book_results"].rstrip("%").replace(" ", ""))
        return row

    bookvalues = bookvalues.apply(modify_row, axis=1)

    table_bar = pd.concat([multigrid, jacobi, bookvalues])
    table_bar["%_string"] = table_bar["%"].apply(lambda x: f'{x:,.2f}%')

    bar = px.bar(table_bar, x="Gitter", y="%", barmode="group", color="Verfahren", text="%_string")
    return jacobi_table, bar


@cache.memoize()
def gauss_seidel_fig():
    results = get_results()
    gauss_seidel = results.loc[results['mode'] == 'gauss_seidel']
    gauss_seidel = gauss_seidel.drop(columns=["mode"], axis=1)
    gauss_seidel = gauss_seidel.sort_values(by=['l'])
    gauss_seidel = gauss_seidel.set_index('l')
    table_df = gauss_seidel.drop(columns=["%", "book_results"], axis=1)

    gauss_seidel_table = go.Figure(go.Table(
        header=dict(values=["Gitter", "Anzahl der Unbekannten", "Stichprobe",
                            "MG Verfahren", "Gauss Seidel Verfahren"]),
        cells=dict(values=table_df.transpose().values.tolist(), align='right')))

    gauss_seidel["Verfahren"] = "Gauss Seidel"
    multigrid = gauss_seidel.copy()
    multigrid["Verfahren"] = "Mehrgitter"
    multigrid["%"] = 100
    table_bar = pd.concat([multigrid, gauss_seidel])
    table_bar["%_string"] = table_bar["%"].apply(lambda x: f'{x:,.2f}%')
    bar = px.bar(table_bar, x="Gitter", y="%", barmode="group", color="Verfahren", text="%_string", )
    bar.update_layout(autosize=True)
    return gauss_seidel_table, bar


@callback(Output('table-laufzeitvergleich', 'figure'),
          Output('bar-laufzeitvergleich', 'figure'),
          Input('tabs-jacobi-gaussseidel-switch', 'value'))
def update_table_bar(value):
    if value == "jacobi":
        return jacobi_fig()
    else:
        return gauss_seidel_fig()


@callback(Output('bar-laufzeitvergleich', 'className'), Input('bar-laufzeitvergleich', 'figure'))
def save_figure(fig):
    return save_image(fig, "bar-laufzeitvergleich")
