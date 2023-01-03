from typing import TypedDict

import dash
import dash_bootstrap_components as dbc
import dash_daq as daq

import os

from dash import html, dcc

from pages.cache import cache

dash.register_page(__name__, name="Mehr zum Thema", order=9)


class ReadMore(TypedDict):
    file: str
    title: str
    url: str
    description: str


@cache.memoize(timeout=600)
def layout():
    read_more: list[ReadMore] = []

    dir = "assets/read_more"
    for file in os.listdir(dir):
        path = os.path.join(dir, file)

        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                title = f.readline()
                url = f.readline()
                description = f.read()

                read_more.append(ReadMore(file=file, title=title, url=url, description=description))

    return html.Div(children=[
        html.H1(children='Mehr zum Thema'),
        dbc.Accordion([dbc.AccordionItem([dcc.Markdown(item["description"], mathjax=True),
                                          dbc.Button(item["title"], href=item["url"], color="info", target="_blank")],
                                         title=item["title"]) for item in read_more])])
