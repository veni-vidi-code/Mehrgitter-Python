import os
import time

import dash
from plotly.graph_objects import Figure
import plotly.io as pio

asset_image_path = 'assets/images'

if not os.path.exists(asset_image_path):
    os.mkdir(asset_image_path)
img_format = 'pdf'
pio.kaleido.scope.default_format = img_format
pio.kaleido.scope.default_width = 425
pio.kaleido.scope.default_scale = 1


def save_image(figure, figname, **kwargs):
    if not isinstance(figure, Figure):
        figure = Figure(figure)
    folder = os.path.join(asset_image_path, figname)
    if not os.path.exists(folder):
        os.mkdir(folder)
    realname = f'{time.time()}.{img_format}'
    print(f'Saving {realname} to {folder}')
    figure.write_image(os.path.join(folder, realname), engine='kaleido', **kwargs)
    print(f'Saved {realname} to {folder}')
    return dash.no_update
