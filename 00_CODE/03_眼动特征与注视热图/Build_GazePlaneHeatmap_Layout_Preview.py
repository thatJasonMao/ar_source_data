"""Build a self-contained offline editor for the balanced heatmap figure."""
import hashlib
import io
import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import Plot_GazePlaneHeatmap_Word_Balanced as balanced

background=balanced.background


def panel_image(scenario, density, config, outlines):
    """Render scientific layers in Python; the editor supplies adjustable text."""
    xr=config['coordinate_system']['x_range_m']
    yr=config['coordinate_system']['vertical_range_m']
    fig=plt.figure(figsize=(3,3*(yr[1]-yr[0])/(xr[1]-xr[0])),dpi=200)
    ax=fig.add_axes([0,0,1,1])
    background.draw_panel(ax,scenario,density,config)
    for text in list(ax.texts):
        text.remove()
    for line in list(ax.lines):
        line.remove()
    ax.grid(False)
    ax.set_axis_off()
    fig.canvas.draw()
    background.draw_outlines(ax,scenario,config,outlines)
    stream=io.BytesIO()
    fig.savefig(stream,format='png',dpi=200,facecolor='white')
    plt.close(fig)
    stream.seek(0)
    with Image.open(stream) as image:
        return background.data_url(image)


def main():
    config=background.read_json(background.ASSETS/'plot_config.json')
    outlines=background.read_json(background.asset_path(
        config['rendering']['arrow_outline']['contours_file']))['groups']
    background.validate(config,outlines)
    densities=background.densities_from_preview()
    cmap=background.original.get_custom_cmap()
    payload={
        'schema':'gaze-plane-balanced-layout','schema_version':1,
        'defaults':balanced.DEFAULT_LAYOUT,
        'background_config_sha256':hashlib.sha256(
            json.dumps(config,sort_keys=True).encode()).hexdigest(),
        'x_range_m':config['coordinate_system']['x_range_m'],
        'vertical_range_m':config['coordinate_system']['vertical_range_m'],
        'names':background.NAMES,'scenarios':{},
        'colorbar':background.data_url(Image.fromarray(
            cmap(np.linspace(0,1,512)[None,:],bytes=True))),
    }
    entropy=background.original.load_entropy_map()
    for scenario,(density,fingerprint,_) in densities.items():
        payload['scenarios'][scenario]={
            'image':panel_image(scenario,density,config,outlines),
            'peak_m':background.peak(density),'entropy_bit':entropy.get(scenario),
            'fingerprint':fingerprint,
        }
        print('Embedded '+scenario,flush=True)
    template=(background.ASSETS/'balanced_layout_preview_template.html').read_text(encoding='utf-8')
    output=background.OUTPUT/'折中版热图_字体与间距调整.html'
    output.write_text(template.replace('__PAYLOAD__',
        json.dumps(payload,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
    print('Saved '+str(output),flush=True)


if __name__=='__main__':
    main()
