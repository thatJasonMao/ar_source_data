"""Approved background/arrow-outline variant of Extract_GazePlaneHeatmap.PY.

Run normally to produce ten PNGs, a 5-by-2 overview and the offline HTML preview.
--config accepts JSON exported by the preview. --preview-only updates only HTML.
--combined-only builds the overview from the existing preview's density caches.
The original source is imported read-only. This variant deliberately corrects
the projection origin to the Camera at local (0, 1.771, 0.2) m and projects to
the existing Canvas plane at Agent-root local z=0.9 m, giving a 0.7 m forward
camera-to-plane separation. KDE retains the original display/data ranges.
"""
import argparse
import base64
import hashlib
import io
import json
import re
from copy import deepcopy
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import font_manager
from matplotlib.colors import ListedColormap, Normalize, is_color_like
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'Extract_GazePlaneHeatmap.py'
OUTPUT = ROOT / 'GAZE_TOPO' / 'Results_GazePlaneHeatmap'
ASSETS = OUTPUT / 'background_alignment'
NAMES = {'StaticSign':'StaticSign', 'ArrowRibbon':'Arrow', 'MiniMap':'MiniMap',
         'TextBar':'TextBar', 'Full':'Multimodal'}
COMBINED_ROWS = ('StaticSign', 'ArrowRibbon', 'MiniMap', 'TextBar', 'Full')
COMBINED_COLUMNS = ('LowSmoke', 'HighSmoke')
COLS = ['SubjectWorldX','SubjectWorldY','SubjectWorldZ','SubjectRotY_deg',
        'GazeWorldX','GazeWorldY','GazeWorldZ']
loader = SourceFileLoader('original_gaze_plane', str(SOURCE))
spec = spec_from_loader(loader.name, loader)
original = module_from_spec(spec)
loader.exec_module(original)

# Keep the imported upstream scripts unchanged. The corrected camera-relative
# projection is implemented locally in this approved background variant.
CAMERA_LOCAL_Y_M = 1.771
CAMERA_LOCAL_Z_M = 0.2
CAMERA_TO_PLANE_DISTANCE_M = 0.7
CANVAS_PLANE_LOCAL_Z_M = CAMERA_LOCAL_Z_M + CAMERA_TO_PLANE_DISTANCE_M


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def asset_path(value):
    path = Path(value)
    return path if path.is_absolute() else OUTPUT / path


def group_for(scenario,config):
    key=scenario.split('-')[0]
    return {**config['groups'][key],**config.get('scenario_overrides',{}).get(scenario,{})}


def outline_for(scenario,outlines):
    return outlines.get(scenario,outlines.get(scenario.split('-')[0],{}))


def extent(group, config):
    ref = config['scale_reference']
    width = ref['width_m'] * group['scale_x']
    height = ref['height_m'] * group['scale_y']
    x, y = group['center_x_m'], group['center_vertical_m']
    return [x-width/2, x+width/2, y-height/2, y+height/2]


def validate(config, outlines):
    if config.get('schema') != 'gaze-plane-background-alignment':
        raise ValueError('Unrecognized alignment configuration')
    system = config['coordinate_system']
    render = config['rendering']
    for bounds in [system['x_range_m'], system['vertical_range_m']]:
        if len(bounds)!=2 or not np.isfinite(bounds).all() or bounds[0]>=bounds[1]:
            raise ValueError('Invalid display range')
    if (not np.allclose(render['density_x_range_m'], original.X_RANGE)
            or not np.allclose(render['density_vertical_range_m'], original.Z_RANGE)
            or not np.isclose(system['eye_height_m'], CAMERA_LOCAL_Y_M)
            or not np.isclose(system['plane_distance_m'], CAMERA_TO_PLANE_DISTANCE_M)
            or not np.isclose(
                CAMERA_LOCAL_Z_M + system['plane_distance_m'],
                CANVAS_PLANE_LOCAL_Z_M,
            )):
        raise ValueError('Corrected camera-relative coordinate system changed; review configuration')
    scenarios=[f'{key}-{smoke}' for key in NAMES for smoke in COMBINED_COLUMNS]
    unknown=set(config.get('scenario_overrides',{}))-set(scenarios)
    if unknown:
        raise ValueError('Unknown smoke-specific backgrounds: '+repr(unknown))
    for scenario in scenarios:
        key=scenario.split('-')[0]
        group = group_for(scenario,config)
        if group['scale_x']<=0 or group['scale_y']<=0:
            raise ValueError(f'Invalid scale: {key}')
        if not np.isfinite(extent(group, config)).all():
            raise ValueError(f'Non-finite image extent: {key}')
        for field in ['background_alpha','heatmap_alpha_multiplier']:
            if not 0 <= group[field] <= 1:
                raise ValueError(f'Invalid opacity: {key}')
        image_path = asset_path(group['image_path'])
        with Image.open(image_path) as image:
            if list(image.size) != group['image_size_px']:
                raise ValueError(f'Image dimensions changed: {image_path}')
        contour_group=outline_for(scenario,outlines)
        if contour_group:
            if hashlib.sha256(image_path.read_bytes()).hexdigest()!=contour_group['image_sha256']:
                raise ValueError(f'Background changed; retrace contours: {image_path}')
            for contour in contour_group['paths']:
                pts=np.asarray(contour['points'])
                if pts.ndim!=2 or pts.shape[1]!=2 or len(pts)<2 or not np.isfinite(pts).all():
                    raise ValueError(f'Invalid contour: {key}')
    style = render['arrow_outline']
    if style['width_screen_px']<=0 or style['halo_extra_px']<0:
        raise ValueError('Invalid outline width')
    for color in [style['color'],style['halo_color'],*render.get('ui_region_colors',{}).values()]:
        if not is_color_like(color):
            raise ValueError(f'Invalid color: {color}')


def project_to_camera_plane(points_3d):
    """Project root-relative hit points onto the Canvas plane from Camera.

    ``points_3d`` follows the imported convention ``(local_x, local_z,
    local_y)`` and is relative to the Agent root. Camera local position is
    ``(0, 0.2, 1.771)`` in that ordering. The target plane remains at Agent
    local z=0.9, equivalently 0.7 m in front of Camera.
    """
    vx = points_3d[:, 0]
    vz = points_3d[:, 1] - CAMERA_LOCAL_Z_M
    vy = points_3d[:, 2] - CAMERA_LOCAL_Y_M
    valid = np.isfinite(points_3d).all(axis=1) & (vz > 1e-5)
    if not np.any(valid):
        return np.empty((0, 2))
    t = CAMERA_TO_PLANE_DISTANCE_M / vz[valid]
    x_plane = vx[valid] * t
    vertical_plane = CAMERA_LOCAL_Y_M + vy[valid] * t
    return np.column_stack([x_plane, vertical_plane])


def density_for(scenario, refresh=False):
    csv = Path(original.RELATIVE_POS_BASE_DIR) / scenario / 'AllSubjects_RelativePosInfo.csv'
    digest = hashlib.sha256(SOURCE.read_bytes())
    digest.update(Path(__file__).read_bytes())
    digest.update(csv.read_bytes())
    entropy_file=Path(original.ENTROPY_CSV)
    if entropy_file.exists():
        digest.update(entropy_file.read_bytes())
    fingerprint=digest.hexdigest()
    cache=ASSETS/'density_cache'/f'{scenario}.npz'
    if cache.exists() and not refresh:
        with np.load(cache) as stored:
            density=stored['density_yx'].copy()
            if (str(stored['fingerprint'])==fingerprint
                    and density.shape==(original.GRID_POINTS,original.GRID_POINTS)
                    and np.isfinite(density).all()):
                return density, fingerprint, csv
    print(f'Computing corrected 0.7 m camera-relative KDE: {scenario}',flush=True)
    frame=pd.read_csv(csv,usecols=COLS).dropna(subset=COLS)
    points=project_to_camera_plane(original.calculate_3d_relative_vectors(frame))
    x0,x1=original.X_RANGE; y0,y1=original.Z_RANGE
    keep=np.isfinite(points).all(axis=1)&(points[:,0]>=x0)&(points[:,0]<=x1)&(points[:,1]>=y0)&(points[:,1]<=y1)
    points=points[keep]
    if len(points)<10:
        raise ValueError(f'Too few valid samples: {scenario}')
    kde=gaussian_kde(points.T,bw_method='scott')
    xx,yy=np.mgrid[x0:x1:complex(original.GRID_POINTS),y0:y1:complex(original.GRID_POINTS)]
    raw=kde(np.vstack([xx.ravel(),yy.ravel()])).reshape(xx.shape)
    delta=raw.max()-raw.min()
    density=((raw-raw.min())/delta if delta>1e-8 else np.zeros_like(raw)).T
    cache.parent.mkdir(exist_ok=True)
    np.savez_compressed(cache,density_yx=density,fingerprint=fingerprint)
    return density, fingerprint, csv


def peak(density):
    iy,ix=np.unravel_index(np.argmax(density),density.shape)
    return (float(np.linspace(*original.X_RANGE,density.shape[1])[ix]),
            float(np.linspace(*original.Z_RANGE,density.shape[0])[iy]))


def display_name(scenario):
    key,smoke=scenario.split('-',1)
    return NAMES[key]+'-'+smoke


def draw_panel(ax, scenario, density, config):
    """Shared background, density and annotations for single and combined plots."""
    key=scenario.split('-')[0]; group=group_for(scenario,config)
    render=config['rendering']
    xr=config['coordinate_system']['x_range_m']; yr=config['coordinate_system']['vertical_range_m']
    bounds=extent(group,config)
    with Image.open(asset_path(group['image_path'])) as image:
        ax.imshow(np.asarray(image.convert('RGB')),extent=bounds,origin='upper',
                  alpha=group['background_alpha'],zorder=0,aspect='equal')
    base_cmap=original.get_custom_cmap()
    colors=base_cmap(np.linspace(0,1,256))
    # Scalar imshow alpha would replace cmap alpha. Multiply the RGBA lookup instead.
    colors[:,3]*=group['heatmap_alpha_multiplier']
    ax.imshow(density,extent=[*original.X_RANGE,*original.Z_RANGE],origin='lower',
              cmap=ListedColormap(colors),vmin=0,vmax=1,zorder=1,aspect='equal',
              interpolation='bilinear',interpolation_stage='rgba')
    ax.set_xlim(xr); ax.set_ylim(yr)
    ax.set_xticks(np.arange(xr[0],xr[1]+1e-9,.25))
    ax.set_yticks(np.arange(yr[0],yr[1]+1e-9,.25))
    ax.set_axisbelow(False)
    ax.grid(color='lightgray',linewidth=.5,alpha=.7)
    original.annotate_density_peak(ax,*peak(density))
    entropy=original.load_entropy_map().get(scenario)
    if entropy is not None and np.isfinite(entropy):
        ax.text(np.mean(xr),yr[1]-.05,f'H = {entropy:.5f} bit',ha='center',va='top',
                fontsize=original.PEAK_FONT_SIZE,zorder=8)


def draw_outlines(ax, scenario, config, outlines):
    key=scenario.split('-')[0]; group=group_for(scenario,config)
    render=config['rendering']; style=render['arrow_outline']
    bounds=extent(group,config)
    specs=[dict(s,color=render.get('ui_region_colors',{}).get(s['name'],s['color']))
           for s in original.ui_specs_for_scenario(scenario)]
    # Convert the approved SVG width (900 units across the axes) to print points.
    points_per_preview_px=ax.get_window_extent().width/ax.figure.dpi*72/render.get('preview_plot_width_px',900)
    for ui in specs:
        left,right,bottom,top=original.unity_ui_rect_to_plane_bounds(ui)
        ax.add_patch(Rectangle((left,bottom),right-left,top-bottom,fill=False,
                               edgecolor=ui['color'],linewidth=2.2*points_per_preview_px,zorder=5))
    contour_group=outline_for(scenario,outlines)
    if style.get('enabled',True) and contour_group:
        w,h=group['image_size_px']
        for contour in contour_group['paths']:
            pts=np.asarray(contour['points'],dtype=float)
            if contour['closed']:
                pts=np.vstack([pts,pts[0]])
            x=bounds[0]+pts[:,0]/w*(bounds[1]-bounds[0])
            y=bounds[3]-pts[:,1]/h*(bounds[3]-bounds[2])
            line=Line2D(x,y,color=style['color'],linewidth=style['width_screen_px']*points_per_preview_px,
                        solid_capstyle='round',solid_joinstyle='round',zorder=7,clip_on=True)
            line.set_path_effects([pe.Stroke(linewidth=(style['width_screen_px']+style['halo_extra_px'])*points_per_preview_px,
                                              foreground=style['halo_color']),pe.Normal()])
            ax.add_line(line)


def validate_figure_font(fig):
    # Fail rather than silently using a fallback font in generated plot text.
    for text in fig.findobj(Text):
        if text.get_text() and text.get_fontproperties().get_name()!='Times New Roman':
            raise RuntimeError(f'Unexpected font: {text.get_text()}')


def draw(scenario, density, config, outlines):
    xr=config['coordinate_system']['x_range_m']; yr=config['coordinate_system']['vertical_range_m']
    fig,ax=plt.subplots(figsize=(8,8*(yr[1]-yr[0])/(xr[1]-xr[0])),dpi=300)
    draw_panel(ax,scenario,density,config)
    ax.set_xlabel('Horizontal (m)',fontsize=14)
    ax.set_ylabel('Vertical (m)',fontsize=14)
    ax.set_title(display_name(scenario),fontsize=16,y=1.02)
    cbar=fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(0,1),cmap=original.get_custom_cmap()),
                      ax=ax,fraction=.046,pad=.04)
    cbar.set_label('Relative density',fontsize=13)
    fig.tight_layout()
    fig.canvas.draw()
    draw_outlines(ax,scenario,config,outlines)
    validate_figure_font(fig)
    output=OUTPUT/f'{display_name(scenario)}_PlaneHeatmap_Background.png'
    fig.savefig(output,dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print('Saved '+output.name,flush=True)


def draw_combined(densities, config, outlines):
    """Render existing densities on one compact, physically aligned 5-by-2 grid."""
    # Overview-only display crop and outline emphasis; other layouts retain
    # their approved configuration and the KDE arrays retain their full extent.
    config=deepcopy(config)
    background_top=max(extent(config['groups'][key],config)[3] for key in COMBINED_ROWS)
    config['coordinate_system']['vertical_range_m']=[1.20,background_top+.03]
    config['rendering']['arrow_outline']['width_screen_px']*=2.25
    xr=config['coordinate_system']['x_range_m']; yr=config['coordinate_system']['vertical_range_m']
    width=14.0
    left,right,top,bottom=1.55,.22,.65,1.55
    gap_x,gap_y=.14,.06
    panel_w=(width-left-right-gap_x)/2
    panel_h=panel_w*(yr[1]-yr[0])/(xr[1]-xr[0])
    height=top+bottom+5*panel_h+4*gap_y
    fig=plt.figure(figsize=(width,height),dpi=300)
    axes=[]
    for row,key in enumerate(COMBINED_ROWS):
        y=height-top-(row+1)*panel_h-row*gap_y
        for col,smoke in enumerate(COMBINED_COLUMNS):
            scenario=f'{key}-{smoke}'
            x=left+col*(panel_w+gap_x)
            ax=fig.add_axes([x/width,y/height,panel_w/width,panel_h/height])
            draw_panel(ax,scenario,densities[scenario][0],config)
            ax.set_yticks(np.arange(yr[0],yr[1]+1e-9,.20))
            bounds=extent(config['groups'][key],config)
            for label in ax.texts:
                if label.get_text().startswith('H = '):
                    label.set_position((bounds[0]+.025,bounds[3]-.020))
                    label.set_horizontalalignment('left')
                    label.set_verticalalignment('top')
            ax.tick_params(axis='both',labelsize=17,length=4,pad=3,
                           labelleft=col==0,labelbottom=row==4)
            for label in ax.get_xticklabels()+ax.get_yticklabels():
                label.set_fontweight('bold')
            # Keep endpoint tick labels inside their own panel at tight seams.
            if row==4:
                ax.get_xticklabels()[0].set_horizontalalignment('left')
                ax.get_xticklabels()[-1].set_horizontalalignment('right')
            if col==0:
                ax.get_yticklabels()[0].set_verticalalignment('bottom')
                ax.get_yticklabels()[-1].set_verticalalignment('top')
            for label in ax.texts:
                label.set_fontsize(19)
                label.set_fontweight('bold')
            axes.append((ax,scenario))
            if row==0:
                ax.set_title('Low smoke' if col==0 else 'High smoke',
                             fontsize=26,fontweight='bold',pad=12)
        fig.text(.55/width,(y+panel_h/2)/height,NAMES[key],rotation=90,
                 ha='center',va='center',fontsize=25,fontweight='bold')
    fig.text(.10/width,(bottom+(5*panel_h+4*gap_y)/2)/height,'Vertical (m)',
             rotation=90,ha='center',va='center',fontsize=23,fontweight='bold')
    fig.text((left+(2*panel_w+gap_x)/2)/width,1.00/height,'Horizontal (m)',
             ha='center',va='center',fontsize=23,fontweight='bold')
    # One shared scale: every density array is independently normalized to [0, 1]
    # by the existing KDE workflow, retaining the original relative-density meaning.
    cax=fig.add_axes([left/width,.57/height,(2*panel_w+gap_x)/width,.18/height])
    cbar=fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(0,1),
                      cmap=original.get_custom_cmap()),cax=cax,orientation='horizontal',
                      ticks=np.linspace(0,1,6))
    cbar.set_label('Relative density',fontsize=22,fontweight='bold',labelpad=6)
    cbar.ax.tick_params(labelsize=17,pad=3)
    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')
    fig.canvas.draw()
    for ax,scenario in axes:
        draw_outlines(ax,scenario,config,outlines)
    validate_figure_font(fig)
    output=OUTPUT/'GazePlaneHeatmap_All_Conditions_Background.png'
    temporary=output.with_name('.'+output.stem+'_render.png')
    try:
        fig.savefig(temporary,dpi=300,bbox_inches='tight',pad_inches=.04,facecolor='white')
        temporary.replace(output)
    finally:
        plt.close(fig)
        if temporary.exists():
            temporary.unlink()
    print('Saved '+output.name,flush=True)


def densities_from_preview():
    """Reuse the exact ten arrays represented in the last generated HTML preview.

    This layout-only mode deliberately does not reevaluate changed CSVs or scripts.
    A normal run (or --no-cache) refreshes the data and all outputs together.
    """
    preview=OUTPUT/'背景叠加预览_箭头描边与裁剪.html'
    match=re.search(r'<script id="payload" type="application/json">(.*?)</script>',
                    preview.read_text(encoding='utf-8'),flags=re.DOTALL)
    if match is None:
        raise ValueError('Missing preview payload; run normally to rebuild all outputs')
    payload=json.loads(match.group(1))
    entropy_map=original.load_entropy_map()
    densities={}
    for key in COMBINED_ROWS:
        for smoke in COMBINED_COLUMNS:
            scenario=f'{key}-{smoke}'
            entry=payload['scenarios'][scenario]
            if entropy_map.get(scenario)!=entry['entropy_bit']:
                raise ValueError(f'Entropy changed since the preview: {scenario}; run normally')
            with np.load(ASSETS/'density_cache'/f'{scenario}.npz') as stored:
                density=stored['density_yx'].copy()
                fingerprint=str(stored['fingerprint'])
            if (fingerprint!=entry['fingerprint']
                    or density.shape!=(original.GRID_POINTS,original.GRID_POINTS)
                    or not np.isfinite(density).all()
                    or density.min()<0 or density.max()>1):
                raise ValueError(f'Cache/preview mismatch: {scenario}; run normally')
            # Check the actual plotted RGBA values as well as the cache identifier.
            encoded=entry['heatmap'].split(',',1)[1]
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as heatmap:
                expected=original.get_custom_cmap()(density[::-1],bytes=True)
                if not np.array_equal(np.asarray(heatmap),expected):
                    raise ValueError(f'Cached density differs from preview: {scenario}')
            densities[scenario]=(density,fingerprint,Path(entry['input_csv']))
    return densities


def data_url(image):
    buf=io.BytesIO(); image.save(buf,format='PNG',optimize=True)
    return 'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode('ascii')


def write_preview(config,outlines,densities):
    render=config['rendering']; style=render['arrow_outline']
    configuration_id=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:12]
    projection_script = Path(__file__).resolve()
    payload={'schema_version':1,'script':str(projection_script),'script_sha256':hashlib.sha256(projection_script.read_bytes()).hexdigest(),
             'upstream_script':str(SOURCE),'upstream_script_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
             'camera_local_z_m':CAMERA_LOCAL_Z_M,'canvas_plane_local_z_m':CANVAS_PLANE_LOCAL_Z_M,
             **config['coordinate_system'],'density_x_range_m':list(original.X_RANGE),'density_vertical_range_m':list(original.Z_RANGE),
             'revision':'background-final-'+configuration_id,'groups':{},'scenarios':{},'initial_states':{},
             'backgrounds':{},'scenario_overrides':config.get('scenario_overrides',{}),
             'initial_outline_style':{'color':style['color'],'width':style['width_screen_px'],
                                      'enabled':style.get('enabled',True),'minimap_color':render['ui_region_colors']['MiniMap']}}
    fields=['center_x_m','center_vertical_m','scale_x','scale_y','background_alpha','heatmap_alpha_multiplier']
    for key in NAMES:
        g=config['groups'][key]; path=asset_path(g['image_path'])
        with Image.open(path) as im:
            im.thumbnail((1800,1200),Image.Resampling.LANCZOS)
            encoded=data_url(im.convert('RGB'))
        payload['groups'][key]={'label':NAMES[key],'image_path':str(path),'image_size_px':g['image_size_px'],
            'image_preview':encoded,'arrow_paths':outlines.get(key,{}).get('paths',[]),
            'regions':[{'name':s['name'],'bounds':original.unity_ui_rect_to_plane_bounds(s),'color':s['color']}
                       for s in original.ui_specs_for_scenario(key)]}
        payload['initial_states'][key]={f:g[f] for f in fields}
    cmap=original.get_custom_cmap()
    for scenario,(density,fingerprint,csv) in densities.items():
        group=group_for(scenario,config); path=asset_path(group['image_path'])
        with Image.open(path) as image:
            image.thumbnail((1800,1200),Image.Resampling.LANCZOS)
            encoded=data_url(image.convert('RGB'))
        payload['backgrounds'][scenario]={'image_path':str(path),'image_size_px':group['image_size_px'],
            'image_preview':encoded,'arrow_paths':outline_for(scenario,outlines).get('paths',[])}
        payload['scenarios'][scenario]={'heatmap':data_url(Image.fromarray(cmap(density[::-1],bytes=True))),
            'peak_m':peak(density),'entropy_bit':original.load_entropy_map().get(scenario),'input_csv':str(csv),'fingerprint':fingerprint}
    payload['colorbar']=data_url(Image.fromarray(cmap(np.linspace(1,0,256)[:,None],bytes=True)))
    template=(ASSETS/'preview_outline_template.html').read_text(encoding='utf-8')
    target=OUTPUT/'背景叠加预览_箭头描边与裁剪.html'
    target.write_text(template.replace('__PAYLOAD__',json.dumps(payload,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
    print('Saved editable HTML preview',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ASSETS/'plot_config.json')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--preview-only',action='store_true')
    modes.add_argument('--combined-only',action='store_true',
                       help='Only render the overview from the existing preview and matching caches')
    modes.add_argument('--redraw-cached',action='store_true',
                       help='Redraw background figures and preview using verified unchanged density caches')
    parser.add_argument('--no-cache',action='store_true',help='Recompute corrected 0.7 m camera-relative KDE from CSVs')
    parser.add_argument('--dry-run',action='store_true',help='Validate assets, configuration and font only')
    args=parser.parse_args()
    if (args.combined_only or args.redraw_cached) and args.no_cache:
        parser.error('Cached redraw modes cannot be used with --no-cache')
    config=read_json(args.config)
    outlines=read_json(asset_path(config['rendering']['arrow_outline']['contours_file']))['groups']
    validate(config,outlines)
    font_manager.findfont(font_manager.FontProperties(family='Times New Roman'),fallback_to_default=False)
    plt.rcParams.update({'font.family':'Times New Roman','font.serif':['Times New Roman'],
                         'mathtext.fontset':'stix','axes.unicode_minus':False})
    if args.dry_run:
        print('Validated backgrounds, contours, configuration and Times New Roman.')
        return
    before=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if args.combined_only:
        draw_combined(densities_from_preview(),config,outlines)
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==before,'Original script changed'
        return
    if args.redraw_cached:
        densities=densities_from_preview()
        for scenario in original.SCENARIO_FOLDERS:
            # Low-smoke pixels and layouts have not changed; retain those PNGs.
            if scenario.endswith('-HighSmoke'):
                draw(scenario,densities[scenario][0],config,outlines)
        write_preview(config,outlines,densities)
        draw_combined(densities,config,outlines)
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==before,'Original script changed'
        return
    densities={}
    for scenario in original.SCENARIO_FOLDERS:
        densities[scenario]=density_for(scenario,args.no_cache)
        if not args.preview_only:
            draw(scenario,densities[scenario][0],config,outlines)
    write_preview(config,outlines,densities)
    if not args.preview_only:
        draw_combined(densities,config,outlines)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==before,'Original script changed'


if __name__=='__main__':
    main()
