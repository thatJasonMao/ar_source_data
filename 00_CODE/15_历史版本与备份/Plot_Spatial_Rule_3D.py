"""Portable paired 3D and 2D cell heatmaps of a fitted spatial IRL rule.

Render an exported Best_Model spatial field from any working directory:
    python -B Plot_Spatial_Rule_3D.py --run "<Best_Model directory>"
Or render a standalone CSV with --input "<global_spatial_rule_field.csv.gz>".
Choose --surface-style cells (original), bars (solid columns), or continuous.
Outputs default to Results_3D / Results_3D_Bars / Results_3D_Continuous beside
the input CSV. Use --force to replace an existing output of the same style.

Only numpy, pandas, matplotlib and Pillow are required. No model, geometry export,
trial file, GPU or project module is loaded. Optional --basemap-root loads the
two registered scene PNGs only for the 3D display base. Colour definitions
are copied verbatim from the requested Visualize_Spatial_Metric_Surfaces.py
palette registry (162 / 99 / 100; all unreversed). V6 additionally renders
StaticSign with the Matplotlib cividis colour map. No training is performed.

Original cells and bars keep each raw signed value as their flat top height.
The continuous display joins each cell centre to four shared corners. Optional
--smooth-iterations 4 --smooth-strength .5 lightly diffuses display heights
along occupied edge neighbours; it does not modify the input field or fill holes.
With smoothing disabled, raw centre heights remain exact. No KDE, clipping or
subsampling is used. Height denotes utility, not building elevation.
Every face is coloured from its original cell value without lighting. Optional
--contrast-channels StaticSign,TextBar --contrast-quantile .3 --contrast-position .1
sets a per-channel power norm whose P30 is at colour 0.1 while retaining both
raw endpoint colours. Or use --contrast-method piecewise --contrast-quantile .15
--contrast-position .1 to compress only the bottom 15% into 10% of the palette,
leaving the rest linear. Paired 2D figures, 3D figures and colour bars share this
same norm; 2D geometry is not smoothed. Defaults reproduce the original display.
--layout-profile publication refines only 3D typography and positions: it retains
the 16 x 13 inch canvas, DPI and original square panel viewports, uses a shared
1.12 projection zoom, and moves the colour bar to the bottom. No image cropping
or resampling is performed. The legacy layout remains available by default.
"""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm
    from matplotlib.collections import PolyCollection, LineCollection
    from matplotlib.ticker import MaxNLocator, ScalarFormatter
    from matplotlib.image import AxesImage
    from matplotlib.transforms import Affine2D
    from matplotlib.path import Path as MplPath
    from mpl_toolkits.mplot3d import proj3d
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
except RuntimeError as exc:
    # The observed failure occurs during Matplotlib import, before any output.
    # A fresh interpreter recovered locally; retry this exact startup error once.
    # Data/render errors and a second startup failure still propagate normally.
    retry_flag = 'A5_IRL_3D_IMPORT_RETRIED'
    if (str(exc) != 'internal error in regular expression engine'
            or __name__ != '__main__' or os.environ.get(retry_flag) == '1'):
        raise
    print('Matplotlib import failed with a regex-engine error; retrying once in a fresh Python process.',
          file=sys.stderr, flush=True)
    retry_env = os.environ.copy()
    retry_env[retry_flag] = '1'
    retry_result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:]],
                                  env=retry_env)
    raise SystemExit(retry_result.returncode)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(items, **kwargs):
        return items

CSV_NAME = 'global_spatial_rule_field.csv.gz'
FIELD = 'relative_information_utility'
LEGACY_GUIDANCES = ('ArrowRibbon', 'MiniMap', 'TextBar')
GUIDANCES = ('StaticSign', *LEGACY_GUIDANCES)
GUIDANCE_LABELS = {'StaticSign': 'StaticSign', 'ArrowRibbon': 'Arrow', 'MiniMap': 'MiniMap', 'TextBar': 'TextBar'}
SMOKES = ('HighSmoke', 'LowSmoke')
FLOORS = ('Hall', 'Platform')
CELL_M = 0.4
DPI = 300
FIGSIZE = (16, 13)
VIEW_ELEVATION = 42
VIEW_AZIMUTH = -65
SURFACE_STYLES = ('cells', 'bars', 'continuous')
OUTPUT_FOLDERS = {'cells': 'Results_3D', 'bars': 'Results_3D_Bars',
                  'continuous': 'Results_3D_Continuous'}
LAYOUT_PROFILES = ('legacy', 'publication')
# The original Axes3D active viewport is square: its size is set by the
# GridSpec row height. Keep that exact physical size when moving the panels.
PANEL_SIZE_INCHES = FIGSIZE[1] * (.91 - .09) / (2 + .32)
PUBLICATION_CENTRES_X = (.345, .655)
PUBLICATION_CENTRES_Y = (.735, .305)
PUBLICATION_ZOOM = 1.12
BASEMAP_FILENAMES = {'Hall': 'Image Sequence_007_0000.png',
                     'Platform': 'Image Sequence_006_0000.png'}
BASEMAP_VARIANTS = {'walls': BASEMAP_FILENAMES,
                   'no-walls': {'Hall': 'Concourse_Unity_Camera.png',
                                'Platform': 'Platform_Unity_Camera.png'}}
# Physical floor elevations (semantic-zone y_min), not the zone volume tops
# used to float metric surfaces in Visualize_Spatial_Metric_Surfaces.py.
BASEMAP_FLOOR_Y = {'Hall': -11.8, 'Platform': -15.771}
UNITY_CAMERA = dict(position=(81.0, 71.2, 69.1), rotation=(35.264, 225.0, 0.0),
                    orthographic_size=53.0, width_px=1920, height_px=1080)


def layout_metadata(profile='legacy'):
    if profile not in LAYOUT_PROFILES:
        raise ValueError('Unknown layout profile: ' + str(profile))
    layout = dict(figsize_inches=list(FIGSIZE), grid=[2, 2],
        row_order=list(SMOKES), column_order=list(FLOORS),
        view_elevation=VIEW_ELEVATION, view_azimuth=VIEW_AZIMUTH, projection='ortho',
        subplot_adjust=dict(left=.03, right=.86, top=.91, bottom=.09, wspace=.03, hspace=.32),
        colorbar_axes=[.90, .19, .018, .60], title_y=.97, footer_y=.03,
        horizontal_aspect='Physical X/Z ratio', relief_display_aspect=35)
    if profile == 'publication':
        width, height = PANEL_SIZE_INCHES / FIGSIZE[0], PANEL_SIZE_INCHES / FIGSIZE[1]
        layout.update(subplot_adjust=None,
            panel_axes=[[x-width/2, y-height/2, width, height]
                        for y in PUBLICATION_CENTRES_Y for x in PUBLICATION_CENTRES_X],
            panel_size_inches=[PANEL_SIZE_INCHES, PANEL_SIZE_INCHES],
            colorbar_axes=[.22, .060, .56, .018], colorbar_orientation='horizontal',
            title_y=.980, title_fontsize=36, title_fontweight='bold', title_content='Guidance label only',
            panel_title_fontsize=26, panel_title_fontweight='bold', panel_title_pad=0,
            axis_label_fontsize=11*1.08*1.5, axis_tick_fontsize=9*1.08*1.8,
            colorbar_label_fontsize=11*1.08, colorbar_tick_fontsize=11*1.08*2,
            colorbar_label_visible=False,
            hall_x_labelpad=14, vertical_utility_axis_visible=False,
            footer_y=None, projection_zoom=PUBLICATION_ZOOM,
            panel_background='Transparent axes over white figure')
    return layout


def basemap_metadata(root=None, variant='walls'):
    if variant not in BASEMAP_VARIANTS:
        raise ValueError('Unknown basemap variant: ' + str(variant))
    filenames = BASEMAP_VARIANTS[variant]
    common = dict(mapping=dict(filenames), variant=variant,
        projection='Unity orthographic camera inverse onto each floor plane',
        display_method='Full-resolution affine raster below utility surfaces',
        alpha=.74, canvas=[1920, 1080], camera=dict(UNITY_CAMERA),
        floor_y=dict(BASEMAP_FLOOR_Y), scope='3D only',
        floor_y_source='Physical floor planes: semantic_zones.csv y_min for Hall and Platform',
        limitation='Single-view screenshot rectified at floor height; elevated walls/fixtures retain source-view relief',
        source_fit='Scale to 1080 px high and centre horizontally on 1920 px canvas',
        clip='Existing horizontal axis bounds; retain source alpha')
    if root in (None, ''):
        return dict(enabled=False, root=None, files={}, **common)
    root = Path(root).resolve()
    files = {}
    for floor, filename in filenames.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            files[floor] = dict(filename=filename, path=str(path), sha256=sha256(path),
                                size=list(image.size), mode=image.mode)
    return dict(enabled=True, root=str(root), files=files, **common)


def _camera_basis():
    def rx(degrees):
        angle = np.deg2rad(degrees); c, s = np.cos(angle), np.sin(angle)
        return np.asarray([[1.,0.,0.],[0.,c,-s],[0.,s,c]])
    def ry(degrees):
        angle = np.deg2rad(degrees); c, s = np.cos(angle), np.sin(angle)
        return np.asarray([[c,0.,s],[0.,1.,0.],[-s,0.,c]])
    rotation = ry(UNITY_CAMERA['rotation'][1]) @ rx(UNITY_CAMERA['rotation'][0])
    return rotation @ np.asarray([1.,0.,0.]), rotation @ np.asarray([0.,1.,0.])


def basemap_world_xz(u, v, floor, source_size):
    """Invert source-image pixels onto the registered Unity floor plane."""
    source_width_px, source_height_px = source_size
    # Match the existing overlay implementation: the 3:2 source is resized to
    # the 1920x1080 Unity canvas and centred horizontally before inverse mapping.
    source_width = int(round(source_width_px * UNITY_CAMERA['height_px'] / source_height_px))
    left = (UNITY_CAMERA['width_px'] - source_width) / 2.0
    uu = left + np.asarray(u) * source_width / source_width_px
    vv = np.asarray(v) * UNITY_CAMERA['height_px'] / source_height_px
    right, up = _camera_basis()
    half_h = UNITY_CAMERA['orthographic_size']
    half_w = half_h * UNITY_CAMERA['width_px'] / UNITY_CAMERA['height_px']
    camera_x = (uu / UNITY_CAMERA['width_px'] * 2.0 - 1.0) * half_w
    camera_y = (1.0 - vv / UNITY_CAMERA['height_px'] * 2.0) * half_h
    position = np.asarray(UNITY_CAMERA['position'], dtype=float)
    matrix = np.asarray([[right[0], right[2]], [up[0], up[2]]], dtype=float)
    floor_y = BASEMAP_FLOOR_Y[floor]
    offsets = np.column_stack([camera_x.ravel() + (position @ right) - floor_y * right[1],
                               camera_y.ravel() + (position @ up) - floor_y * up[1]])
    world_xz = np.linalg.solve(matrix, offsets.T).T
    return world_xz


@lru_cache(maxsize=4)
def basemap_pixels(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGBA'))


class FloorBasemapImage(AxesImage):
    """Full-resolution raster on the display base, under all utility faces.

    Both cameras are orthographic, so the ground-plane mapping is affine.
    Recompute its projection at draw time; avoid texture-mesh seams, coarse
    pixel polygons and mplot3d collection depth sorting over the utility mesh.
    """
    def __init__(self, ax, root, floor, base, bounds, variant='walls'):
        rgba = basemap_pixels(str(Path(root) / BASEMAP_VARIANTS[variant][floor]))
        height, width = rgba.shape[:2]
        super().__init__(ax, origin='upper', extent=(0, width, height, 0),
                         interpolation='bilinear', zorder=0, alpha=.74)
        self.set_data(rgba)
        xz = basemap_world_xz([0, width, 0], [0, 0, height], floor, (width, height))
        self.plane_anchors = np.column_stack([-xz, np.full(3, base)])
        xmin, xmax, ymin, ymax = bounds
        self.plane_clip = np.asarray([[xmin, ymin, base], [xmax, ymin, base],
                                      [xmax, ymax, base], [xmin, ymax, base],
                                      [xmin, ymin, base]])
        self.source_size = width, height

    def draw(self, renderer):
        matrix = self.axes.get_proj()
        px, py, _ = proj3d.proj_transform(*self.plane_anchors.T, matrix)
        points = np.column_stack([px, py])
        width, height = self.source_size
        dx, dy = (points[1]-points[0])/width, (points[2]-points[0])/height
        affine = Affine2D.from_values(dx[0], dx[1], dy[0], dy[1], *points[0])
        self.set_transform(affine + self.axes.transData)
        cx, cy, _ = proj3d.proj_transform(*self.plane_clip.T, matrix)
        self.set_clip_path(MplPath(np.column_stack([cx, cy])), self.axes.transData)
        super().draw(renderer)
PALETTE_NUMBERS = {'StaticSign': None, 'ArrowRibbon': 162, 'MiniMap': 99, 'TextBar': 100}
# This marker is replaced once when installing the script. The installed
# version embeds every original colour stop and has no palette file dependency.
PALETTES = {'ArrowRibbon': ('#8000ff', '#7e03ff', '#7c06ff', '#7a09ff', '#780dff', '#7610ff', '#7413ff', '#7216ff',
                 '#7019ff', '#6e1cff', '#6c1fff', '#6a22fe', '#6826fe', '#6629fe', '#642cfe', '#622ffe',
                 '#6032fe', '#5e35fe', '#5c38fd', '#5a3bfd', '#583efd', '#5641fd', '#5444fd', '#5247fc',
                 '#504afc', '#4e4dfc', '#4c50fc', '#4a53fb', '#4856fb', '#4659fb', '#445cfb', '#425ffa',
                 '#4062fa', '#3e65fa', '#3c68f9', '#396bf9', '#386df9', '#3670f8', '#3473f8', '#3176f8',
                 '#3079f7', '#2e7bf7', '#2c7ef7', '#2981f6', '#2884f6', '#2686f5', '#2489f5', '#218cf4',
                 '#208ef4', '#1e91f3', '#1c93f3', '#1996f3', '#1898f2', '#169bf2', '#149df1', '#11a0f1',
                 '#10a2f0', '#0ea5ef', '#0ca7ef', '#09a9ee', '#08acee', '#06aeed', '#04b0ed', '#01b3ec',
                 '#00b5eb', '#02b7eb', '#04b9ea', '#07bbea', '#08bee9', '#0ac0e8', '#0dc2e8', '#0fc4e7',
                 '#10c6e6', '#12c8e6', '#14cae5', '#17cbe4', '#18cde4', '#1acfe3', '#1dd1e2', '#1fd3e1',
                 '#20d5e1', '#22d6e0', '#24d8df', '#27dade', '#28dbde', '#2adddd', '#2ddedc', '#2fe0db',
                 '#30e1da', '#32e3da', '#34e4d9', '#37e6d8', '#38e7d7', '#3ae8d6', '#3dead5', '#3febd5',
                 '#40ecd4', '#42edd3', '#44eed2', '#46efd1', '#48f1d0', '#4af2cf', '#4df3ce', '#4ef3cd',
                 '#50f4cc', '#52f5cb', '#54f6cb', '#56f7ca', '#58f8c9', '#5af8c8', '#5df9c7', '#5efac6',
                 '#60fac5', '#62fbc4', '#64fbc3', '#66fcc2', '#68fcc1', '#6afdc0', '#6dfdbf', '#6efebe',
                 '#70febc', '#72febb', '#74feba', '#76ffb9', '#78ffb8', '#7affb7', '#7dffb6', '#7effb5',
                 '#80ffb4', '#82ffb3', '#84ffb2', '#86ffb0', '#88ffaf', '#8bfeae', '#8cfead', '#8efeac',
                 '#90feab', '#92fda9', '#94fda8', '#96fca7', '#99fca6', '#9bfba5', '#9cfba4', '#9efaa2',
                 '#a0faa1', '#a2f9a0', '#a4f89f', '#a6f89d', '#a8f79c', '#abf69b', '#acf59a', '#aef498',
                 '#b0f397', '#b2f396', '#b4f295', '#b6f193', '#b9ef92', '#bbee91', '#bced8f', '#beec8e',
                 '#c0eb8d', '#c2ea8c', '#c4e88a', '#c6e789', '#c8e688', '#cbe486', '#cce385', '#cee184',
                 '#d0e082', '#d2de81', '#d4dd80', '#d6db7e', '#d9da7d', '#dbd87b', '#dcd67a', '#ded579',
                 '#e0d377', '#e2d176', '#e4cf74', '#e6cd73', '#e8cb72', '#ebca70', '#ecc86f', '#eec66d',
                 '#f0c46c', '#f2c26b', '#f4c069', '#f6be68', '#f9bb66', '#fbb965', '#fcb763', '#feb562',
                 '#ffb360', '#ffb05f', '#ffae5e', '#ffac5c', '#ffa95b', '#ffa759', '#ffa558', '#ffa256',
                 '#ffa055', '#ff9d53', '#ff9b52', '#ff9850', '#ff964f', '#ff934d', '#ff914c', '#ff8e4a',
                 '#ff8c49', '#ff8947', '#ff8646', '#ff8444', '#ff8143', '#ff7e41', '#ff7b40', '#ff793e',
                 '#ff763d', '#ff733b', '#ff703a', '#ff6d38', '#ff6b37', '#ff6835', '#ff6533', '#ff6232',
                 '#ff5f30', '#ff5c2f', '#ff592d', '#ff562c', '#ff532a', '#ff5029', '#ff4d27', '#ff4a26',
                 '#ff4724', '#ff4422', '#ff4121', '#ff3e1f', '#ff3b1e', '#ff381c', '#ff351b', '#ff3219',
                 '#ff2f18', '#ff2c16', '#ff2914', '#ff2613', '#ff2211', '#ff1f10', '#ff1c0e', '#ff190d',
                 '#ff160b', '#ff1309', '#ff1008', '#ff0d06', '#ff0905', '#ff0603', '#ff0302', '#ff0000'),
 'MiniMap': ('#0d0887', '#100788', '#130789', '#16078a', '#19068c', '#1b068d', '#1d068e', '#20068f',
             '#220690', '#240691', '#260591', '#280592', '#2a0593', '#2c0594', '#2e0595', '#2f0596',
             '#310597', '#330597', '#350498', '#370499', '#38049a', '#3a049a', '#3c049b', '#3e049c',
             '#3f049c', '#41049d', '#43039e', '#44039e', '#46039f', '#48039f', '#4903a0', '#4b03a1',
             '#4c02a1', '#4e02a2', '#5002a2', '#5102a3', '#5302a3', '#5502a4', '#5601a4', '#5801a4',
             '#5901a5', '#5b01a5', '#5c01a6', '#5e01a6', '#6001a6', '#6100a7', '#6300a7', '#6400a7',
             '#6600a7', '#6700a8', '#6900a8', '#6a00a8', '#6c00a8', '#6e00a8', '#6f00a8', '#7100a8',
             '#7201a8', '#7401a8', '#7501a8', '#7701a8', '#7801a8', '#7a02a8', '#7b02a8', '#7d03a8',
             '#7e03a8', '#8004a8', '#8104a7', '#8305a7', '#8405a7', '#8606a6', '#8707a6', '#8808a6',
             '#8a09a5', '#8b0aa5', '#8d0ba5', '#8e0ca4', '#8f0da4', '#910ea3', '#920fa3', '#9410a2',
             '#9511a1', '#9613a1', '#9814a0', '#99159f', '#9a169f', '#9c179e', '#9d189d', '#9e199d',
             '#a01a9c', '#a11b9b', '#a21d9a', '#a31e9a', '#a51f99', '#a62098', '#a72197', '#a82296',
             '#aa2395', '#ab2494', '#ac2694', '#ad2793', '#ae2892', '#b02991', '#b12a90', '#b22b8f',
             '#b32c8e', '#b42e8d', '#b52f8c', '#b6308b', '#b7318a', '#b83289', '#ba3388', '#bb3488',
             '#bc3587', '#bd3786', '#be3885', '#bf3984', '#c03a83', '#c13b82', '#c23c81', '#c33d80',
             '#c43e7f', '#c5407e', '#c6417d', '#c7427c', '#c8437b', '#c9447a', '#ca457a', '#cb4679',
             '#cc4778', '#cc4977', '#cd4a76', '#ce4b75', '#cf4c74', '#d04d73', '#d14e72', '#d24f71',
             '#d35171', '#d45270', '#d5536f', '#d5546e', '#d6556d', '#d7566c', '#d8576b', '#d9586a',
             '#da5a6a', '#da5b69', '#db5c68', '#dc5d67', '#dd5e66', '#de5f65', '#de6164', '#df6263',
             '#e06363', '#e16462', '#e26561', '#e26660', '#e3685f', '#e4695e', '#e56a5d', '#e56b5d',
             '#e66c5c', '#e76e5b', '#e76f5a', '#e87059', '#e97158', '#e97257', '#ea7457', '#eb7556',
             '#eb7655', '#ec7754', '#ed7953', '#ed7a52', '#ee7b51', '#ef7c51', '#ef7e50', '#f07f4f',
             '#f0804e', '#f1814d', '#f1834c', '#f2844b', '#f3854b', '#f3874a', '#f48849', '#f48948',
             '#f58b47', '#f58c46', '#f68d45', '#f68f44', '#f79044', '#f79143', '#f79342', '#f89441',
             '#f89540', '#f9973f', '#f9983e', '#f99a3e', '#fa9b3d', '#fa9c3c', '#fa9e3b', '#fb9f3a',
             '#fba139', '#fba238', '#fca338', '#fca537', '#fca636', '#fca835', '#fca934', '#fdab33',
             '#fdac33', '#fdae32', '#fdaf31', '#fdb130', '#fdb22f', '#fdb42f', '#fdb52e', '#feb72d',
             '#feb82c', '#feba2c', '#febb2b', '#febd2a', '#febe2a', '#fec029', '#fdc229', '#fdc328',
             '#fdc527', '#fdc627', '#fdc827', '#fdca26', '#fdcb26', '#fccd25', '#fcce25', '#fcd025',
             '#fcd225', '#fbd324', '#fbd524', '#fbd724', '#fad824', '#fada24', '#f9dc24', '#f9dd25',
             '#f8df25', '#f8e125', '#f7e225', '#f7e425', '#f6e626', '#f6e826', '#f5e926', '#f5eb27',
             '#f4ed27', '#f3ee27', '#f3f027', '#f2f227', '#f1f426', '#f1f525', '#f0f724', '#f0f921'),
 'TextBar': ('#440154', '#440256', '#450457', '#450559', '#46075a', '#46085c', '#460a5d', '#460b5e',
             '#470d60', '#470e61', '#471063', '#471164', '#471365', '#481467', '#481668', '#481769',
             '#48186a', '#481a6c', '#481b6d', '#481c6e', '#481d6f', '#481f70', '#482071', '#482173',
             '#482374', '#482475', '#482576', '#482677', '#482878', '#482979', '#472a7a', '#472c7a',
             '#472d7b', '#472e7c', '#472f7d', '#46307e', '#46327e', '#46337f', '#463480', '#453581',
             '#453781', '#453882', '#443983', '#443a83', '#443b84', '#433d84', '#433e85', '#423f85',
             '#424086', '#424186', '#414287', '#414487', '#404588', '#404688', '#3f4788', '#3f4889',
             '#3e4989', '#3e4a89', '#3e4c8a', '#3d4d8a', '#3d4e8a', '#3c4f8a', '#3c508b', '#3b518b',
             '#3b528b', '#3a538b', '#3a548c', '#39558c', '#39568c', '#38588c', '#38598c', '#375a8c',
             '#375b8d', '#365c8d', '#365d8d', '#355e8d', '#355f8d', '#34608d', '#34618d', '#33628d',
             '#33638d', '#32648e', '#32658e', '#31668e', '#31678e', '#31688e', '#30698e', '#306a8e',
             '#2f6b8e', '#2f6c8e', '#2e6d8e', '#2e6e8e', '#2e6f8e', '#2d708e', '#2d718e', '#2c718e',
             '#2c728e', '#2c738e', '#2b748e', '#2b758e', '#2a768e', '#2a778e', '#2a788e', '#29798e',
             '#297a8e', '#297b8e', '#287c8e', '#287d8e', '#277e8e', '#277f8e', '#27808e', '#26818e',
             '#26828e', '#26828e', '#25838e', '#25848e', '#25858e', '#24868e', '#24878e', '#23888e',
             '#23898e', '#238a8d', '#228b8d', '#228c8d', '#228d8d', '#218e8d', '#218f8d', '#21908d',
             '#21918c', '#20928c', '#20928c', '#20938c', '#1f948c', '#1f958b', '#1f968b', '#1f978b',
             '#1f988b', '#1f998a', '#1f9a8a', '#1e9b8a', '#1e9c89', '#1e9d89', '#1f9e89', '#1f9f88',
             '#1fa088', '#1fa188', '#1fa187', '#1fa287', '#20a386', '#20a486', '#21a585', '#21a685',
             '#22a785', '#22a884', '#23a983', '#24aa83', '#25ab82', '#25ac82', '#26ad81', '#27ad81',
             '#28ae80', '#29af7f', '#2ab07f', '#2cb17e', '#2db27d', '#2eb37c', '#2fb47c', '#31b57b',
             '#32b67a', '#34b679', '#35b779', '#37b878', '#38b977', '#3aba76', '#3bbb75', '#3dbc74',
             '#3fbc73', '#40bd72', '#42be71', '#44bf70', '#46c06f', '#48c16e', '#4ac16d', '#4cc26c',
             '#4ec36b', '#50c46a', '#52c569', '#54c568', '#56c667', '#58c765', '#5ac864', '#5cc863',
             '#5ec962', '#60ca60', '#63cb5f', '#65cb5e', '#67cc5c', '#69cd5b', '#6ccd5a', '#6ece58',
             '#70cf57', '#73d056', '#75d054', '#77d153', '#7ad151', '#7cd250', '#7fd34e', '#81d34d',
             '#84d44b', '#86d549', '#89d548', '#8bd646', '#8ed645', '#90d743', '#93d741', '#95d840',
             '#98d83e', '#9bd93c', '#9dd93b', '#a0da39', '#a2da37', '#a5db36', '#a8db34', '#aadc32',
             '#addc30', '#b0dd2f', '#b2dd2d', '#b5de2b', '#b8de29', '#bade28', '#bddf26', '#c0df25',
             '#c2df23', '#c5e021', '#c8e020', '#cae11f', '#cde11d', '#d0e11c', '#d2e21b', '#d5e21a',
             '#d8e219', '#dae319', '#dde318', '#dfe318', '#e2e418', '#e5e419', '#e7e419', '#eae51a',
             '#ece51b', '#efe51c', '#f1e51d', '#f4e61e', '#f6e620', '#f8e621', '#fbe723', '#fde725')}
PALETTE_PROVENANCE = {'reference': 'Visualize_Spatial_Metric_Surfaces.py',
 'registry': 'a5_colormaps.py',
 'reference_sha256': 'ee0770cc25640c39c563ae8c5bf277b9131efdfe8633a8fe4e5d96f4749e7eab',
 'registry_sha256': '6197db27aa756bc9d478369918249c86c4c6b940472c54c6aabd48b245fc5bb8',
 'copied_exact': True,
 'original_stops': {'ArrowRibbon': 256, 'MiniMap': 256, 'TextBar': 256}}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def resolve_input(*, run=None, input_path=None):
    """Resolve explicit inputs, never guess which training run is best/latest."""
    if (run is None) == (input_path is None):
        raise ValueError('Specify exactly one of --run or --input')
    guidances = None
    if run is not None:
        run = Path(run).resolve()
        config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
        if config.get('status') != 'complete':
            raise ValueError('Requires a completed Best_Model export')
        if config.get('method') == 'four_channel_full_frame_spatial_irl_v6':
            folder, guidances = 'Results_Global_Spatial_Field_V6', GUIDANCES
        elif config.get('method') == 'budgeted_full_frame_spatial_IRL_v4':
            folder, guidances = 'Results_Global_Spatial_Field_V4', LEGACY_GUIDANCES
        else:
            raise ValueError('Unsupported model method; use --input for a standalone CSV')
        input_path = run / folder / CSV_NAME
    source = Path(input_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f'Spatial field not found: {source}. Export the 2D spatial field first.')
    return source, guidances


def load_data(path, guidances=None):
    required = ['x', 'z', 'floor', 'smoke', 'guidance', FIELD]
    frame = pd.read_csv(path, usecols=required)
    # A direct CSV auto-detects V6; --run explicitly requires all four V6 channels.
    if guidances is None:
        guidances = GUIDANCES if frame.guidance.eq('StaticSign').any() else LEGACY_GUIDANCES
    frame = frame.loc[frame.guidance.isin(guidances)].copy()
    if frame.empty:
        raise ValueError('No supported information-channel cells in the input')
    if not np.isfinite(frame[['x', 'z', FIELD]].to_numpy(float)).all():
        raise ValueError('Selected input cells have invalid coordinates or utility')
    if not frame.floor.isin(FLOORS).all() or not frame.smoke.isin(SMOKES).all():
        raise ValueError('Unknown floor or smoke level')
    if frame.duplicated(['guidance', 'smoke', 'floor', 'x', 'z']).any():
        raise ValueError('Duplicate cell identities; values will not be averaged')
    expected = {(g, s, f) for g in guidances for s in SMOKES for f in FLOORS}
    if set(frame.groupby(['guidance', 'smoke', 'floor']).groups) != expected:
        raise ValueError(f'Expected all {len(expected)} guidance / smoke / floor combinations: {guidances}')
    for name in ['x', 'z']:
        k = np.rint(frame[name].to_numpy() / CELL_M - .5)
        if not np.allclose(frame[name].to_numpy(), (k + .5) * CELL_M, atol=1e-6, rtol=0):
            raise ValueError('Input is not the expected 0.4m cell-centre grid')
    return frame


def cell_faces(part):
    """Four corners per observed cell; all corners keep its original height."""
    x = -part.x.to_numpy(float)
    y = -part.z.to_numpy(float)
    value = part[FIELD].to_numpy(float)
    corners = CELL_M / 2 * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    faces = np.empty((len(part), 4, 3), float)
    faces[:, :, 0] = x[:, None] + corners[None, :, 0]
    faces[:, :, 1] = y[:, None] + corners[None, :, 1]
    faces[:, :, 2] = value[:, None]
    return faces, value


def bar_faces(part, base):
    """Six closed faces per raw cell: top, bottom, then four vertical sides.

    The output is cell-major: reshape to (N, 6, 4, 3). Every face carries
    its cell's original utility for colour, regardless of the face's height.
    The display base must sit strictly below all original signed values.
    """
    top, values = cell_faces(part)
    if not np.isfinite(base) or (len(values) and base >= values.min()):
        raise ValueError('The display base must be finite and below all cell tops')
    bottom = top.copy()
    bottom[:, :, 2] = base
    faces = np.empty((len(part), 6, 4, 3), dtype=float)
    faces[:, 0] = top
    faces[:, 1] = bottom[:, ::-1]
    for side in range(4):
        following = (side + 1) % 4
        faces[:, side + 2] = np.stack(
            [bottom[:, side], bottom[:, following], top[:, following], top[:, side]], axis=1)
    return faces.reshape(-1, 4, 3), np.repeat(values, 6)


def smooth_display_values(part, iterations=0, strength=.5):
    """Convex display-only diffusion on four-connected occupied cells.

    Each step mixes the previous value with its valid edge-neighbour mean.
    Missing cells are never introduced; isolated cells are unchanged. No
    propagation occurs across an empty cell, a diagonal contact or another panel.
    """
    if not isinstance(iterations, (int, np.integer)) or iterations < 0:
        raise ValueError('Smoothing iterations must be a nonnegative integer')
    if not np.isfinite(strength) or not 0 < strength <= 1:
        raise ValueError('Smoothing strength must be in (0, 1]')
    values = part[FIELD].to_numpy(float).copy()
    if not np.isfinite(values).all():
        raise ValueError('Smoothing input values must be finite')
    if not iterations or not len(values):
        return values
    keys = np.rint(part[['x', 'z']].to_numpy(float) / CELL_M - .5).astype(np.int64)
    lookup = {tuple(key): index for index, key in enumerate(keys)}
    if len(lookup) != len(values):
        raise ValueError('Smoothing requires unique occupied grid cells')
    neighbours = np.array([[lookup.get((ix+dx, iz+dz), -1)
                            for dx, dz in ((-1, 0), (1, 0), (0, -1), (0, 1))]
                           for ix, iz in keys], dtype=np.int64)
    valid = neighbours >= 0
    degree = valid.sum(axis=1)
    occupied_neighbours = degree > 0
    indices = np.maximum(neighbours, 0)
    for _ in range(iterations):
        means = (values[indices] * valid).sum(axis=1) / np.maximum(degree, 1)
        updated = values.copy()
        updated[occupied_neighbours] = ((1-strength)*values[occupied_neighbours]
                                        + strength*means[occupied_neighbours])
        values = updated
    return values


class PiecewiseQuantileNorm(Normalize):
    """Continuous two-segment scale with fixed endpoints and a quantile hinge.

    Only the low tail is compressed. Values above the hinge use one constant
    linear slope, preserving their numerical spacing. Masks and scalar inputs
    behave like Normalize; clip=False extrapolates outside the endpoint range.
    """
    def __init__(self, vmin, vmax, quantile_value, target_position, clip=False):
        if not np.isfinite([vmin, vmax, quantile_value, target_position]).all():
            raise ValueError('Piecewise colour anchors must be finite')
        if not vmin < quantile_value < vmax or not 0 < target_position < 1:
            raise ValueError('Piecewise colour hinge must lie strictly inside both ranges')
        self.quantile_value = float(quantile_value)
        self.target_position = float(target_position)
        super().__init__(vmin=float(vmin), vmax=float(vmax), clip=clip)

    def __call__(self, value, clip=None):
        result, scalar = self.process_value(value)
        data = result.data
        if self.clip if clip is None else clip:
            data = np.clip(data, self.vmin, self.vmax)
        q, t = self.quantile_value, self.target_position
        mapped = np.where(data <= q,
                          t * (data-self.vmin) / (q-self.vmin),
                          t + (1-t) * (data-q) / (self.vmax-q))
        mapped = np.ma.array(mapped, mask=np.ma.getmask(result), copy=False)
        return mapped[0] if scalar else mapped

    def inverse(self, value):
        result, scalar = self.process_value(value)
        data = result.data
        q, t = self.quantile_value, self.target_position
        restored = np.where(data <= t,
                            self.vmin + data * (q-self.vmin) / t,
                            q + (data-t) * (self.vmax-q) / (1-t))
        restored = np.ma.array(restored, mask=np.ma.getmask(result), copy=False)
        return restored[0] if scalar else restored


def make_color_norm(values, channel, contrast_channels=(), contrast_quantile=.3, contrast_position=.1,
                    contrast_method='power'):
    """One raw-extrema norm per channel, optionally anchoring a chosen quantile."""
    if contrast_method not in ('power', 'piecewise'):
        raise ValueError('Unknown contrast method: ' + str(contrast_method))
    if not 0 < contrast_quantile < 1 or not 0 < contrast_position < 1:
        raise ValueError('Contrast quantile and target position must be strictly between 0 and 1')
    values = np.asarray(values, dtype=float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError('Colour normalization requires finite, nonempty raw values')
    minimum, maximum = float(values.min()), float(values.max())
    metadata = dict(kind='linear', vmin=minimum, vmax=maximum, gamma=1.0,
                    quantile=None, quantile_value=None, target_position=None,
                    normalization_scope='Per channel; all raw occupied cells across both smoke levels and floors')
    if channel in contrast_channels:
        if minimum == maximum:
            raise ValueError('Cannot anchor a quantile in a constant field: ' + channel)
        value = float(np.quantile(values, contrast_quantile))
        position = (value-minimum)/(maximum-minimum)
        if not 0 < position < 1:
            raise ValueError('Requested quantile coincides with a raw endpoint: ' + channel)
        gamma = None
        if contrast_method == 'power':
            gamma = float(np.log(contrast_position)/np.log(position))
            if not np.isfinite(gamma) or gamma <= 0:
                raise ValueError('Invalid power-normalization exponent: ' + channel)
        metadata.update(kind=contrast_method, gamma=gamma, quantile=float(contrast_quantile),
                        quantile_value=value, target_position=float(contrast_position))
    elif minimum == maximum:
        eps = max(1e-6, abs(minimum)*1e-6)
        metadata.update(vmin=minimum-eps, vmax=maximum+eps)
    return norm_from_metadata(metadata), metadata


def norm_from_metadata(metadata):
    """Rebuild the exact colour transform for independent colour bars and tables."""
    minimum, maximum = float(metadata['vmin']), float(metadata['vmax'])
    if not np.isfinite([minimum, maximum]).all() or minimum >= maximum:
        raise ValueError('Colour norm needs finite increasing endpoints')
    if metadata['kind'] == 'piecewise':
        return PiecewiseQuantileNorm(minimum, maximum, metadata['quantile_value'],
                                     metadata['target_position'], clip=False)
    if metadata['kind'] == 'power':
        gamma = float(metadata['gamma'])
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError('Power norm gamma must be finite and positive')
        return PowerNorm(gamma=gamma, vmin=minimum, vmax=maximum, clip=False)
    if metadata['kind'] != 'linear':
        raise ValueError('Unknown colour norm kind: ' + str(metadata['kind']))
    return Normalize(minimum, maximum)


def continuous_faces(part, smoothing_iterations=0, smoothing_strength=.5):
    """Four planar triangles per raw cell, without crossing unobserved cells.

    With smoothing disabled, original centre heights stay exact. Optional
    four-neighbour diffusion changes only display heights. Each grid corner has
    one shared height, the arithmetic mean of displayed cells touching it. Interpolation occurs
    only inside a cell's existing 0.4 m footprint. The output is cell-major:
    reshape to (N, 4, 3, 3); each triangle's first vertex is that cell's displayed
    centre. Every triangle keeps the source cell's original utility for colour.
    """
    centres = part[['x', 'z']].to_numpy(float)
    values = part[FIELD].to_numpy(float)
    display_values = smooth_display_values(part, smoothing_iterations, smoothing_strength)
    keys = np.rint(centres / CELL_M - .5).astype(np.int64)
    # Integer keys make shared corners bit-identical, including around holes.
    offsets = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.int64)
    corner_keys = keys[:, None, :] + offsets[None, :, :]
    unique_keys, inverse = np.unique(corner_keys.reshape(-1, 2), axis=0, return_inverse=True)
    sums = np.bincount(inverse, weights=np.repeat(display_values, 4), minlength=len(unique_keys))
    counts = np.bincount(inverse, minlength=len(unique_keys))
    shared_heights = sums / counts
    corners = np.empty((len(part), 4, 3), dtype=float)
    corners[:, :, :2] = -corner_keys * CELL_M
    corners[:, :, 2] = shared_heights[inverse].reshape(-1, 4)
    centre_vertices = np.column_stack([-centres, display_values])
    faces = np.empty((len(part), 4, 3, 3), dtype=float)
    faces[:, :, 0] = centre_vertices[:, None, :]
    faces[:, :, 1] = corners
    faces[:, :, 2] = np.roll(corners, -1, axis=1)
    return faces.reshape(-1, 3, 3), np.repeat(values, 4)


def geometry_policy(surface_style, smoothing_iterations=0, smoothing_strength=.5):
    """Describe display geometry separately from unchanged raw field values."""
    if surface_style not in SURFACE_STYLES:
        raise ValueError('Unknown surface style: ' + str(surface_style))
    common = dict(raw_cell_centres_exact=not smoothing_iterations, raw_values_colour=True, original_footprint_preserved=True,
                  missing_cells_filled=False, colour_source='Original source-cell utility for every face',
                  lighting=False)
    if surface_style == 'bars':
        return dict(common, primitive='Closed six-face column per observed cell', faces_per_cell=6,
                    base='channel raw minimum - 0.08 * raw range; same display base as original cells',
                    height='Flat column top equals original signed utility; base is a display reference, not zero')
    if surface_style == 'continuous':
        return dict(common, primitive='Four planar triangles per observed cell', faces_per_cell=4,
                    corners='Arithmetic mean of display heights of cells sharing the corner; no missing-cell values inserted',
                    continuity='Shared corner and edge heights; full occupied footprint preserved',
                    smoothing_iterations=int(smoothing_iterations), smoothing_strength=float(smoothing_strength),
                    smoothing_method='Four-connected occupied-cell convex neighbour diffusion; isolated cells unchanged',
                    height=('Display-only smoothed centre heights and local planar interpolation; original values unchanged'
                            if smoothing_iterations else 'Local piecewise-linear display interpolation; original centre heights exact'))
    return dict(common, primitive='One flat top face per observed cell', faces_per_cell=1,
                height='Original signed utility at every top-face vertex')


def boundary_segments(part, base_height):
    """Outline the observed cell footprint, including holes, without assets."""
    keys = np.rint(part[['x', 'z']].to_numpy(float) / CELL_M - .5).astype(np.int64)
    occupied = set(map(tuple, keys))
    segments = []
    # Raw Unity cell edges, then invert X/Z consistently with existing figures.
    for ix, iz in keys:
        x, z = (ix + .5) * CELL_M, (iz + .5) * CELL_M
        h = CELL_M / 2
        edges = [((-1, 0), ((x-h, z-h), (x-h, z+h))),
                 ((1, 0), ((x+h, z-h), (x+h, z+h))),
                 ((0, -1), ((x-h, z-h), (x+h, z-h))),
                 ((0, 1), ((x-h, z+h), (x+h, z+h)))]
        for (dx, dz), edge in edges:
            if (ix + dx, iz + dz) not in occupied:
                segments.append([(-xx, -zz, base_height) for xx, zz in edge])
    return segments


def draw_panel(ax, part, cmap, norm, limits, bounds, floor='Hall', basemap_root=None,
               surface_style='cells', smoothing_iterations=0, smoothing_strength=.5,
               basemap_variant='walls'):
    if surface_style not in SURFACE_STYLES:
        raise ValueError('Unknown surface style: ' + str(surface_style))
    faces, values = cell_faces(part)
    lo, hi = limits
    span = hi - lo
    if span <= 0:
        span = max(1e-6, abs(lo) * .01)
    base = lo - .08 * span
    if basemap_root:
        ax.add_image(FloorBasemapImage(ax, basemap_root, floor, base, bounds, basemap_variant))
    else:
        ground = faces.copy()
        ground[:, :, 2] = base
        ax.add_collection3d(Poly3DCollection(ground, facecolors='#e9edf1', edgecolors='none', linewidths=0,
            antialiased=False, zsort='average'))
    if surface_style == 'bars':
        faces, values = bar_faces(part, base)
    elif surface_style == 'continuous':
        faces, values = continuous_faces(part, smoothing_iterations, smoothing_strength)
    surface = Poly3DCollection(faces, facecolors=cmap(norm(values)), edgecolors='none', linewidths=0,
        antialiased=False, zsort='average')
    # Explicit face colours preserve the cmap; no lighting changes them.
    ax.add_collection3d(surface)
    if not basemap_root:
        outline = boundary_segments(part, base)
        if outline:
            ax.add_collection3d(Line3DCollection(outline, colors='#48515c', linewidths=.5))
    xmin, xmax, ymin, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(base, hi + .06 * span if hi > lo else hi + span)
    # Horizontal axes preserve the floor's physical aspect ratio. The relief
    # dimension is a display choice; its ticks retain the signed utility units.
    ax.set_box_aspect((xmax-xmin, ymax-ymin, 35))
    ax.view_init(elev=VIEW_ELEVATION, azim=VIEW_AZIMUTH)
    ax.set_proj_type('ortho')
    ax.set_xlabel('X (m)', labelpad=6)
    ax.set_ylabel('Z (m)', labelpad=6)
    ax.set_zlabel('Relative utility', labelpad=8)
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(MaxNLocator(nbins=4))
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)
    ax.zaxis.set_major_locator(MaxNLocator(nbins=5))
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor('#eeeeee')
    ax.tick_params(labelsize=9, pad=1)
    ax.grid(False)
    return dict(cells=len(part),minimum=float(values.min()),maximum=float(values.max()))


def draw_panel_2d(ax, part, cmap, norm, bounds):
    """Top view of exactly the same raw cell faces and footprint as 3D."""
    faces, values = cell_faces(part)
    ax.add_collection(PolyCollection(
        faces[:, :, :2], facecolors=cmap(norm(values)), edgecolors='none',
        linewidths=0, antialiased=False))
    outline = boundary_segments(part, 0.0)
    if outline:
        ax.add_collection(LineCollection(
            np.asarray(outline)[:, :, :2], colors='#48515c', linewidths=.5))
    xmin, xmax, ymin, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X (m)', labelpad=6)
    ax.set_ylabel('Z (m)', labelpad=6)
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(MaxNLocator(nbins=4))
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)
    ax.tick_params(labelsize=9, pad=1)
    ax.grid(False)
    return dict(cells=len(part), minimum=float(values.min()), maximum=float(values.max()))


def render_guidance(frame, guidance, target, dimension='3d', dpi=DPI, surface_style='cells',
                    smoothing_iterations=0, smoothing_strength=.5, contrast_channels=(),
                    contrast_quantile=.3, contrast_position=.1, contrast_method='power', layout_profile='legacy',
                    basemap_root=None, basemap_variant='walls'):
    if dimension not in ('3d', '2d'):
        raise ValueError('Expected dimension 3d or 2d')
    if surface_style not in SURFACE_STYLES:
        raise ValueError('Unknown surface style: ' + str(surface_style))
    if smoothing_iterations and surface_style != 'continuous':
        raise ValueError('Display smoothing is supported only for continuous surfaces')
    subset = frame.loc[frame.guidance.eq(guidance)]
    minimum, maximum = float(subset[FIELD].min()), float(subset[FIELD].max())
    norm, normalization = make_color_norm(subset[FIELD].to_numpy(), guidance, contrast_channels,
                                         contrast_quantile, contrast_position, contrast_method)
    cmap = (plt.get_cmap('cividis') if guidance == 'StaticSign' else
            LinearSegmentedColormap.from_list('guide_'+guidance, PALETTES[guidance], N=256))
    bounds = {}
    for floor in FLOORS:
        part = subset.loc[subset.floor.eq(floor)]
        bounds[floor] = (-float(part.x.max())-.2, -float(part.x.min())+.2,
                         -float(part.z.max())-.2, -float(part.z.min())+.2)
    layout = layout_metadata(layout_profile if dimension == '3d' else 'legacy')
    publication = dimension == '3d' and layout_profile == 'publication'
    fig = plt.figure(figsize=FIGSIZE)
    fig.subplots_adjust(left=.03, right=.86, top=.91, bottom=.09, wspace=.03, hspace=.32)
    panels = {}
    panel_axes = []
    rendered_layout = {}
    try:
        for row, smoke in enumerate(SMOKES):
            for col, floor in enumerate(FLOORS):
                options = {'projection': '3d'} if dimension == '3d' else {}
                ax = (fig.add_axes(layout['panel_axes'][row*2+col], **options) if publication else
                      fig.add_subplot(2, 2, row*2+col+1, **options))
                panel_axes.append(ax)
                part = subset.loc[subset.smoke.eq(smoke)&subset.floor.eq(floor)]
                if dimension == '3d':
                    panels[smoke+'/'+floor] = draw_panel(ax, part, cmap, norm, (minimum, maximum), bounds[floor],
                                                       floor, basemap_root, surface_style, smoothing_iterations,
                                                       smoothing_strength, basemap_variant)
                else:
                    panels[smoke+'/'+floor] = draw_panel_2d(ax, part, cmap, norm, bounds[floor])
                if publication:
                    # Compact panels place outward axis labels over adjacent
                    # axes rectangles. Transparent patches prevent later panels
                    # from hiding those labels; the figure itself stays white.
                    ax.patch.set_alpha(0)
                    xmin, xmax, ymin, ymax = bounds[floor]
                    ax.set_box_aspect((xmax-xmin, ymax-ymin, 35), zoom=PUBLICATION_ZOOM)
                    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                        axis.label.set_fontsize(layout['axis_label_fontsize'])
                    ax.tick_params(labelsize=layout['axis_tick_fontsize'], pad=1)
                    if floor == 'Hall':
                        # Larger labels need clearance on Hall's slanted X axis.
                        ax.xaxis.labelpad = layout['hall_x_labelpad']
                    # Matplotlib's z dimension is utility, not the floor's Z
                    # coordinate (which is plotted on y). Keep the relief and
                    # colour bar; suppress only the vertical numeric axis.
                    ax.set_zticks([])
                    ax.set_zticks([], minor=True)
                    ax.set_zlabel('')
                    ax.zaxis.set_visible(False)
                    ax.zaxis.line.set_visible(False)
                    ax.zaxis.label.set_visible(False)
                    ax.zaxis.offsetText.set_visible(False)
                ax.set_title(('High smoke' if smoke=='HighSmoke' else 'Low smoke')+' | '+floor,
                             fontsize=26 if publication else 13, pad=0 if publication else 14,
                             fontweight='bold' if publication else 'normal')
        cax = fig.add_axes(layout['colorbar_axes'])
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                          orientation='horizontal' if publication else 'vertical')
        if not publication:
            bar.set_label('Relative guidance-information demand (reward contrast)',
                          fontsize=11, labelpad=12)
        else:
            # Publication figures intentionally keep only the numerical ticks;
            # the former bottom annotation is not part of the final artwork.
            bar.ax.xaxis.label.set_visible(False)
            bar.ax.xaxis.label.set_fontsize(layout['colorbar_label_fontsize'])
        if publication:
            bar.ax.tick_params(labelsize=layout['colorbar_tick_fontsize'])
        bar.set_ticks(norm.inverse(np.linspace(0, 1, 6)) if normalization['kind'] != 'linear'
                      else np.linspace(norm.vmin, norm.vmax, 6))
        fig.suptitle(GUIDANCE_LABELS[guidance] if publication else
                     GUIDANCE_LABELS[guidance]+' | '+dimension.upper()+' spatial guidance-information demand',
                     fontsize=36 if publication else 18, y=layout['title_y'],
                     fontweight='bold' if publication else 'normal')
        encoding = 'height and colour' if dimension == '3d' else 'colour'
        if dimension == '3d' and surface_style == 'bars':
            footer = ('X/Z = -Unity X/Z; solid 0.4 m cells; top and colour = signed utility. '
                      'One colour range across both smoke levels and floors.')
        elif dimension == '3d' and surface_style == 'continuous':
            footer = (('X/Z = -Unity X/Z; mildly smoothed display mesh; colours use raw cell values. '
                       if smoothing_iterations else
                       'X/Z = -Unity X/Z; local continuous mesh; raw centre heights and cell colours. ')
                      + 'One colour range across both smoke levels and floors.')
        else:
            footer = ('X/Z = -Unity X/Z; raw 0.4 m cells; '+encoding+' = signed utility. '
                      'One colour range across both smoke levels and floors.')
        if not publication:
            fig.text(.5, .03, footer, ha='center', fontsize=11)
        fig.savefig(target, dpi=dpi, facecolor='white')
        if publication:
            actual_boxes = [list(ax.get_position().bounds) for ax in panel_axes]
            if not np.allclose(actual_boxes, layout['panel_axes'], rtol=0, atol=1e-12):
                raise RuntimeError('Rendered panel viewports differ from the fixed-size layout')
            rendered_layout = dict(panel_axes=actual_boxes,
                panel_size_inches=[[box[2]*FIGSIZE[0], box[3]*FIGSIZE[1]] for box in actual_boxes],
                title=fig._suptitle.get_text(), title_fontsize=fig._suptitle.get_fontsize(),
                title_fontweight=fig._suptitle.get_fontweight(),
                panel_title_fontsizes=[ax.title.get_fontsize() for ax in panel_axes],
                panel_title_fontweights=[ax.title.get_fontweight() for ax in panel_axes],
                axis_label_fontsizes=[[axis.label.get_fontsize() for axis in (ax.xaxis, ax.yaxis, ax.zaxis)]
                                      for ax in panel_axes],
                axis_tick_fontsizes=[[tick.get_fontsize() for axis in (ax.xaxis, ax.yaxis, ax.zaxis)
                                      for tick in axis.get_ticklabels()] for ax in panel_axes],
                colorbar_orientation=bar.orientation, colorbar_axes=list(cax.get_position().bounds),
                colorbar_tick_fontsizes=[tick.get_fontsize() for tick in bar.ax.get_xticklabels()],
                colorbar_label_fontsize=bar.ax.xaxis.label.get_fontsize(),
                colorbar_label_visible=bool(bar.ax.xaxis.label.get_visible()),
                panel_background_alpha=[ax.patch.get_alpha() for ax in panel_axes],
                vertical_utility_axes=[dict(axis_visible=ax.zaxis.get_visible(),
                    line_visible=ax.zaxis.line.get_visible(), label_visible=ax.zaxis.label.get_visible(),
                    label=ax.zaxis.label.get_text(), ticks=ax.get_zticks().tolist(),
                    minor_ticks=ax.get_zticks(minor=True).tolist(),
                    offset_visible=ax.zaxis.offsetText.get_visible(), pane_visible=ax.zaxis.pane.get_visible())
                    for ax in panel_axes],
                figure_texts=[text.get_text() for text in fig.texts])
    finally:
        plt.close(fig)
    result = dict(file=target.name,dimension=dimension,palette_number=PALETTE_NUMBERS[guidance],
        palette_name=cmap.name,palette_reversed=False,
        data_minimum=minimum,data_maximum=maximum,cmap_minimum=float(norm.vmin),cmap_maximum=float(norm.vmax),
        color_normalization=normalization,panels=panels,
        basemap_enabled=bool(basemap_root) and dimension == '3d', colorbar_label_visible=not publication)
    if publication:
        result['rendered_layout'] = rendered_layout
    return result


def plot_field(*, run=None, input_path=None, output=None, dpi=DPI, force=False, surface_style='cells',
               smoothing_iterations=0, smoothing_strength=.5, contrast_channels=(),
               contrast_quantile=.3, contrast_position=.1, contrast_method='power', layout_profile='legacy',
               basemap_root=None, basemap_variant='walls'):
    if not isinstance(dpi, int) or dpi < 50:
        raise ValueError('DPI must be an integer >= 50')
    if surface_style not in SURFACE_STYLES:
        raise ValueError('Unknown surface style: ' + str(surface_style))
    layout_metadata(layout_profile)
    basemap_info = basemap_metadata(basemap_root, basemap_variant)
    if not isinstance(smoothing_iterations, (int, np.integer)) or smoothing_iterations < 0:
        raise ValueError('Smoothing iterations must be a nonnegative integer')
    if not np.isfinite(smoothing_strength) or not 0 < smoothing_strength <= 1:
        raise ValueError('Smoothing strength must be in (0, 1]')
    if smoothing_iterations and surface_style != 'continuous':
        raise ValueError('Display smoothing is supported only for continuous surfaces')
    if isinstance(contrast_channels, str):
        contrast_channels = tuple(c.strip() for c in contrast_channels.split(',') if c.strip())
    if set(contrast_channels) - set(GUIDANCES):
        raise ValueError('Unknown contrast channel')
    contrast_channels = tuple(c for c in GUIDANCES if c in contrast_channels)
    processing = dict(smoothing_iterations=int(smoothing_iterations), smoothing_strength=float(smoothing_strength),
                      contrast_channels=list(contrast_channels), contrast_quantile=float(contrast_quantile),
                      contrast_position=float(contrast_position), contrast_method=contrast_method)
    source, required_guidances = resolve_input(run=run, input_path=input_path)
    frame = load_data(source, required_guidances)
    guidances = tuple(g for g in GUIDANCES if frame.guidance.eq(g).any())
    if set(contrast_channels) - set(guidances):
        raise ValueError('Requested contrast channel is absent from input')
    # Validate all requested norms before touching an existing output directory.
    normalizations = {g: make_color_norm(frame.loc[frame.guidance.eq(g), FIELD].to_numpy(), g,
                                       contrast_channels, contrast_quantile, contrast_position, contrast_method)[1]
                      for g in guidances}
    output = Path(output).resolve() if output is not None else source.parent / OUTPUT_FOLDERS[surface_style]
    source_hash = sha256(source)
    script_hash = sha256(Path(__file__))
    manifest_path = output/'manifest_3d.json'
    expected = [g+'_spatial_demand_'+d+'.png' for g in guidances for d in ('3d', '2d')]
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding='utf-8'))
        if saved.get('surface_style', 'cells') != surface_style:
            raise FileExistsError('Existing output has a different surface style; choose a separate directory')
        if not force:
            if (saved.get('status')=='complete' and saved.get('input_sha256')==source_hash
                    and saved.get('script_sha256')==script_hash and saved.get('dpi')==dpi
                    and saved.get('guidances')==list(guidances)
                    and saved.get('layout_profile', 'legacy')==layout_profile
                    and saved.get('basemap')==json.loads(json.dumps(basemap_info))
                    and {'contrast_method': 'power', **saved.get('display_processing', {})}==processing
                    and all((output/n).exists() for n in expected)):
                print('Already complete: '+str(output))
                return saved
            raise FileExistsError('Existing 3D/2D output differs or is incomplete; rerun with --force')
    if not force and output.exists() and any((output/n).exists() for n in expected):
        raise FileExistsError('Existing 3D/2D images; rerun with --force')
    output.mkdir(parents=True, exist_ok=True)
    # A failed --force rerender must not leave an old "complete" manifest.
    manifest_path.write_text(json.dumps(dict(status='rendering', input_path=str(source), surface_style=surface_style,
                                            display_processing=processing), indent=2), encoding='utf-8')
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':11})
    results = {}
    results_2d = {}
    for guidance in tqdm(guidances, desc='Render paired 3D/2D guidance maps'):
        print('Render '+guidance+' 3D + 2D', flush=True)
        options = dict(surface_style=surface_style, smoothing_iterations=smoothing_iterations,
                       smoothing_strength=smoothing_strength, contrast_channels=contrast_channels,
                       contrast_quantile=contrast_quantile, contrast_position=contrast_position,
                       contrast_method=contrast_method, layout_profile=layout_profile,
                       basemap_root=basemap_root, basemap_variant=basemap_variant)
        results[guidance] = render_guidance(frame, guidance, output/(guidance+'_spatial_demand_3d.png'), '3d', dpi, **options)
        results_2d[guidance] = render_guidance(frame, guidance, output/(guidance+'_spatial_demand_2d.png'), '2d', dpi, **options)
    if sha256(source) != source_hash:
        raise RuntimeError('Source CSV changed during rendering')
    provenance = dict(PALETTE_PROVENANCE)
    if 'StaticSign' in guidances:
        provenance['StaticSign'] = dict(source='Matplotlib built-in cividis', matplotlib_version=matplotlib.__version__, reversed=False)
    manifest = dict(status='complete',input_filename=source.name,input_path=str(source),input_sha256=source_hash,script_sha256=script_hash,
        guidances=list(guidances),field=FIELD,palette_provenance=provenance,cell_m=CELL_M,dpi=dpi,
        surface_style=surface_style,geometry_policy=geometry_policy(surface_style, smoothing_iterations, smoothing_strength),
        display_processing=processing,color_normalization=normalizations,
        layout_profile=layout_profile, layout_3d=layout_metadata(layout_profile),
        basemap=basemap_info,
        layout=dict(figsize_inches=list(FIGSIZE), grid=[2, 2],
                    row_order=list(SMOKES), column_order=list(FLOORS),
                    view_elevation=VIEW_ELEVATION, view_azimuth=VIEW_AZIMUTH, projection='ortho',
                    subplot_adjust=dict(left=.03, right=.86, top=.91, bottom=.09, wspace=.03, hspace=.32),
                    colorbar_axes=[.90, .19, .018, .60], title_y=.97, footer_y=.03,
                    horizontal_aspect='Physical X/Z ratio', relief_display_aspect=35),
        rows_rendered=len(frame),figures=results,figures_2d=results_2d,
        png_count=2*len(guidances),png_count_3d=len(guidances),png_count_2d=len(guidances),pdf_count=0,raw_data_modified=False,
        smoothing=bool(smoothing_iterations),interpolation=(surface_style == 'continuous'),clipping=False,subsampling=False,
        height=('Original signed reward contrast at every cell; not physical building elevation' if surface_style == 'cells' else
                'Original signed reward contrast at column tops; common display base below raw minimum, not zero' if surface_style == 'bars' else
                'Neighbour-diffused display heights; shared corner means and local planar interpolation; original field unchanged' if smoothing_iterations else
                'Original signed reward contrast at cell centres; shared corner means and local planar interpolation for display only'),
        display_coordinates='(-Unity X, -Unity Z)',cmap_scope='Per guidance, raw extrema across both smoke levels and floors',
        missing_cell_policy='Empty; no zero-fill or extrapolation',
        paired_2d_policy=('Same cell corners, values, cmap, limits and footprint as 3D; equal physical X/Z aspect' if surface_style == 'cells' else
                          'Unchanged raw-cell 2D; same original values, cmap, limits and footprint as 3D; equal physical X/Z aspect'))
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('OUTPUT='+str(output))
    for guidance, result in results.items():
        print(f"{guidance}: min={result['data_minimum']:.8g}, max={result['data_maximum']:.8g}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run', type=Path, help='Completed Best_Model directory (V6 or V3 Optimize/V4)')
    source.add_argument('--input', type=Path, help='Standalone exported spatial field CSV, optionally gzip-compressed')
    parser.add_argument('--output', type=Path, help='Defaults to the surface-style folder beside the CSV')
    parser.add_argument('--surface-style', choices=SURFACE_STYLES, default='cells',
                        help='Original flat cells, closed solid columns, or locally continuous display mesh')
    parser.add_argument('--layout-profile', choices=LAYOUT_PROFILES, default='legacy',
                        help='Publication 3D layout keeps canvas/DPI/panel size; 2D remains unchanged')
    parser.add_argument('--basemap-root', type=Path,
                        help='Directory containing the selected pair of Unity scene PNGs')
    parser.add_argument('--basemap-variant', choices=tuple(BASEMAP_VARIANTS), default='walls',
                        help='walls: Image Sequence 007/006; no-walls: Concourse/Platform_Unity_Camera')
    parser.add_argument('--smooth-iterations', type=int, default=0,
                        help='Display-only four-neighbour diffusion steps; continuous surfaces only (suggested: 4)')
    parser.add_argument('--smooth-strength', type=float, default=.5,
                        help='Neighbour-mean mixing weight per diffusion step, in (0,1]')
    parser.add_argument('--contrast-channels', default='',
                        help='Comma-separated channels for quantile-anchored colour mapping')
    parser.add_argument('--contrast-method', choices=('power', 'piecewise'), default='power',
                        help='Power transform or two linear segments with a low-tail quantile hinge')
    parser.add_argument('--contrast-quantile', type=float, default=.3, help='Raw quantile anchor (default: P30)')
    parser.add_argument('--contrast-position', type=float, default=.1, help='Palette position of the quantile anchor')
    parser.add_argument('--dpi', type=int, default=DPI)
    parser.add_argument('--force', action='store_true', help='Replace this script\'s existing 3D and paired 2D outputs')
    args = parser.parse_args()
    plot_field(run=args.run, input_path=args.input, output=args.output, dpi=args.dpi, force=args.force,
               surface_style=args.surface_style, smoothing_iterations=args.smooth_iterations,
               smoothing_strength=args.smooth_strength, contrast_channels=args.contrast_channels,
               contrast_quantile=args.contrast_quantile, contrast_position=args.contrast_position,
               contrast_method=args.contrast_method, layout_profile=args.layout_profile,
               basemap_root=args.basemap_root, basemap_variant=args.basemap_variant)


if __name__=='__main__':
    main()
