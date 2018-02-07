# -*- coding: utf-8 -*-
# Copyright (c) 2017 - for information on the respective copyright owner
# see the NOTICE file and/or the repository https://github.com/VolkerFischer/statestream
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import numpy as np

from statestream.utils.pygame_import import pg, pggfx
from statestream.utils.defaults import DEFAULT_COLORS



# =============================================================================
# =============================================================================
# =============================================================================

def brighter(col, factor=0.5):
    """Function returns a brightened (by factor) color.
    """
    c0 = int(np.clip(col[0] + factor * (255 - col[0]), 0, 255))
    c1 = int(np.clip(col[1] + factor * (255 - col[1]), 0, 255))
    c2 = int(np.clip(col[2] + factor * (255 - col[2]), 0, 255))
    return (c0, c1, c2)

def darker(col, factor=0.5):
    """Function returns a darkened (by factor) color.
    """
    c0 = int(col[0] * factor)
    c1 = int(col[1] * factor)
    c2 = int(col[2] * factor)
    return (c0, c1, c2)

# =============================================================================
# =============================================================================
# =============================================================================

def cc(col, ccol):
    """Color correction for pygame visualization.
    """
    if ccol:
        return (col[0]/2, col[1], col[2])
    else:
        return (col[2], col[1], col[0])

# =============================================================================
# =============================================================================
# =============================================================================

def num2str(x):
    """Converter for numeras to nice short strings.
    """
    n = ""
    if x >= 0:
        n += "+"
    else:
        n += ""
    n += "{:0.1e}".format(x)
    return n.replace("e", "")

# =============================================================================
# =============================================================================
# =============================================================================

def plot_circle(screen, x, y, rad, border_col=None, fill_col=None, border_width=0):
    """Try to draw an anti-aliased circle to the screen.
    
    Parameters
    ----------
    screen : pygame.Surface
        Description ...
    x : float, int
    y : float, int
    rad : float, int
    border_col : 3 tuple
    fill_col : 3 tuple
    border_width : float, int
    
    Returns
    -------
    rect : pygame.Rect
    """
    rect = None
    # Set default filling color to black.
    fill_col_loc = fill_col
    border_col_loc = border_col
    if fill_col is None:
        fill_col_loc = (0,0,0)
    if border_col is None:
        border_col_loc = (0,0,0)
    # Blit circle if rad is big enough.
    if rad >= border_width:
        if pggfx is not None:
            rect = pg.draw.circle(screen,(0,0,0), (x, y), int(rad), 0)
            if border_width != 0:
                pggfx.aacircle(screen, x, y, rad, border_col_loc)
                pggfx.filled_circle(screen, x, y, rad, border_col_loc)
            pggfx.aacircle(screen, x, y, int(rad - border_width), fill_col_loc)
            pggfx.filled_circle(screen, 
                                x, 
                                y, 
                                int(rad - border_width), 
                                fill_col_loc)
        else:
            rect = pg.draw.circle(screen, 
                                  border_col_loc, 
                                  (x, y), 
                                  rad - border_width // 2, 
                                  border_width)
            if fill_col_loc is not None:
                pg.draw.circle(screen, 
                               fill_col_loc, 
                               (x, y), 
                               rad - border_width, 
                               0)
    return rect

# =============================================================================
# =============================================================================
# =============================================================================

def blit_plot(surf, 
              font, 
              sw_size, 
              x, 
              y, 
              ccol, 
              color=None,
              y_var=None, 
              number_color=DEFAULT_COLORS["orange"],
              legend=None,
              scale='combined',
              axes=True,
              frame=0):
    """Blits a curve plot onto a pygame surface.
    """
    # Check some constraints.
    if sw_size[0] < 120 or sw_size[1] < 120 or x.shape[0] != y.shape[0]:
        pass
    else:
        zX = 0
        zY = sw_size[1]
        if axes:
            zX = 18
            zY = sw_size[1] - 18
            # Blit x/y axes.
            pg.draw.line(surf, cc(DEFAULT_COLORS["light"], ccol), (zX, 0), (zX, sw_size[1] - 1), 2)
            pg.draw.line(surf, cc(DEFAULT_COLORS["light"], ccol), (0, zY), (sw_size[0] - 1, zY), 2)
        # Blit frame.
        if frame > 0:
            pg.draw.rect(surf, cc((196,196,196), ccol), 
                         [0, 0, sw_size[0], sw_size[1]], 
                         int(frame))
        # Points are distributed equally over the x axis (even if not true).
        points = x.shape[0]
        dx = (sw_size[0] - zX) / points
        # Add plot axis to y.
        if len(y.shape) == 1:
            y = y[:,np.newaxis]
        if y_var is not None:
            if len(y_var.shape) == 1:
                y_var = y_var[:,np.newaxis]
        # Determine min/max for scaling.
        yMin = []
        yMax = []
        if scale == 'combined':
            if y_var is None:
                y_min = np.min(y.flatten())
                y_max = np.max(y.flatten())
            else:
                y_min = np.min(y.flatten() - y_var.flatten())
                y_max = np.max(y.flatten() + y_var.flatten())
            yMin = [y_min for i in range(y.shape[1])]
            yMax = [y_max for i in range(y.shape[1])]
        elif scale == 'separated':
            for p in range(y.shape[1]):
                if y_var is None:
                    yMin.append(np.min(y[:,p]))
                    yMax.append(np.max(y[:,p]))
                else:
                    yMin.append(np.min(y[:,p] - y_var[:,p]))
                    yMax.append(np.max(y[:,p] + y_var[:,p]))
                    
        for p in range(y.shape[1]):
            if yMin[p] == yMax[p]:
                # Draw flat line.
                if color is not None:
                    col = color[p]
                else:
                    col = (0,255,0)
                pg.draw.line(surf, 
                             cc(col, ccol), 
                             (int(zX + dx / 2), int(zY / 2)), 
                             (int(sw_size[0] - dx / 2), int(zY / 2)), 
                             2)
            else:
                # Create and draw point list.
                point_list = []
                for pts in range(points):
                    point_list.append([int(zX + dx / 2 + pts * dx), 
                                       int(zY * (1.0 - (y[pts,p] - yMin[p]) / max(yMax[p] - yMin[p], 1e-6)))])
                if color is not None:
                    col = color[p]
                else:
                    col = (0,255,0)
                # Draw variances.
                if y_var is not None:
                    for pts in range(points):
                        pg.draw.line(surf, cc(col, ccol),  
                                     [point_list[pts][0], 
                                      int(zY * (1.0 - (y[pts,p] + y_var[pts,p] - yMin[p]) / max(yMax[p] - yMin[p], 1e-6)))],
                                     [point_list[pts][0], 
                                      int(zY * (1.0 - (y[pts,p] - y_var[pts,p] - yMin[p]) / max(yMax[p] - yMin[p], 1e-6)))], 
                                     2)
                pg.draw.lines(surf, cc(col, ccol), False, point_list, 2)
            # blit x-axis time steps
        # Draw legend beside plot.
        if legend is not None:
            if len(legend) == y.shape[1]:
                dy = sw_size[1] // y.shape[1]
                for p in range(len(legend)):
                    if color is not None:
                        col = color[p]
                    else:
                        col = (0,255,0)
                    surf.blit(font.render(legend[p],
                                          1,
                                          cc(col, ccol)), (sw_size[0] - 60, p * dy))
        # Blit current value.
        if y.shape[1] == 1:
            if y.shape[1] == 1:
                surf.blit(font.render(num2str(y[points - 1,0]), 
                                      1, 
                                      cc(number_color, ccol)), (sw_size[0] - 80, int(zY / 2)))

# =============================================================================
# =============================================================================
# =============================================================================

def blit_pcp(surf, font, sw_size, x, ccol, number_color=DEFAULT_COLORS["orange"], dim=0, color=None):
    """Blits tensor x to surf using parallel coordinates.
    """
    if dim == 0:
        X = x.reshape([x.shape[0], np.prod(x.shape[1:])])
    elif dim == 1:
        X = np.swapaxes(x, 0, 1)
        X = X.reshape([X.shape[0], np.prod(X.shape[1:])])
    X = np.transpose(X)
    Y = np.arange(0, X.shape[0], 1, dtype=np.int64)
    blit_plot(surf, 
              font, 
              sw_size, 
              Y, 
              X, 
              ccol,
              color=color,
              axes=False,
              frame=2)

# =============================================================================
# =============================================================================
# =============================================================================

def blit_hist(surf, font, sw_size, x, ccol, number_color=DEFAULT_COLORS["orange"]):
    """Blits a histogram on a pygame surface.
    """
    # Check some constraints.
    if sw_size[0] < 120 or sw_size[1] < 120:
        pass
    else:
        zX = 18
        zY = sw_size[1] - 18
        # Blit x/y axes.
        pg.draw.line(surf, cc(DEFAULT_COLORS["light"], ccol), (zX, 0), (zX, sw_size[1] - 1), 2)
        pg.draw.line(surf, cc(DEFAULT_COLORS["light"], ccol), (0, zY), (sw_size[0] - 1, zY), 2)
        # Determine bins.
        dx = 12
        bins = int((sw_size[0] - zX) / dx)
        [hist_value, hist_edge] = np.histogram(x, bins=bins)
        max_val = np.max(hist_value)
        # Plot bars.
        for b in range(bins):
            dy = int(float(hist_value[b]) * float(zY) / float(max_val))
            pg.draw.rect(surf, cc((127,255,127), ccol), [zX + b * dx, zY - dy, dx, dy], 0)
        # Blit zero line.
        if hist_edge[0] < 0 and hist_edge[-1] > 0:
            zero_x = float(sw_size[0] - zX) * abs(hist_edge[0]) \
                     / (hist_edge[-1] - hist_edge[0])
            pg.draw.rect(surf, 
                         cc((32,32,127), ccol), 
                         [zX + int(zero_x) - 5, zY - 25, 10, 50], 
                         0)
        # Blit mean rect.
        mean_x = float(sw_size[0] - zX) * (np.mean(x.flatten()) - hist_edge[0]) \
                 / (hist_edge[-1] - hist_edge[0])
        pg.draw.rect(surf, 
                     cc((127,32,32), ccol), 
                     [zX + int(mean_x) - 10, zY - 10, 20, 20], 
                     0)
        # Blit min / max on x axis.
        surf.blit(font.render(num2str(hist_edge[0]), 
                              1, 
                              cc(number_color, ccol)), (zX, zY))
        surf.blit(font.render(num2str(hist_edge[-1]), 
                              1, 
                              cc(number_color, ccol)), (sw_size[0] - 90, zY))


# =============================================================================
# =============================================================================
# =============================================================================

def empty_subwin():
    """Create default empty subwindow structure.
    """
    SW = {}
    SW["type"] = ""
    SW["size"] = np.array([127, 127])
    SW["pos"] = np.array([0,0])
    SW["colormap"] = "magma"
    SW["anker"] = True
    SW['loc_idx'] = None
    SW['glob_idx'] = None
    SW["rect"] = pg.Rect(0, 0, 2, 2)
    SW["rect drag"] = pg.Rect(0, 0, 2, 2)
    SW["rect scale"] = pg.Rect(0, 0, 2, 2)
    SW["rect close"] = pg.Rect(0, 0, 2, 2)
    SW["rect mode"] = pg.Rect(0, 0, 2, 2)
    SW["rect tiles"] = pg.Rect(0, 0, 2, 2)
    SW["rect magic"] = pg.Rect(0, 0, 2, 2)
    SW["rect patterns"] = pg.Rect(0, 0, 2, 2)
    SW["rect tmem+"] = pg.Rect(0, 0, 2, 2)
    SW["rect tmem-"] = pg.Rect(0, 0, 2, 2)
    SW["go left"] = pg.Rect(0, 0, 2, 2)
    SW["go right"] = pg.Rect(0, 0, 2, 2)
    SW["cm flag"] = False
    SW["cm rect"] = pg.Rect(0, 0, 2, 2)
    SW["cm buffer"] = pg.Surface([1, 256])
    SW["cm buffer"].convert()
    SW["cm array"] = np.zeros([1, 256], dtype=np.int32)
    SW["manip flag"] = False
    SW["manip par"] = ""
    SW["manip rect"] = pg.Rect(0, 0, 2, 2)
    SW["manip dv"] = 0
    SW["manip px"] = 0
    SW["manip orig"] = 0
    SW["map"] = -1
    SW["tmem"] = -1
    SW["sub shape"] = [1, 1]
    SW["tiles"] = False
    SW["tileable"] = False
    SW["magicable"] = True
    SW["magic"] = ''
    SW["patterns"] = ''
    SW['legend left'] = []
    SW['legend right'] = []
    SW['legend left relative'] = False
    SW['legend right relative'] = False
    SW['legend left color'] = False
    SW['legend right color'] = False
    SW['min'] = None
    SW['max'] = None
    SW['area'] = None
    SW['axis x'] = ''
    SW['axis y'] = ''
    return SW

# =============================================================================
# =============================================================================
# =============================================================================
