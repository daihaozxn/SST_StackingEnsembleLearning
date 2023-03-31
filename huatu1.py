# 画 研究海域

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
import cartopy.geodesic as cgeo
from numpy import sin, cos, isclose, array, asarray, pi, linalg, floor, log10

# def _axes_to_lonlat(ax, coords):
#     """(lon, lat) from axes coordinates."""
#     display = ax.transAxes.transform(coords)
#     data = ax.transData.inverted().transform(display)
#     lonlat = ccrs.PlateCarree().transform_point(*data, ax.projection)
#
#     return lonlat
#
#
# def _upper_bound(start, direction, distance, dist_func):
#     """A point farther than distance from start, in the given direction.
#
#     It doesn't matter which coordinate system start is given in, as long
#     as dist_func takes points in that coordinate system.
#
#     Args:
#         start:     Starting point for the line.
#         direction  Nonzero (2, 1)-shaped array, a direction vector.
#         distance:  Positive distance to go past.
#         dist_func: A two-argument function which returns distance.
#
#     Returns:
#         Coordinates of a point (a (2, 1)-shaped NumPy array).
#     """
#     if distance <= 0:
#         raise ValueError(f"Minimum distance is not positive: {distance}")
#
#     if linalg.norm(direction) == 0:
#         raise ValueError("Direction vector must not be zero.")
#
#     # Exponential search until the distance between start and end is
#     # greater than the given limit.
#     length = 0.1
#     end = start + length * direction
#
#     while dist_func(start, end) < distance:
#         length *= 2
#         end = start + length * direction
#
#     return end
#
#
# def _distance_along_line(start, end, distance, dist_func, tol):
#     """Point at a distance from start on the segment  from start to end.
#
#     It doesn't matter which coordinate system start is given in, as long
#     as dist_func takes points in that coordinate system.
#
#     Args:
#         start:     Starting point for the line.
#         end:       Outer bound on point's location.
#         distance:  Positive distance to travel.
#         dist_func: Two-argument function which returns distance.
#         tol:       Relative error in distance to allow.
#
#     Returns:
#         Coordinates of a point (a (2, 1)-shaped NumPy array).
#     """
#     initial_distance = dist_func(start, end)
#     if initial_distance < distance:
#         raise ValueError(f"End is closer to start ({initial_distance}) than "
#                          f"given distance ({distance}).")
#
#     if tol <= 0:
#         raise ValueError(f"Tolerance is not positive: {tol}")
#
#     # Binary search for a point at the given distance.
#     left = start
#     right = end
#
#     while not isclose(dist_func(start, right), distance, rtol=tol):
#         midpoint = (left + right) / 2
#
#         # If midpoint is too close, search in second half.
#         if dist_func(start, midpoint) < distance:
#             left = midpoint
#         # Otherwise the midpoint is too far, so search in first half.
#         else:
#             right = midpoint
#
#     return right
#
#
# def _point_along_line(ax, start, distance, angle=0, tol=0.01):
#     """Point at a given distance from start at a given angle.
#
#     Args:
#         ax:       CartoPy axes.
#         start:    Starting point for the line in axes coordinates.
#         distance: Positive physical distance to travel.
#         angle:    Anti-clockwise angle for the bar, in radians. Default: 0
#         tol:      Relative error in distance to allow. Default: 0.01
#
#     Returns:
#         Coordinates of a point (a (2, 1)-shaped NumPy array).
#     """
#     # Direction vector of the line in axes coordinates.
#     direction = array([cos(angle), sin(angle)])
#
#     geodesic = cgeo.Geodesic()
#
#     # Physical distance between points.
#     def dist_func(a_axes, b_axes):
#         a_phys = _axes_to_lonlat(ax, a_axes)
#         b_phys = _axes_to_lonlat(ax, b_axes)
#
#         # Geodesic().inverse returns a NumPy MemoryView like [[distance,
#         # start azimuth, end azimuth]].
#         return geodesic.inverse(a_phys, b_phys).base[0, 0]
#
#     end = _upper_bound(start, direction, distance, dist_func)
#
#     return _distance_along_line(start, end, distance, dist_func, tol)
#
#
# def scale_bar(ax, location, length, metres_per_unit=1000, unit_name='km',
#               tol=0.01, angle=0, color='black', linewidth=3, text_offset=0.005,
#               ha='center', va='bottom', plot_kwargs=None, text_kwargs=None,
#               **kwargs):
#     """Add a scale bar to CartoPy axes.
#
#     For angles between 0 and 90 the text and line may be plotted at
#     slightly different angles for unknown reasons. To work around this,
#     override the 'rotation' keyword argument with text_kwargs.
#
#     Args:
#         ax:              CartoPy axes.
#         location:        Position of left-side of bar in axes coordinates.
#         length:          Geodesic length of the scale bar.
#         metres_per_unit: Number of metres in the given unit. Default: 1000
#         unit_name:       Name of the given unit. Default: 'km'
#         tol:             Allowed relative error in length of bar. Default: 0.01
#         angle:           Anti-clockwise rotation of the bar.
#         color:           Color of the bar and text. Default: 'black'
#         linewidth:       Same argument as for plot.
#         text_offset:     Perpendicular offset for text in axes coordinates.
#                          Default: 0.005
#         ha:              Horizontal alignment. Default: 'center'
#         va:              Vertical alignment. Default: 'bottom'
#         **plot_kwargs:   Keyword arguments for plot, overridden by **kwargs.
#         **text_kwargs:   Keyword arguments for text, overridden by **kwargs.
#         **kwargs:        Keyword arguments for both plot and text.
#     """
#     # Setup kwargs, update plot_kwargs and text_kwargs.
#     if plot_kwargs is None:
#         plot_kwargs = {}
#     if text_kwargs is None:
#         text_kwargs = {}
#
#     plot_kwargs = {'linewidth': linewidth, 'color': color, **plot_kwargs,
#                    **kwargs}
#     text_kwargs = {'ha': ha, 'va': va, 'rotation': angle, 'color': color,
#                    **text_kwargs, **kwargs}
#
#     # Convert all units and types.
#     location = asarray(location)  # For vector addition.
#     length_metres = length * metres_per_unit
#     angle_rad = angle * pi / 180
#
#     # End-point of bar.
#     end = _point_along_line(ax, location, length_metres, angle=angle_rad,
#                             tol=tol)
#
#     # Coordinates are currently in axes coordinates, so use transAxes to
#     # put into data coordinates. *zip(a, b) produces a list of x-coords,
#     # then a list of y-coords.
#     ax.plot(*zip(location, end), transform=ax.transAxes, **plot_kwargs)
#
#     # Push text away from bar in the perpendicular direction.
#     midpoint = (location + end) / 2
#     offset = text_offset * array([-sin(angle_rad), cos(angle_rad)])
#     text_location = midpoint + offset
#
#     # 'rotation' keyword argument is in text_kwargs.
#     ax.text(*text_location, f"{length} {unit_name}", rotation_mode='anchor',
#             transform=ax.transAxes, **text_kwargs)

def scale_bar(ax, length=None, location=(0.5, 0.05), linewidth=3):
    """
    ax is the axes to draw the scalebar on.
    length is the length of the scalebar in km.
    location is center of the scalebar in axis coordinates.
    (ie. 0.5 is the middle of the plot)
    linewidth is the thickness of the scalebar.
    """
    #Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    #Make tmc horizontally centred on the middle of the map,
    #vertically at scale bar location
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly)
    #Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    #Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    #Calculate a scale bar length if none has been given
    #(Theres probably a more pythonic way of rounding the number but this works)
    if not length:
        length = (x1 - x0) / 5000 #in km
        ndim = int(floor(log10(length))) #number of digits in number
        length = round(length, -ndim) #round to 1sf
        #Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ['1', '2', '5']: return int(x)
            else: return scale_number(x - 10 ** ndim)
        length = scale_number(length)

    #Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx - length * 500, sbx + length * 500]
    #Plot the scalebar
    ax.plot(bar_xs, [sby, sby], transform=tmc, color='k', linewidth=linewidth)
    #Plot the scalebar label
    ax.text(sbx, sby, str(length) + ' km', transform=tmc, fontsize=12,
            horizontalalignment='center', verticalalignment='bottom')

proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(15, 7))
ax = fig.subplots(1, 1, subplot_kw={'projection': proj})
# ax.coastlines()

ax.add_feature(cfeature.LAND.with_scale('10m')) # 添加陆地
ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.3)# 添加海岸线  高精度ax.add_feature(cfeature.COASTLINE.with_scale('10m'),lw=0.5)
ax.add_feature(cfeature.RIVERS.with_scale('10m'), linewidth=0.25)# 添加河流
ax.add_feature(cfeature.LAKES.with_scale('10m'))# 添加湖泊
ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle='-', linewidth=0.25)# 不推荐，我国丢失了藏南、台湾等领土
ax.add_feature(cfeature.OCEAN.with_scale('10m'))#添加海洋

extent = [115, 128, 22, 30]
ax.set_extent(extent)

gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='k', alpha=0.5, linestyle='--')
# 调节字体大小,属性用size,字体加粗用'weight':'bold',字体'font':'Arial'
gl.xlabel_style={'size':16}
gl.ylabel_style={'size':16}

start_point_tw = [119.125, 23.375]
start_point_dh = [124.125, 28.375]

# facecolor=(0.59375, 0.71484375, 0.8828125)这个是原始海水 water 的颜色
RE_tw = Rectangle(start_point_tw, 1, 1,linewidth=1,linestyle='-' ,zorder=2,edgecolor='red',facecolor=(0.59375, 0.71484375, 0.8828125), transform=ccrs.PlateCarree())
ax.add_patch(RE_tw)  #台湾海峡
RE_dh = Rectangle(start_point_dh, 1, 1,linewidth=1,linestyle='-' ,zorder=2,edgecolor='red',facecolor='none', transform=ccrs.PlateCarree())
ax.add_patch(RE_dh)  #东海

# scale_bar(ax=ax, location=(0.95, 0.6), length=5_00)
scale_bar(ax=ax, length=100, location=(0.9, 0.05))
plt.text(119.5, 24.8, 'Taiwan Strait', ha='left', rotation=48, fontsize=16, weight='bold')
plt.text(123, 28, 'East China Sea', fontsize=16, weight='bold')
t1 = ax.text(127.5, 22.2, u'\u25B2\nN', transform=ccrs.PlateCarree(), fontsize=12,
             horizontalalignment='center', verticalalignment='bottom', zorder=2)
plt.savefig('figure1.jpg', dpi=300)
plt.show()
