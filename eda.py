import utils.preprocess as pp
import utils.vis as vis


# import data
data_combined = pp.data_imported_combine(
    ['./data/waterLevel_Columbus.txt', 'water_level'],
    ['./data/discharge_Columbus.txt', 'discharge'],
)
data_combined = pp.data_add_target(data_combined, 2, 1, target_name='water_level')


# # show time series
# vis.plot_lines(data_combined,
#                ['water_level'],
#                './outputs/figs/line_water_level.html',
#                'Water Level',
#                )
# vis.plot_lines(data_combined,
#                ['discharge'],
#                './outputs/figs/line_discharge.html',
#                'Discharge',
#                )


# partial auto-correlation plot
# vis.plot_bar_pac(
#     data_combined['water_level'],
#     'Partial Auto-correlation Plot',
#     './outputs/figs/bar_water_level_pac.html',
# )
# vis.plot_bar_pac(
#     data_combined['water_level'].diff()[1:],
#     'Auto-correlation Plot of Diff',
#     './outputs/figs/bar_water_level_diff_pac.html'
# )


# cross partial auto-correlation
print('The cross partial auto-cor between water level and discharge is ',
      pp.partial_xcorr(data_combined.water_level.values, data_combined.discharge.values, max_lag = 50))


# # show surge
# vis.plot_lines_surge(data_combined, 'water_level', 'surge', './outputs/figs/line_water_surge.html')

print()