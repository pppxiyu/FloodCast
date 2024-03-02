def add_vertex_in_line(line, point, line_name, tolorence):
    coords = list(line.coords)
    for i in range(len(coords) - 1):
        segment = LineString(coords[i : i + 2])
        if point.distance(segment) <= tolorence:
            coords.insert(i + 1, (point.x, point.y))
            print(f"Add vertex {(point.x, point.y)} for {line_name}")
            break
    return LineString(coords)


def apply_vertex_in_line(row, gages, tolorence=1e-10):
    has_gages = gages[gages["HYRIV_ID"] == row["HYRIV_ID"]]
    line = row.geometry
    if len(has_gages) > 0:
        for gage in has_gages["nearest_point_on_river"]:
            line = add_vertex_in_line(line, gage, row["HYRIV_ID"], tolorence)
    return line


def split_river(rivers, gage_geo_merge_river):
    from shapely.ops import split
    from shapely import MultiPoint
    import matplotlib.pyplot as plt

    rivers_split = rivers[
        ~rivers["HYRIV_ID"].isin(gage_geo_merge_river["HYRIV_ID"].to_list())
    ]
    for index, row in rivers[
        rivers["HYRIV_ID"].isin(gage_geo_merge_river["HYRIV_ID"].to_list())
    ].iterrows():  # for river seg with gage
        split_points = gage_geo_merge_river[
            gage_geo_merge_river["HYRIV_ID"] == row["HYRIV_ID"]
        ][
            "nearest_point_on_river"
        ].values  # gages
        split_points = MultiPoint(split_points)  # gages
        river_segs = split(row.geometry, split_points)  # split river segs
        num_splits = 0
        for river_seg in river_segs.geoms:
            row.geometry = river_seg
            new_row = {
                col_k: [col_v]
                for col_k, col_v in zip(rivers_split.columns, row.to_list())
            }  # new row
            # rivers_split.loc[len(rivers_split), :] = row.to_list()
            rivers_split = pd.concat(
                (rivers_split, gpd.GeoDataFrame(new_row)), ignore_index=True
            )
            num_splits += 1

        if num_splits == 1:
            print(f"Split failed for {row['HYRIV_ID']}")
            fig, ax = plt.subplots()
            gage_geo_merge_river[gage_geo_merge_river["HYRIV_ID"] == row["HYRIV_ID"]][
                "nearest_point_on_river"
            ].plot(ax=ax)
            rivers[rivers["HYRIV_ID"] == row["HYRIV_ID"]].plot(ax=ax)
            plt.show()
        print()
    return rivers_split


def calculate_connections_among_gages(river_graph, gage_node_list):

    gage_graph_dict = {}
    for gage in gage_node_list:
        adj_nodes_layer = list(river_graph.successors(gage))
        path_dict = {
            adj_node: {"path": [gage, adj_node]} for adj_node in adj_nodes_layer
        }
        adj_nodes_layer_continue = [
            i for i in adj_nodes_layer if i not in gage_node_list
        ]

        while adj_nodes_layer_continue != []:
            adj_nodes_layer = []
            for adj_node_c in adj_nodes_layer_continue:
                adj_nodes = list(river_graph.successors(adj_node_c))

                for adj_node in adj_nodes:
                    exist_path = path_dict[adj_node_c]["path"]
                    exist_path.append(adj_node)
                    path_dict[adj_node] = {"path": exist_path}

                adj_nodes_layer.append(adj_nodes)

            if adj_nodes_layer != [[]]:
                del path_dict[adj_node_c]

            adj_nodes_layer = [i for list in adj_nodes_layer for i in list]
            if adj_nodes_layer == []:
                break
            adj_nodes_layer_continue = [
                i for i in adj_nodes_layer if i not in gage_node_list
            ]

        gage_graph_dict[gage] = path_dict

    # remove path beyond gage range
    gage_graph_dict_out = gage_graph_dict.copy()
    for k, v in gage_graph_dict.items():
        if list(v.keys()) == []:
            v = {None: {"path": None}}
            gage_graph_dict_out[k] = v
        if list(v.keys())[0] not in gage_node_list:
            print(f"Route from {k} to {list(v.keys())[0]} is taken off.")
            del gage_graph_dict_out[k]

    return gage_graph_dict_out


def calculate_path_length(gage_graph_dict, river_graph):
    gage_graph_dict_out = gage_graph_dict.copy()
    for k, v in gage_graph_dict.items():
        path = list(v.values())[0]["path"]
        path_length = sum(
            list(river_graph[path[i]][path[i + 1]].values())[0]["weight"]
            for i in range(len(path) - 1)
        )
        gage_graph_dict_out[k][list(v.keys())[0]]["weight"] = path_length
    return gage_graph_dict_out


def convert_point_to_name(names, graph_dict):
    assert isinstance(names, list)
    name_info_list = []
    for n in names:
        point2name_temp = n.set_index("nearest_point_on_river")["identifier"]
        point2name_temp = {
            idx: group.values.tolist()
            for idx, group in point2name_temp.groupby(point2name_temp.index)
        }
        point2name = {(k.x, k.y): v for k, v in point2name_temp.items()}
        name_info_list.append(point2name)

    # check consistency
    all_keys_1 = [k1 for k1, v1 in graph_dict.items()]
    all_keys_2 = [k2 for k1, v1 in graph_dict.items() for k2, v2 in v1.items()]
    all_keys = all_keys_1 + all_keys_2
    all_keys_if_name = [
        k in [k for ni in name_info_list for k, v in ni.items()] for k in all_keys
    ]
    if not (sum(all_keys_if_name) == len(all_keys_if_name)):
        raise ValueError("Not all points could be assigned with a name. Need check.")

    gage_graph_dict_out = {}
    for k1, v1 in graph_dict.items():
        for k2, v2 in v1.items():

            new_k1_list = []
            new_k2_list = []
            for name_info in name_info_list:
                try:
                    for ni in name_info[k1]:
                        new_k1_list.append(ni)
                except:
                    pass
                try:
                    for ni in name_info[k2]:
                        new_k2_list.append(ni)
                except:
                    pass

            for new_k1 in new_k1_list:
                gage_graph_dict_out[new_k1] = {new_k2: v2 for new_k2 in new_k2_list}
            # any pair of element in the same list indicates an edge of length 0
            for pairs in [
                (new_k1_list[i], new_k1_list[j])
                for i in range(len(new_k1_list))
                for j in range(len(new_k1_list))
                if i != j
            ]:
                if pairs[0] in gage_graph_dict_out.keys():
                    gage_graph_dict_out[pairs[0]][pairs[1]] = {'path': [], 'weight': 0}
                else:
                    gage_graph_dict_out[pairs[0]] = {pairs[1]: {'path': [], 'weight': 0}}
                print("Add an edges of length 0.")
            for pairs in [
                (new_k2_list[i], new_k2_list[j])
                for i in range(len(new_k2_list))
                for j in range(len(new_k2_list))
                if i != j
            ]:
                if pairs[0] in gage_graph_dict_out.keys():
                    gage_graph_dict_out[pairs[0]][pairs[1]] = {'path': [], 'weight': 0}
                else:
                    gage_graph_dict_out[pairs[0]] = {pairs[1]: {'path': [], 'weight': 0}}
                print("Add an edges of length 0.")

    return gage_graph_dict_out


def create_anchor_points(lat_label=None, lon_label=None):
    if lon_label is None:
        lon_label = ["-76.8", "-76.7", "-76.6", "-76.5", "-76.4", "-76.3"]
    if lat_label is None:
        lat_label = ["40.2", "40.3", "40.4", "40.5", "40.6"]
    lat_list = [float(i) + 0.05 for i in lat_label]
    lon_list = [float(i) + 0.05 for i in lon_label]
    geometry = [Point(lon, lat) for lon in lon_list for lat in lat_list]
    anchor_geo = gpd.GeoDataFrame(geometry=geometry).set_crs("epsg:4326")
    anchor_geo["lat"] = [g.y for g in geometry]
    anchor_geo["lon"] = [g.x for g in geometry]
    anchor_geo["identifier"] = (
        anchor_geo["lat"].round(2).astype(str)
        + "_"
        + anchor_geo["lon"].round(2).astype(str)
    )
    return anchor_geo


def create_grid(anchor_geo):
    anchor_geo["cell"] = anchor_geo.apply(
        lambda x: Polygon(
            [
                (x["lon"] - 0.05, x["lat"] - 0.05),
                (x["lon"] - 0.05, x["lat"] + 0.05),
                (x["lon"] + 0.05, x["lat"] + 0.05),
                (x["lon"] + 0.05, x["lat"] - 0.05),
            ]
        ),
        axis=1,
    )
    grid_geo = anchor_geo[["identifier", "cell"]]
    grid_geo = grid_geo.set_geometry("cell")
    grid_geo = grid_geo.set_crs("epsg:4326")
    grid_geo["area"] = grid_geo.geometry.area
    return grid_geo


def create_adjusted_centers(grid_geo, basin_geo, save_dir):
    grid_geo_f = gpd.overlay(grid_geo, basin_geo, how="intersection")
    grid_geo_f["centroid"] = grid_geo_f.geometry.centroid
    grid_geo_f["updated_area"] = grid_geo_f.geometry.area
    grid_geo_f["updated_area_ratio"] = grid_geo_f["updated_area"] / grid_geo_f["area"]
    grid_geo_f[["identifier", "updated_area_ratio"]].to_csv(
        f"{save_dir}/area_in_boundary_ratio.csv", index=False
    )

    gage_geo = grid_geo_f[["identifier", "centroid"]]
    gage_geo = gage_geo.rename(columns={"centroid": "geometry"})
    gage_geo = gage_geo.set_geometry("geometry")
    return gage_geo


def calculate_nearest_node_on_river(points, river):
    gage_merge_river = points.sjoin_nearest(
        river, how="left"
    )  # get nearest river for gages
    gage_merge_river = gage_merge_river[~gage_merge_river.duplicated("identifier")]
    gage_merge_river["nearest_point_on_river"] = gage_merge_river.apply(
        lambda x: nearest_points(
            x.geometry, rivers[rivers["HYRIV_ID"] == x["HYRIV_ID"]].geometry.values[0],
        )[1],
        axis=1,
    )  # get nearest point on river
    return gage_merge_river


def count_active_USGS_up_gage(
        row, search_distance, scope, work_dir_up, work_dir_self, num_to_select, active_gauges,
):
    gage = row['SITENO']
    gage_geo = gpd.read_file(f"{work_dir_self}/{gage}_geo.geojson")
    up_gage_geo = gpd.read_file(f'{work_dir_up}/{gage}_s{search_distance}_{scope}.geojson')

    if len(up_gage_geo) == 0:
        print('No active upstream surface water gage.')
        return pd.Series([0, []])
    up_gage_geo['SITENO'] = up_gage_geo['identifier'].str.split('-').str[1]
    up_gage_geo = up_gage_geo[up_gage_geo['identifier'] != gage_geo.iloc[0]['identifier']]
    up_gage_geo = up_gage_geo[up_gage_geo['SITENO'].isin(active_gauges)]
    if len(up_gage_geo) == 0:
        print("No active upstream surface water gage.")
        return pd.Series([0, []])

    up_gage_geo["distance"] = up_gage_geo.apply(
        lambda row: row.geometry.distance(gage_geo.iloc[0].geometry), axis=1
    )
    up_gage_geo_select = up_gage_geo.sort_values(by="distance").head(num_to_select)
    return pd.Series([len(up_gage_geo_select), up_gage_geo_select['SITENO'].to_list()])


analysis = "generate_river_adjacency_gauge"
dir_rivers = "./data/HydroRIVERS_v10_na_shp/HydroRIVERS_v10_na_shp/HydroRIVERS_v10_na_selected_mainstream.shp"
dir_all_USGS_gauges = "data/USGS_gage_geo/NWISMapperExport.shp"


if analysis == "filer_large_flooding_river":

    import pandas as pd
    import utils.preprocess as pp
    import os
    import geopandas as gpd

    working_dir_USGS_iv = "./data/USGS_gage_iv"
    working_dir_USGS_iv_20y = "./data/USGS_gage_iv_20y"
    working_dir_USGS_field = "./data/USGS_gage_field"
    working_dir_USGS_flood_stage = "./data/USGS_gage_flood_stage"
    working_dir_USGS_upstream = "./data/USGS_gage_upstream_geo"
    working_dir_USGS_gage_filtering = "./outputs/USGS_gaga_filtering"

    # read data
    rivers = gpd.read_file(dir_rivers).to_crs("EPSG:3395")
    rivers = rivers[
        rivers.ORD_FLOW <= 5
    ]  # keep rivers with a long term ave discharge over 10 m^3/s
    USGS_gauges = gpd.read_file(dir_all_USGS_gauges).to_crs("EPSG:3395")
    USGS_gauges = USGS_gauges[USGS_gauges['SITENO'].str.len() <= 10] # take off wells and miscellaneous sites

    # select the gauges located near the medium-to-large rivers
    USGS_gauges = USGS_gauges.sjoin_nearest(
        rivers[["geometry", "HYRIV_ID"]],
        how="inner",
        max_distance=500,
        rsuffix="river",
    )

    # select the gauges suffering from flooding
    pp.pull_USGS_gage_flood_stage(
        working_dir_USGS_flood_stage, USGS_gauges
    )  # pull flood stage
    USGS_gauges = USGS_gauges.merge(
        pd.read_csv(
            working_dir_USGS_flood_stage + "/flood_stages.csv",
            index_col=0,
            dtype={"site_no": str},
        ),
        how="left",
        left_on="SITENO",
        right_on="site_no",
    )  # merge
    USGS_gauges = USGS_gauges[
        ~USGS_gauges["action"].isna()
    ]  # remove gages w/o flood stage
    USGS_gauges = USGS_gauges[
        ~USGS_gauges["flood"].isna()
    ]  # remove gages w/o flood stage
    USGS_gauges = USGS_gauges[
        ~USGS_gauges["moderate"].isna()
    ]  # remove gages w/o flood stage
    USGS_gauges = USGS_gauges[
        ~USGS_gauges["major"].isna()
    ]  # remove gages w/o flood stage
    pp.pull_USGS_gage_iv(
        working_dir_USGS_iv, USGS_gauges, skip_gauge=["05427530"]
    )  # pull time series
    if os.path.exists(f"{working_dir_USGS_gage_filtering}/gauge_flood_counts.csv"):
        flood_counts_df = pd.read_csv(
            f"{working_dir_USGS_gage_filtering}/gauge_flood_counts.csv",
            dtype={"SITENO": str},
        )# USGS_gauges.to_csv(working_dir_USGS_major_flood_riv + '/gauge_flood_counts.csv', index=False)
        USGS_gauges = USGS_gauges.merge(
            flood_counts_df, on="SITENO", how="left"
        )  # merge
    else:
        print("No saved results. Do calculation.")
        USGS_gauges = pp.count_USGS_gage_flood(
            working_dir_USGS_iv,
            USGS_gauges,
            working_dir_USGS_gage_filtering,
            skip=["05427530", "08458000", "08458800", "08459000"],
        )
    USGS_gauges = USGS_gauges[
        ~USGS_gauges["SITENO"].isin(["05288670", "02462500", "03039035"])
    ]  # MANUAL CHECK
    USGS_gauges_selected = USGS_gauges[USGS_gauges["action_count"] > 0]  # filtering
    USGS_gauges_selected = USGS_gauges_selected[
        ~USGS_gauges_selected["SITENO"].duplicated()
    ]
    USGS_gauges_selected = USGS_gauges_selected[
        ~USGS_gauges_selected["SITENO"].duplicated()
    ]

    # select gages with a certain level of rc modeling bias
    if "dis_modeled_error" in USGS_gauges_selected.columns:
        USGS_gauges_selected = USGS_gauges_selected[
            USGS_gauges_selected[
                [
                    "dis_modeled_error",
                    "dis_modeled_error_action",
                    "dis_modeled_error_flood",
                    "dis_modeled_error_moderate",
                    "dis_modeled_error_major",
                ]
            ]
            .gt(5)
            .any(axis=1)
        ]

        USGS_gauges_selected = USGS_gauges_selected[
            USGS_gauges_selected["dis_modeled_error_action"] >= 5
        ]
        # USGS_gauges_selected = USGS_gauges_selected[
        #     USGS_gauges_selected["dis_modeled_error"] >= 3
        # ]

    # select the gauges with a certain amount of usable field visits
    pp.pull_USGS_gage_field(
        working_dir_USGS_field, USGS_gauges_selected
    )  # pull field measurements
    pp.pull_USGS_gage_iv(
        working_dir_USGS_iv_20y,
        USGS_gauges_selected,
        start="2004-01-01",
        end="2023-12-31",
    )
    if os.path.exists(f"{working_dir_USGS_gage_filtering}/gauge_field_measures.csv"):
        USGS_gauges_selected = pd.read_csv(
            f"{working_dir_USGS_gage_filtering}/gauge_field_measures.csv",
            dtype={"SITENO": str},
        )
    else:
        USGS_gauges_selected = pp.count_USGS_gage_field(
            working_dir_USGS_field,
            working_dir_USGS_iv_20y,
            working_dir_USGS_flood_stage,
            USGS_gauges_selected,
        )  # merge
        USGS_gauges_selected.to_csv(f'{working_dir_USGS_gage_filtering}/gauge_field_measures.csv', index=False)
    USGS_gauges_selected_2 = USGS_gauges_selected[
        USGS_gauges_selected["field_measure_count"] > 100
    ]  # filter


    archive_calculation = False
    if archive_calculation:
        # select gauges with a certain number of upstream gauges
        pp.pull_USGS_up_gage(
            working_dir_USGS_upstream,
            USGS_gauges_selected_2,
            search_distance=10,
            scope="UT",
        )
        pp.pull_USGS_up_gage(
            working_dir_USGS_upstream,
            USGS_gauges_selected_2,
            search_distance=50,
            scope="UT",
        )
        pp.pull_USGS_up_gage(
            working_dir_USGS_upstream,
            USGS_gauges_selected_2,
            search_distance=500,
            scope="UM",
        )
        USGS_gauges_selected_2 = pp.count_USGS_up_gage(
            working_dir_USGS_upstream,
            USGS_gauges_selected_2,
            search_distance=10,
            scope="UT",
        )
        USGS_gauges_selected_2 = pp.count_USGS_up_gage(
            working_dir_USGS_upstream,
            USGS_gauges_selected_2,
            search_distance=50,
            scope="UT",
        )
        USGS_gauges_selected_2 = pp.count_USGS_up_gage(
            working_dir_USGS_upstream,
            USGS_gauges_selected_2,
            search_distance=500,
            scope="UM",
        )
        USGS_gauges_selected_3 = USGS_gauges_selected_2[
            (USGS_gauges_selected_2["10_UT_upstream_gage_count"] > 1)
            & (USGS_gauges_selected_2["50_UT_upstream_gage_count"] > 3)
            & (USGS_gauges_selected_2["500_UM_upstream_gage_count"] > 5)
        ]

        # check the availability of features
        USGS_gauges_selected_3 = pp.check_USGS_gage_feature(
            working_dir_USGS_iv, USGS_gauges_selected_3
        )
        USGS_gauges_selected_3 = pp.check_USGS_up_gage_feature(
            working_dir_USGS_iv,
            working_dir_USGS_upstream,
            USGS_gauges_selected_3,
            search_distance=500,
            check_on="UM",
            check_for=["00020"],
        )
        USGS_gauges_selected_3 = pp.check_USGS_up_gage_feature(
            working_dir_USGS_iv,
            working_dir_USGS_upstream,
            USGS_gauges_selected_3,
            search_distance=50,
            check_on="UT",
            check_for=["00020"],
        )
        USGS_gauges_selected_3 = pp.check_USGS_up_gage_feature(
            working_dir_USGS_iv,
            working_dir_USGS_upstream,
            USGS_gauges_selected_3,
            search_distance=500,
            check_on="UM",
            check_for=["00045"],
        )
        USGS_gauges_selected_3 = pp.check_USGS_up_gage_feature(
            working_dir_USGS_iv,
            working_dir_USGS_upstream,
            USGS_gauges_selected_3,
            search_distance=50,
            check_on="UT",
            check_for=["00045"],
        )

if analysis == "filer_gauge_with_enough_data":
    import pandas as pd
    import os
    from ast import literal_eval
    import utils.preprocess as pp
    import json
    dir_cache = './data/cache'

    with open('./outputs/USGS_gaga_filtering/gauge_upstream_delete.json', 'r') as f:
        remove_dict = json.load(f)
    gauge_forecast = pd.read_csv(
        "./outputs/USGS_gaga_filtering/gauge_forecast.csv",
        dtype={"SITENO": str},
    )
    gauge_forecast['up_gage_names'] = gauge_forecast.apply(
        lambda row: sorted(list(set(
            literal_eval(row['active_up_gage_tri']) + literal_eval(row['active_up_gage_main']),
        )), reverse=True), axis=1
    )
    target_gage_list = gauge_forecast['SITENO'].to_list()
    upstream_gages_list = gauge_forecast['up_gage_names'].to_list()

    for target_gage, upstream_gages in zip(target_gage_list, upstream_gages_list):

        # data
        if os.path.isfile(f'{dir_cache}/data_{target_gage}.csv'):
            data = pd.read_csv(f'{dir_cache}/data_{target_gage}.csv', index_col=0, parse_dates=True)
            data.index = pd.to_datetime(data.index, utc=True)
            data.index = data.index.tz_convert('America/New_York')
        else:
            data = pp.import_data_combine(
                [f'./data/USGS_gage_iv_20y/{gage}.csv' for gage in upstream_gages + [target_gage]],
                tz='America/New_York',
                keep_col=['00065', '00060']
            )
            data.to_csv(f'{dir_cache}/data_{target_gage}.csv')
        data = data.resample('H', closed='right', label='right').mean()
        target_cols = [col for col in data.columns if col.endswith('00065')] + [f'{target_gage}_00060']
        data = data[target_cols]

        # up gages
        upstream_gages_exist = [i.split('_')[0] for i in data.columns]
        upstream_gages_exist = list(set(upstream_gages_exist))
        upstream_gages_exist.remove(target_gage)

        # no_data_up_gage = [g for g in upstream_gages if g not in upstream_gages_exist]
        # print(f'Up gages {no_data_up_gage} of {target_gage} do not have 00065 data.')

        if target_gage in list(remove_dict.keys()):
            upstream_gages_exist = [i for i in upstream_gages_exist if i not in remove_dict[target_gage]]

        # check overlapping missing
        sufficiency_dict = {}
        sufficiency_dict[target_gage] = len(data[f'{target_gage}_00065'].dropna()) / (24 * 365)

        for up_gg in upstream_gages_exist:
            sufficiency_dict[f'w_{up_gg}'] = len(data[[
                                                     f'{target_gage}_00065', f'{up_gg}_00065'
                                                 ]].dropna()) / (24 * 365)
        sufficiency_df = pd.DataFrame(list(sufficiency_dict.items()), columns=['gage', 'Year'])
        print(sufficiency_df)
        print()

        # report
        print(f'Gage {target_gage} has {len(upstream_gages_exist)} up gages. The year count of usable data is:')
        print(len(
            data[ [f'{target_gage}_00065', f'{target_gage}_00060'] + [f'{i}_00065' for i in upstream_gages_exist]].dropna()
        ) / (24 * 365))
        print()

if analysis == "generate_river_adjacency_gauge":

    import momepy
    import networkx as nx
    import geopandas as gpd
    import pandas as pd
    from shapely import Polygon
    from shapely.geometry import Point
    import matplotlib.pyplot as plt
    from shapely.ops import nearest_points
    from shapely.ops import split
    import matplotlib.pyplot as plt
    from shapely import MultiPoint
    from shapely import LineString
    import numpy as np
    from ast import literal_eval
    import os
    import json

    dir_output = './outputs/'
    dir_correction_file = './data/HydroRIVERS_v10_na_shp'
    rivers = gpd.read_file(dir_rivers)
    rivers_correction = pd.read_csv(
        f'{dir_correction_file}/manual_correction.txt', dtype={'gage': str, 'take_off_HYRIV_ID':int}
    )
    rivers = rivers[~rivers['HYRIV_ID'].isin(rivers_correction['take_off_HYRIV_ID'].to_list())]

    # gage lists
    gauge_forecast = pd.read_csv(
        "./outputs/USGS_gaga_filtering/gauge_forecast.csv",
        dtype={"SITENO": str},
    )
    gauge_forecast['up_gage_names'] = gauge_forecast.apply(
        lambda row: sorted(list(set(
            literal_eval(row['active_up_gage_tri']) + literal_eval(row['active_up_gage_main']),
        )), reverse=True), axis=1
    )
    gage_list = gauge_forecast['SITENO'].to_list()
    up_gage_list  = gauge_forecast['up_gage_names'].to_list()
    # gage_list = ["01573560"]
    # up_gage_list = [["01573160", "01573000", "01572190", "01572025",]]

    with open('./outputs/USGS_gaga_filtering/gauge_upstream_delete.json', 'r') as f:
        remove_dict = json.load(f)
    with open('./outputs/USGS_gaga_filtering/gauge_delete.json', 'r') as f:
        remove_list = json.load(f)

    for gage, up_gage in zip(gage_list, up_gage_list):

        up_gage = list(set(up_gage))

        if gage in remove_list:
            continue
        if gage in list(remove_dict.keys()):
            up_gage = [i for i in up_gage if i not in remove_dict[gage]]

        if not os.path.exists(f'./outputs/USGS_{gage}/adj_matrix_USGS_{gage}/adj_matrix.csv'):
        # if gage in list(remove_dict.keys()):
        # if gage == '05551580':

            dir_save = f"./outputs/USGS_{gage}/adj_matrix_USGS_{gage}"
            if not os.path.exists(dir_save):
                os.makedirs(dir_save)

            gage_geo_list = [gpd.read_file(f"./data/USGS_gage_geo/{gage}_geo.geojson")]
            for gg in up_gage:
                gage_geo_list.append(gpd.read_file(f"./data/USGS_gage_geo/{gg}_geo.geojson"))
            gage_geo = pd.concat(gage_geo_list, axis=0)

            # filter rivers
            buffer = 0.5
            minx, miny = gage_geo.bounds['minx'].min() - buffer, gage_geo.bounds['miny'].min() - buffer
            maxx, maxy = gage_geo.bounds['maxx'].max() + buffer, gage_geo.bounds['maxy'].max() + buffer
            bbox = Polygon(
                [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny), (minx, miny)]
            )
            rivers_select = rivers[rivers.intersects(bbox)]

            # # check rivers
            fig, ax = plt.subplots()
            rivers_select.plot(ax=ax)
            gage_geo.plot(ax=ax, markersize=4, color='red')
            plt.show()

            # split river at gage locations
            gage_merge_river = gage_geo.sjoin_nearest(
                rivers_select, how="left"
            )  # get nearest river for gages
            gage_merge_river["nearest_point_on_river"] = gage_merge_river.apply(
                lambda x: nearest_points(
                    x.geometry, rivers_select[rivers_select["HYRIV_ID"] == x["HYRIV_ID"]].geometry.values[0],
                )[1],
                axis=1,
            )  # get nearest point on river
            rivers_select.geometry = rivers_select.apply(
                lambda x: apply_vertex_in_line(x, gage_merge_river), axis=1
            )  # add nearest point as a vertex in river line
            rivers_select_split = split_river(rivers_select, gage_merge_river)

            # reset length
            rivers_select_split_proj_crs = rivers_select_split.to_crs("EPSG:3395")
            rivers_select_split["new_length"] = rivers_select_split_proj_crs.geometry.length

            # gdf to directed graph
            river_graph = momepy.gdf_to_nx(rivers_select_split, approach="primal", directed=True)

            # # check river graph
            # positions = {n: [n[0], n[1]] for n in list(river_graph.nodes)}
            # f, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
            # rivers_select.plot(color="k", ax=ax[0])
            # for i, facet in enumerate(ax):
            #     facet.set_title(("Rivers", "Graph")[i])
            #     facet.axis("off")
            # nx.draw(river_graph, positions, ax=ax[1], node_size=5)

            # set length as edge weights
            for u, v, data in river_graph.edges(data=True):
                data["weight"] = data["new_length"]

            # connections among gages
            gage_node_list = [
                (point.x, point.y)
                for point in gage_merge_river["nearest_point_on_river"].to_list()
            ]
            gage_graph_dict = calculate_connections_among_gages(river_graph, gage_node_list)

            # calculate weights
            gage_graph_dict_2 = calculate_path_length(gage_graph_dict, river_graph)

            # convert keys from points to names
            gage_graph_dict_3 = convert_point_to_name([gage_merge_river], gage_graph_dict_2)

            # sort
            gage_graph_dict_3 = dict(sorted(gage_graph_dict_3.items(), reverse=True))

            # # create gage network and get adj matrix
            # gage_graph = nx.from_dict_of_dicts(gage_graph_dict_3, create_using=nx.DiGraph)
            # adj_matrix = np.array(nx.adjacency_matrix(gage_graph).todense())

            # create gage network and get adj matrix
            gage_graph = nx.from_dict_of_dicts(gage_graph_dict_3, create_using=nx.DiGraph)
            gage_graph_sorted = nx.DiGraph()
            sorted_node_list = sorted(gage_graph.nodes(), reverse=True)
            try:
                sorted_node_list.remove(f'USGS-{gage}')
            except:
                print('Pause, target gage is lost in the calculation.')
            sorted_node_list = [f'USGS-{gage}'] + sorted_node_list  # make sure target gage is at front
            for node in sorted_node_list:
                gage_graph_sorted.add_node(node, **gage_graph.nodes[node])
            gage_graph_sorted.add_edges_from(gage_graph.edges(data=True))
            adj_matrix = np.array(nx.adjacency_matrix(gage_graph_sorted).todense())
            if adj_matrix.sum() == 0:
                raise ValueError("Adj matrix is all zero!")

            # dist to proximity
            adj_matrix_min = adj_matrix[adj_matrix != 0].min()
            adj_matrix_max = adj_matrix[adj_matrix != 0].max()
            if adj_matrix_max == adj_matrix_min:
                adj_matrix = np.where(
                    adj_matrix != 0, 1, 0,)
            else:
                adj_matrix = np.where(
                    adj_matrix != 0,
                    1 / (np.exp((adj_matrix - adj_matrix_min) / (adj_matrix_max - adj_matrix_min))),
                    0,
                )

            # vis adj matrix
            plt.figure()
            plt.imshow(adj_matrix, cmap="hot")
            plt.xticks(
                ticks=np.arange(adj_matrix.shape[0]),
                labels=[i.split("-")[1] for i in list(gage_graph_sorted.nodes)],
                rotation=45,
            )
            plt.yticks(
                ticks=np.arange(adj_matrix.shape[0]),
                labels=[i.split("-")[1] for i in list(gage_graph_sorted.nodes)],
            )
            for i in range(adj_matrix.shape[0]):
                for j in range(adj_matrix.shape[0]):
                    if adj_matrix[i, j] != 0:
                        text = plt.text(
                            j,
                            i,
                            round(adj_matrix[i, j], 2),
                            ha="center",
                            va="center",
                            color="#808080",
                            fontsize=10,
                            rotation=60,
                        )
            plt.savefig(dir_save + f"/adj_matrix.png", dpi=300)
            plt.show()

            # save
            adj_matrix_df = pd.DataFrame(
                adj_matrix,
                index=[i.split("-")[1] for i in list(gage_graph_sorted.nodes)],
                columns=[i.split("-")[1] for i in list(gage_graph_sorted.nodes)],
            )
            adj_matrix_df.to_csv(dir_save + f"/adj_matrix.csv")


if analysis == "generate_river_adjacency_precip":

    import momepy
    import networkx as nx
    import geopandas as gpd
    import pandas as pd
    from shapely import Polygon
    from shapely.geometry import Point
    import matplotlib.pyplot as plt
    from shapely.ops import nearest_points
    from shapely import LineString
    import numpy as np
    import utils.vis as vis

    dir_basin = "./data/USGS_gage_01573560"
    dir_save = "./outputs/USGS_01573560/adj_matrix_USGS_01573560"
    dir_map_save = "./papers/figs"

    # import basin
    basin_geo = gpd.read_file(f"{dir_basin}/01573560_basin_geo.geojson")
    basin_geo = basin_geo.iloc[1:]

    # import rivers
    rivers = gpd.read_file(dir_rivers)
    rivers = gpd.overlay(rivers, basin_geo, how="intersection")

    # anchor points and create grid
    anchor_geo = create_anchor_points()
    grid_geo = create_grid(anchor_geo)

    # filter grid using basin and create virtual gages
    gage_geo = create_adjusted_centers(grid_geo, basin_geo, dir_save)

    # get nearest point on river
    new_center_merge_river = calculate_nearest_node_on_river(gage_geo, rivers)

    # # check
    # fig, ax = plt.subplots()
    # rivers.plot(ax=ax)
    # anchor_geo.plot(ax=ax, markersize=15)
    # grid_geo.plot(ax=ax, alpha=0.5, facecolor="none", edgecolor="black")
    # gage_geo.plot(ax=ax, markersize=15)
    # basin_geo.plot(ax=ax, facecolor="none", edgecolor="black")
    # new_center_merge_river["nearest_point_on_river"].plot(ax=ax, markersize=20)
    # plt.show()
    #
    # vis.plot_map_nodes(
    #     rivers,
    #     anchor_geo,
    #     grid_geo,
    #     gage_geo,
    #     basin_geo,
    #     new_center_merge_river["nearest_point_on_river"],
    #     dir_map_save,
    # )

    # split river at gage locations
    rivers.geometry = rivers.apply(
        lambda x: apply_vertex_in_line(x, new_center_merge_river), axis=1
    )  # add nearest point as a vertex in river line
    rivers_split = split_river(rivers, new_center_merge_river)

    # reset length
    rivers_split_proj_crs = rivers_split.to_crs("EPSG:3395")
    rivers_split["new_length"] = rivers_split_proj_crs.geometry.length

    # gdf to directed graph
    river_graph = momepy.gdf_to_nx(rivers_split, approach="primal", directed=True)

    # # check river graph
    # positions = {n: [n[0], n[1]] for n in list(river_graph.nodes)}
    # f, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    # rivers.plot(color="k", ax=ax[0])
    # for i, facet in enumerate(ax):
    #     facet.set_title(("Rivers", "Graph")[i])
    #     facet.axis("off")
    # nx.draw(river_graph, positions, ax=ax[1], node_size=5)
    # plt.show()

    # set length as edge weights
    for u, v, data in river_graph.edges(data=True):
        data["weight"] = data["new_length"]

    # connections among gages
    gage_node_list = [
        (point.x, point.y)
        for point in new_center_merge_river["nearest_point_on_river"].to_list()
    ]
    gage_graph_dict = calculate_connections_among_gages(river_graph, gage_node_list)

    # # check connections
    # end_points = [list(v.keys())[0] for k, v in gage_graph_dict.items()]
    # end_points_geo = gpd.GeoSeries([Point(xy) for xy in end_points]).set_crs(
    #     "epsg:4326"
    # )
    # start_points = [k for k, v in gage_graph_dict.items()]
    # start_points_geo = gpd.GeoSeries([Point(xy) for xy in start_points]).set_crs(
    #     "epsg:4326"
    # )
    # for i in range(len(start_points_geo)):
    #     fig, ax = plt.subplots()
    #     rivers.plot(ax=ax)
    #     basin_geo.plot(ax=ax, facecolor="none", edgecolor="black")
    #     start_points_geo.iloc[i : i + 1].plot(ax=ax, markersize=20)
    #     end_points_geo.iloc[i : i + 1].plot(ax=ax, markersize=20)
    #     plt.show()

    # calculate weights
    gage_graph_dict_2 = calculate_path_length(gage_graph_dict, river_graph)

    # convert keys from points to names
    gage_graph_dict_3 = convert_point_to_name([new_center_merge_river], gage_graph_dict_2)

    # create gage network and get adj matrix
    gage_graph = nx.from_dict_of_dicts(gage_graph_dict_3, create_using=nx.DiGraph)
    gage_graph_sorted = nx.DiGraph()
    for node in sorted(gage_graph.nodes(), reverse=True):
        gage_graph_sorted.add_node(node, **gage_graph.nodes[node])
    gage_graph_sorted.add_edges_from(gage_graph.edges(data=True))
    adj_matrix = np.array(nx.adjacency_matrix(gage_graph_sorted).todense())

    # dist to proximity
    adj_matrix_min = adj_matrix[adj_matrix != 0].min()
    adj_matrix_max = adj_matrix[adj_matrix != 0].max()
    adj_matrix = np.where(
        adj_matrix != 0,
        1
        / (
            np.exp((adj_matrix - adj_matrix_min) / (adj_matrix_max - adj_matrix_min))
            + 0
        ),
        0,
    )

    # vis adj matrix
    plt.figure()
    plt.imshow(adj_matrix, cmap="hot")
    plt.xticks(
        ticks=np.arange(adj_matrix.shape[0]),
        labels=list(gage_graph_sorted.nodes),
        rotation=45,
    )
    plt.yticks(
        ticks=np.arange(adj_matrix.shape[0]), labels=list(gage_graph_sorted.nodes),
    )
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[0]):
            if adj_matrix[i, j] != 0:
                text = plt.text(
                    j,
                    i,
                    round(adj_matrix[i, j], 2),
                    ha="center",
                    va="center",
                    color="#808080",
                    fontsize=10,
                    rotation=60,
                )
    plt.savefig(dir_save + f"/adj_matrix_precipitation.png", dpi=300)
    plt.show()

    # save
    adj_matrix_df = pd.DataFrame(
        adj_matrix,
        index=list(gage_graph_sorted.nodes),
        columns=list(gage_graph_sorted.nodes),
    )
    adj_matrix_df.to_csv(dir_save + f"/adj_matrix_precipitation.csv")

if analysis == "generate_river_adjacency_bipartite":

    import momepy
    import networkx as nx
    import geopandas as gpd
    import pandas as pd
    from shapely import Polygon
    from shapely.geometry import Point
    import matplotlib.pyplot as plt
    from shapely.ops import nearest_points
    from shapely import LineString
    import numpy as np
    import utils.vis as vis

    dir_basin = "./data/USGS_gage_01573560"
    dir_save = "./outputs/USGS_01573560/adj_matrix_USGS_01573560"
    dir_map_save = "./papers/figs/map_all_nodes.html"

    gage = "01573560"
    up_gage = [
        "01573160",
        "01573000",
        "01572190",
        "01572025",
    ]

    # import basin
    basin_geo = gpd.read_file(f"{dir_basin}/01573560_basin_geo.geojson")
    basin_geo = basin_geo.iloc[1:]

    # import rivers
    rivers = gpd.read_file(dir_rivers)
    rivers = gpd.overlay(rivers, basin_geo, how="intersection")

    # anchor points and create grid
    anchor_geo = create_anchor_points()
    grid_geo = create_grid(anchor_geo)

    # filter grid using basin and create virtual gages
    new_center_geo = create_adjusted_centers(grid_geo, basin_geo, dir_save)

    # get nearest point on river
    new_center_merge_river = calculate_nearest_node_on_river(new_center_geo, rivers)

    # import gages
    gage_geo_list = [gpd.read_file(f"./data/USGS_gage_geo/{gage}_geo.geojson")]
    for gg in up_gage:
        gage_geo_list.append(gpd.read_file(f"./data/USGS_gage_geo/{gg}_geo.geojson"))
    gage_geo = pd.concat(gage_geo_list, axis=0)
    gage_merge_river = calculate_nearest_node_on_river(gage_geo, rivers)

    # check
    # vis.plot_map_nodes(
    #     rivers,
    #     anchor_geo,
    #     grid_geo,
    #     new_center_geo,
    #     basin_geo,
    #     new_center_merge_river["nearest_point_on_river"],
    #     dir_map_save,
    #     gage_merge_river["nearest_point_on_river"],
    # )

    # split river at gage locations
    rivers.geometry = rivers.apply(
        lambda x: apply_vertex_in_line(x, new_center_merge_river), axis=1
    )  # add nearest point as a vertex in river line
    rivers.geometry = rivers.apply(
        lambda x: apply_vertex_in_line(x, gage_merge_river), axis=1
    )  # add nearest point as a vertex in river line
    rivers_split = split_river(rivers, new_center_merge_river)
    rivers_split = split_river(rivers_split, gage_merge_river)

    # reset length
    rivers_split_proj_crs = rivers_split.to_crs("EPSG:3395")
    rivers_split["new_length"] = rivers_split_proj_crs.geometry.length

    # gdf to directed graph
    river_graph = momepy.gdf_to_nx(rivers_split, approach="primal", directed=True)

    # # check river graph
    # positions = {n: [n[0], n[1]] for n in list(river_graph.nodes)}
    # f, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    # rivers.plot(color="k", ax=ax[0])
    # for i, facet in enumerate(ax):
    #     facet.set_title(("Rivers", "Graph")[i])
    #     facet.axis("off")
    # nx.draw(river_graph, positions, ax=ax[1], node_size=5)
    # plt.show()

    # set length as edge weights
    for u, v, data in river_graph.edges(data=True):
        data["weight"] = data["new_length"]

    # connections among gages. NOTE: nodes of same location will overlap each other!
    gage_node_list = [
        (point.x, point.y)
        for point in (
            new_center_merge_river["nearest_point_on_river"].to_list()
            + gage_merge_river["nearest_point_on_river"].to_list()
        )
    ]
    gage_graph_dict = calculate_connections_among_gages(river_graph, gage_node_list)

    # # check connections
    # end_points = [list(v.keys())[0] for k, v in gage_graph_dict.items()]
    # end_points_geo = gpd.GeoSeries([Point(xy) for xy in end_points]).set_crs(
    #     "epsg:4326"
    # )
    # start_points = [k for k, v in gage_graph_dict.items()]
    # start_points_geo = gpd.GeoSeries([Point(xy) for xy in start_points]).set_crs(
    #     "epsg:4326"
    # )
    # for i in range(len(start_points_geo)):
    #     fig, ax = plt.subplots()
    #     rivers.plot(ax=ax)
    #     basin_geo.plot(ax=ax, facecolor="none", edgecolor="black")
    #     start_points_geo.iloc[i : i + 1].plot(ax=ax, markersize=20)
    #     end_points_geo.iloc[i : i + 1].plot(ax=ax, markersize=20)
    #     plt.show()

    # calculate weights
    gage_graph_dict_2 = calculate_path_length(gage_graph_dict, river_graph)

    # convert keys from points to names
    gage_graph_dict_3 = convert_point_to_name(
        [new_center_merge_river, gage_merge_river], gage_graph_dict_2
    )

    # create gage network
    graph = nx.from_dict_of_dicts(gage_graph_dict_3, create_using=nx.DiGraph)

    # create bipartite graph from precip to gauges
    edges_2_remove = []
    for u, v in graph.edges():
        if u.startswith("USGS") or not v.startswith("USGS"):
            edges_2_remove.append((u, v))
    graph_bipartite = graph.copy()
    graph_bipartite.remove_edges_from(edges_2_remove)

    # check if any gauge has no predecessor precip points
    for node in graph_bipartite.nodes():
        if node.startswith("USGS"):
            if len(list(graph_bipartite.predecessors(node))) < 1:
                raise ValueError('Bipartite graph cannot fully reflect the adjacency. Check it!')

    # convert dist to proximity
    edge_dists = [data['weight'] for node1, node2, data in graph_bipartite.edges(data=True)]
    for node1, node2, data in graph_bipartite.edges(data=True):
        data['weight'] = 1 / (np.exp((data['weight'] - min(edge_dists)) / (max(edge_dists) - min(edge_dists))))

    # bipartite adj matrix
    gauge_set = {n for n in graph_bipartite.nodes() if n.startswith('USGS')}
    gauge_set = sorted(gauge_set, reverse=True)
    precip_set = {n for n in graph_bipartite.nodes() if not n.startswith('USGS')}
    precip_set = sorted(precip_set, reverse=True)
    adj_matrix_bipartite = nx.algorithms.bipartite.matrix.biadjacency_matrix(
        graph_bipartite, row_order=precip_set, column_order=gauge_set,
    )
    adj_matrix_bipartite = adj_matrix_bipartite.toarray()

    # vis adj matrix
    gauge_set_short = {i.split('-')[1] for i in gauge_set}
    gauge_set_short = sorted(gauge_set_short, reverse=True)

    plt.figure(figsize=(4, 10))
    plt.imshow(adj_matrix_bipartite, cmap="hot")
    plt.xticks(
        ticks=np.arange(adj_matrix_bipartite.shape[1]),
        labels=list(gauge_set_short),
        rotation=45,
    )
    plt.yticks(
        ticks=np.arange(adj_matrix_bipartite.shape[0]), labels=list(precip_set),
    )
    for i in range(adj_matrix_bipartite.shape[0]):
        for j in range(adj_matrix_bipartite.shape[1]):
            if adj_matrix_bipartite[i, j] != 0:
                text = plt.text(
                    j,
                    i,
                    round(adj_matrix_bipartite[i, j], 2),
                    ha="center",
                    va="center",
                    color="#808080",
                    fontsize=8,
                    rotation=60,
                )
    plt.savefig(dir_save + f"/adj_matrix_both.png", dpi=300)
    plt.show()

    # save
    adj_matrix_df = pd.DataFrame(
        adj_matrix_bipartite,
        index=list(precip_set),
        columns=list(gauge_set_short),
    )
    adj_matrix_df.to_csv(dir_save + f"/adj_matrix_both.csv")

if analysis == 'select_close_up_gages':

    import pandas as pd
    import geopandas as gpd
    import utils.preprocess as pp
    import numpy as np
    working_dir_USGS_gage_filtering = "./outputs/USGS_gaga_filtering"
    working_dir_USGS_upstream_geo = "./data/USGS_gage_upstream_geo"
    working_dir_USGS_gage_geo = "./data/USGS_gage_geo"
    working_dir_USGS_gage_iv_20y = "./data/USGS_gage_iv_20y"
    working_dir_USGS_field = "./data/USGS_gage_field"
    working_dir_USGS_rc = "./data/USGS_gage_rc"
    working_dir_USGS_basin = './data/USGS_basin_geo'

    USGS_gauges = gpd.read_file(dir_all_USGS_gauges).to_crs("EPSG:3395")
    USGS_gauges = USGS_gauges[USGS_gauges['SITENO'].str.len() <= 10]
    USGS_gauges = USGS_gauges['SITENO'].to_list()

    USGS_gauges_select = pd.read_csv(
        f"{working_dir_USGS_gage_filtering}/gauge_field_measures.csv",
        dtype={"SITENO": str},
    )

    # select gauges with a certain number of upstream gauges
    pp.pull_USGS_up_gage(
        working_dir_USGS_upstream_geo, USGS_gauges_select, search_distance=100, scope="UT",
    )
    pp.pull_USGS_up_gage(
        working_dir_USGS_upstream_geo, USGS_gauges_select, search_distance=500, scope="UM",
    )
    USGS_gauges_select = pp.count_USGS_up_gage(
        working_dir_USGS_upstream_geo, USGS_gauges_select, search_distance=100, scope="UT",
    )
    USGS_gauges_select = pp.count_USGS_up_gage(
        working_dir_USGS_upstream_geo, USGS_gauges_select, search_distance=500, scope="UM",
    )

    pp.pull_USGS_gage_geo(working_dir_USGS_gage_geo, USGS_gauges_select['SITENO'].to_list())
    USGS_gauges_select[['active_up_gage_num_tri', 'active_up_gage_tri']] = USGS_gauges_select.apply(
        lambda x: count_active_USGS_up_gage(
            x, 100, 'UT',
            working_dir_USGS_upstream_geo, working_dir_USGS_gage_geo,
            4, USGS_gauges
        ), axis=1
    )
    USGS_gauges_select[['active_up_gage_num_main', 'active_up_gage_main']] = USGS_gauges_select.apply(
        lambda x: count_active_USGS_up_gage(
            x, 500, 'UM',
            working_dir_USGS_upstream_geo, working_dir_USGS_gage_geo,
            3, USGS_gauges
        ), axis=1
    )

    USGS_gauges_adequate = USGS_gauges_select[
        USGS_gauges_select['active_up_gage_num_main'] >= 1
        ]

    gage_pull_list = USGS_gauges_select['active_up_gage_main'].sum() + USGS_gauges_select['active_up_gage_tri'].sum()
    gage_pull_list = list(set(gage_pull_list))
    pp.pull_USGS_gage_iv(
        working_dir_USGS_gage_iv_20y,
        pd.DataFrame(gage_pull_list, columns=['SITENO']),
        start='2003-12-24', end='2023-12-24'
    )
    # pp.pull_USGS_gage_field(working_dir_USGS_field, USGS_gauges_adequate)
    # pp.pull_USGS_gage_rc(working_dir_USGS_rc, USGS_gauges_adequate)
    # pp.pull_USGS_gage_geo(working_dir_USGS_gage_geo, gage_pull_list)

    # USGS_gauges_adequate.to_csv(f'{working_dir_USGS_gage_filtering}/gauge_forecast.csv', index=False)

    print()

if analysis == "select_meteo_stations":
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point
    import matplotlib.pyplot as plt

    dir_stations = "data/NCEI_meteo_data_geo"
    dir_watershed = "data/USGS_gage_05311000/05311000_upstream_boundary_geo.json"

    candidate_stations = pd.read_csv(dir_stations + "/candidate_stations.csv")
    geometry = [
        Point(x, y)
        for x, y in zip(candidate_stations["LONGITUDE"], candidate_stations["LATITUDE"])
    ]
    candidate_stations = gpd.GeoDataFrame(
        candidate_stations, geometry=geometry
    ).set_crs(epsg=4326)

    watershed = gpd.read_file(dir_watershed)

    fig, ax = plt.subplots()
    candidate_stations.plot(ax=ax)
    watershed.plot(ax=ax)
    plt.show()

    select_stations = gpd.sjoin(candidate_stations, watershed, how="inner", op="within")
    select_stations.to_file(
        dir_stations + "/selected_stations.geojson", driver="GeoJSON"
    )

if analysis == "sufficiency":

    import geopandas as gpd
    import plotly.figure_factory as ff

    dir_counties = "./data/cb_2018_us_county_500k/cb_2018_us_county_500k.shp"

    def search_gauge(river, df, base_weight, log=False, ratio_clip=True):
        # USE: given a river segment, search and accumulate the contribution of all upstream gauges
        # INPUT: int, target river segment index
        #        df, dataframe contains upstreams and gauge
        #        base_weight, the base weight of this river segment
        # OUTPUT: float, the coverage ratio, 0-1

        upstream = df[df["HYRIV_ID_down"] == river].copy()
        upstream["weight"] = (
            upstream["DIS_AV_CMS_up"] / upstream["DIS_AV_CMS_down"] * base_weight
        )

        if ratio_clip:
            upstream.loc[upstream["weight"] > 1, "weight"] = 1

        gauged_upstream = upstream[upstream["index_gauge"].notna()]
        discharge_covered_ratio = gauged_upstream["weight"].sum()

        if log:
            print(
                f"River:{river}. Base_weight:{base_weight}. "
                f"Gauged upstream{upstream[upstream['index_gauge'].notna()]['HYRIV_ID_up'].to_list()}. "
                f"Gauged upstream weight added:{discharge_covered_ratio}"
            )

        not_gauged_upstream = upstream[upstream["index_gauge"].isna()][
            "HYRIV_ID_up"
        ].to_list()
        not_gauged_upstream_weight = upstream[upstream["index_gauge"].isna()][
            "weight"
        ].to_list()

        for stream, weight in zip(not_gauged_upstream, not_gauged_upstream_weight):
            discharge_covered_ratio += search_gauge(stream, df, weight)

        return discharge_covered_ratio

    def apply_search_gauge(row, df, base_weight):
        print(f"Searching for {row['HYRIV_ID']}")
        return search_gauge(row["HYRIV_ID"], df, base_weight)

    counties = gpd.read_file(dir_counties).to_crs("EPSG:3395")
    rivers = gpd.read_file(dir_rivers).to_crs("EPSG:3395")
    gauges = gpd.read_file(dir_all_USGS_gauges).to_crs("EPSG:3395")
    print("Data imported")

    # topological relationship and tell if gauge is at upstream
    rivers_w_upstream = rivers.merge(
        rivers.drop(columns=["geometry"]),
        left_on="NEXT_DOWN",
        right_on="HYRIV_ID",
        how="right",
        suffixes=["_up", "_down"],
    )
    rivers_w_upstream_gauge = rivers_w_upstream.sjoin_nearest(
        gauges[["geometry"]], how="left", max_distance=500, rsuffix="gauge"
    )

    # get target rivers with level 1-5
    rivers_target = rivers.sjoin_nearest(
        gauges[["geometry"]], how="left", max_distance=500, rsuffix="gauge"
    )
    rivers_target = rivers_target[
        (rivers_target["ORD_FLOW"] <= 5) & (rivers_target["index_gauge"].notna())
    ]
    rivers_target = rivers_target.drop_duplicates(subset="HYRIV_ID", keep="first")

    # calculate flow coverage rate
    rivers_target["DIS_COV_RATIO"] = rivers_target.apply(
        apply_search_gauge, axis=1, df=rivers_w_upstream_gauge, base_weight=1
    )

    # initialize the calculation of 1st-order upstream discharge
    # dis_up = rivers_target[['HYRIV_ID_down', 'DIS_AV_CMS_up']].groupby('HYRIV_ID_down').sum().reset_index()
    # rivers_target = rivers_target.merge(dis_up, on='HYRIV_ID_down', how='left', suffixes=['_itself', '_all'])
    # rivers_target.loc[rivers_target['HYRIV_ID_up'].isna(), ['DIS_AV_CMS_up_all']] = rivers_target.loc[
    #     rivers_target['HYRIV_ID_up'].isna(), 'DIS_AV_CMS_down']
    # rivers_target['UP_DIS_RATIO'] = rivers_target['DIS_AV_CMS_down'] / rivers_target['DIS_AV_CMS_up_all']

    # vis
    fig = ff.create_distplot(
        [rivers_target["DIS_COV_RATIO"].to_list()],
        ["Discharge Coverage Ratio"],
        bin_size=0.1,
    )
    fig.write_html("./outputs/figs/dis_cov_ratio.html")

if analysis == "map_gage_stream":

    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import utils.preprocess as pp
    import numpy as np
    import shapely
    import plotly.express as px
    import plotly.graph_objects as go

    working_dir = "./data/"
    gage = "01573560"
    up_gage = [
        "01573160",
        "01573000",
        "01572190",
        "01572025",
    ]

    pp.pull_USGS_stream_geo(f"./data/USGS_gage_{gage}", [gage], on="UT", dist=125)
    pp.pull_USGS_stream_geo(f"./data/USGS_gage_{gage}", [gage], on="UM", dist=100)
    pp.pull_USGS_gage_geo(f"./data/USGS_gage_geo", [gage])
    pp.pull_USGS_gage_geo(f"./data/USGS_gage_geo", up_gage)

    upstream_UM_geo = gpd.read_file(
        f"./data/USGS_gage_{gage}/{gage}_s100_UM_stream.geojson"
    )
    upstream_UT_geo = gpd.read_file(
        f"./data/USGS_gage_{gage}/{gage}_s125_UT_stream.geojson"
    )

    gage_geo = gpd.read_file(f"./data/USGS_gage_{gage}/{gage}.geojson")
    up_gage_geo_list = []
    for gg in up_gage:
        up_gage_geo_list.append(gpd.read_file(f"./data/USGS_gage_{gage}/{gg}.geojson"))
    up_gage_geo = pd.concat(up_gage_geo_list, axis=0)

    lats = []
    lons = []
    for feature in upstream_UT_geo.geometry:
        if isinstance(feature, shapely.geometry.linestring.LineString):
            linestrings = [feature]
        elif isinstance(feature, shapely.geometry.multilinestring.MultiLineString):
            linestrings = feature.geoms
        else:
            continue
        for linestring in linestrings:
            x, y = linestring.xy
            lats = np.append(lats, y)
            lons = np.append(lons, x)
            lats = np.append(lats, None)
            lons = np.append(lons, None)
    fig = px.line_mapbox(
        lat=lats,
        lon=lons,
        mapbox_style="outdoors",
        # center={"lat": 45.26504, "lon": -96.29581},
        zoom=9.2,
    )
    fig.update_traces(line=dict(color="#7FAFE3", width=1))

    lats = []
    lons = []
    for feature in upstream_UM_geo.geometry:
        if isinstance(feature, shapely.geometry.linestring.LineString):
            linestrings = [feature]
        elif isinstance(feature, shapely.geometry.multilinestring.MultiLineString):
            linestrings = feature.geoms
        else:
            continue
        for linestring in linestrings:
            x, y = linestring.xy
            lats = np.append(lats, y)
            lons = np.append(lons, x)
            lats = np.append(lats, None)
            lons = np.append(lons, None)
    fig.add_trace(
        go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode="lines",  # 6B80BF
            marker=go.scattermapbox.Marker(size=10, color="#6B80BF",),
            text=["Target Gauge"],
            name="Streams",
        )
    )

    fig.add_trace(
        go.Scattermapbox(
            lat=gage_geo.geometry.y,
            lon=gage_geo.geometry.x,
            mode="markers",
            marker=go.scattermapbox.Marker(size=15, color="#C23C2C",),
            text=gage_geo["identifier"].str.replace("USGS-", "").to_list(),
            name="Target gauge",
        )
    )
    fig.add_trace(
        go.Scattermapbox(
            lat=up_gage_geo.geometry.y,
            lon=up_gage_geo.geometry.x,
            mode="markers",
            marker=go.scattermapbox.Marker(size=10, color="#ED4937",),
            text=up_gage_geo["identifier"].str.replace("USGS-", "").to_list(),
            name="Upstream gauges",
        )
    )
    with open("./utils/mapbox_token.txt") as f:
        mapbox_token = f.readline().strip()

    print(
        f"Annotation x coord: {up_gage_geo.geometry.x.to_list() + gage_geo.geometry.x.to_list()}"
    )
    print(
        f"Annotation y coord: {up_gage_geo.geometry.y.to_list() + gage_geo.geometry.y.to_list()}"
    )
    print(
        f"Annotation text: {up_gage_geo['identifier'].str.replace('USGS-', '').to_list() + gage_geo['identifier'].str.replace('USGS-', '').to_list()}"
    )
    for lon, lat, text in zip(
        [-76.56188889, -76.57718829, -76.530797, -76.4021793, -76.6677469,],
        [40.3426111, 40.4025897, 40.4792558, 40.53258999, 40.29842358,],
        up_gage_geo["identifier"].str.replace("USGS-", "").to_list()
        + gage_geo["identifier"].str.replace("USGS-", "").to_list(),
    ):
        fig.add_annotation(
            go.layout.Annotation(
                text=text,
                x=lon,
                y=lat,
                xref="x",
                yref="y",
                showarrow=False,
                font=dict(size=12),
                bgcolor="white",
            )
        )
    fig.update_layout(
        mapbox_accesstoken=mapbox_token,
        # xaxis_visible=False, yaxis_visible=False
    )
    fig.write_html("./outputs/figs/map_gage_stream.html")

print()
