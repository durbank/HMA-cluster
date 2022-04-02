# This script prepares data for importing into R for use in INLA modeling.

# %% Set environment

# Import modules
import re
import time
from pathlib import Path
import pandas as pd
import pyproj
import geopandas as gpd
import xarray as xr
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.cluster import KMeans
from sklearn.linear_model import TheilSenRegressor
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import geoviews as gv
gv.extension('bokeh')
import holoviews as hv
hv.extension('bokeh')

# Environment setup
ROOT_DIR = Path('').absolute().parent
DATA_DIR = ROOT_DIR.joinpath('data')

# %% Import and format HAR climate data

# Get list of all downloaded climate files
har_fn = [
    p.name for p in list(
        DATA_DIR.joinpath('har-data').glob('*.nc'))]

# Assign 2D variables of interest (and names to use)
vars_2d = ['t2', 'prcp']
var_names = ['temp', 'prcp']

# Format 2D data into mean daily value xr arrays
das = []
for var in vars_2d:

    # Subset files to those matching regex of current variable
    var_regex = r".+_" + re.escape(var) + r"_(\d)+.+"
    files = sorted(
        [fn for fn in har_fn if re.search(var_regex, fn)])

    # Intialize first year as new array with days-of-year dim 
    tmp = xr.open_dataarray(
        DATA_DIR.joinpath('har-data', files[0]))
    da = xr.DataArray(
        data=tmp[0:365,:,:].data, 
        coords={
            'day': np.arange(0,365), 
            'south_north': tmp.south_north, 
            'west_east': tmp.west_east, 
            'lat':tmp.lat, 'lon':tmp.lon}, 
        dims=['day', 'south_north', 'west_east'], 
        attrs=tmp.attrs)

    # Concatenate additional years of data into 
    # array along 'year' dim
    for file in files[1:]:
        tmp = xr.open_dataarray(
            DATA_DIR.joinpath('har-data', file))
        tmp = xr.DataArray(
            data=tmp[0:365,:,:].data, 
            coords={
                'day': np.arange(0,365), 
                'south_north': tmp.south_north, 
                'west_east': tmp.west_east, 
                'lat':tmp.lat, 'lon':tmp.lon}, 
            dims=['day', 'south_north', 'west_east'], 
            attrs=tmp.attrs)
        da = xr.concat([da, tmp], 'year')

    # Compute mean daily value across all years
    da_clim = da.mean(dim='year')

    # Assign attributes
    da_clim.attrs = da.attrs
    da_clim.day.attrs = {
        'long_name': 'day of (365-day) year', 
        'units': '24-hour day'}

    das.append(da_clim)

# Combine 2D xr arrays into single xr dataset
ds = xr.Dataset(dict(zip(var_names, das)))

# Convert prcp from rainfall intensity to total 
# daily precipitation
ds['prcp'] = 24*ds['prcp']
ds['prcp'].attrs = {
    'long_name': 'total daily precipitation', 
    'units': 'mm'}


# Define season indices
DJF = np.concatenate((np.arange(335,365), np.arange(0,60)))
# MAM = np.arange(60,152)
JJA = np.arange(152,244)
# SON = np.arange(244,335)
WARM = np.arange(91,273)

# Create xr-arrays for single-valued variables 
# (temp amplitude, seasonal means/totals, etc.)
das_season = []
seasons = [DJF, JJA, WARM]
name_season = ['winter', 'summer', 'warm']
for i, season in enumerate(seasons):
    # Calculate mean seasonal air temperature
    da_T = ds['temp'].sel(day=season).mean(dim='day')
    da_T.attrs =  {
        'long_name': 'Mean '+name_season[i]+' 2-m air temperature',
        'units': ds['temp'].units}
    das_season.append(da_T)

    # Calculate total seasonal precipitation
    da_P = ds['prcp'].sel(day=season).sum(dim='day')
    da_P.attrs = {
        'long_name': 'Total '+name_season[i]+' precipitation', 
        'units': ds['prcp'].units}
    das_season.append(da_P)

# Combine seasonal arrays to dataset
var_seasons = [
    'temp_DJF', 'prcp_DJF', 'temp_JJA', 'prcp_JJA', 
    'temp_WARM', 'prcp_WARM']
ds_season = xr.Dataset(dict(zip(var_seasons, das_season)))

# Calculate mean in annual air temperature 
T_mu = ds['temp'].mean(dim='day')
T_mu.attrs = {
    'long_name': 'Mean annual 2-m air temperature', 
    'units': ds['temp'].units}
ds_season['T_mu'] = T_mu

# # Calculate seasonal amplitude in daily air temperature 
# # (and add to seasonal dataset)
# # NOTE: This needs to be refined to not simply use max/min
# T_amp = ds['temp'].max(dim='day') - ds['temp'].min(dim='day')
# T_amp.attrs = {
#     'long_name': 'Amplitude in annual 2-m air temperature', 
#     'units': ds['temp'].units}
# ds_season['T_amp'] = T_amp

# Calculate total annual precipitation and add to seasonal dataset
P_tot = ds['prcp'].sum(dim='day')
P_tot.attrs = {
    'long_name': 'Total annual precipitation', 
    'units': ds['prcp'].units}
ds_season['P_tot'] = P_tot

# %%[markdown]

# **Question: What period of time are you defining as the "warm season"?**
# **Also, does including info on the "cool" season influence the modeling results?**

# %% Import and format HAR static data

# Assign static variables of interest (and names to use)
vars_static = ['hgt']
static_names = ['har_elev']

# Import static variables
das_static = []
for var in vars_static:

    # Subset files to those matching regex of current variable
    var_regex = r".+_" + "static_" + re.escape(var)
    file = sorted(
        [fn for fn in har_fn if re.search(var_regex, fn)])

    # Import xarray
    da = xr.open_dataarray(
        DATA_DIR.joinpath('har-data', file[0]))

    # Drop time dimension (bc data are static)
    da_static = da.mean(dim='time')

    # Assign attributes
    da_static.attrs = da.attrs

    das_static.append(da_static)

# Combine static xr arrays into single xr dataset
ds_static = xr.Dataset(dict(zip(static_names, das_static)))

# # Combine ds_season results with ds_static
# ds_season = ds_season.merge(ds_static, compat='override')

##########
# Issues with mismatched lon/lat coordinates 
# (possibly rounding error?)
# Manually add array for elevation to ds_season
ds_season['har_elev'] = xr.DataArray(
    data=ds_static['har_elev'].data, coords=ds_season.coords, 
    attrs=ds_static['har_elev'].attrs, 
    name=ds_static['har_elev'].name)

##########

# %% Import and format RGI data

# Load glacier mb data
mb_df = pd.read_csv(DATA_DIR.joinpath(
    'mb-data/hma_mb_20190214_1015_nmad.csv'))

# Define custom crs (from Shean et al, 2020)
mb_crs = pyproj.CRS.from_proj4(
    '+proj=aea +lat_1=25 +lat_2=47 +lat_0=36 +lon_0=85 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs')

# Convert to epsg=4326
tmp_gdf = gpd.GeoDataFrame(
    mb_df.drop(['x','y'], axis=1), 
    geometry=gpd.points_from_xy(
        mb_df['x'],mb_df['y'], 
    crs=mb_crs))
tmp_gdf.to_crs(epsg=4326, inplace=True)

# Remove results outside the bounds of the HAR data
RGI = pd.DataFrame(
    tmp_gdf.drop('geometry', axis=1))
RGI['Lon'] = tmp_gdf.geometry.x
RGI['Lat'] = tmp_gdf.geometry.y
RGI.query(
    'Lat <= @ds_season.lat.max().values' 
    + '& Lat >= @ds_season.lat.min().values', 
    inplace=True)
RGI.query(
    'Lon <= @ds_season.lon.max().values' 
    + '& Lon >= @ds_season.lon.min().values', 
    inplace=True)

# Calculate hypsometric indices of glaciers
HI = (RGI['z_max']-RGI['z_med']) / (RGI['z_med']-RGI['z_min'])
HI[HI<1] = -1/HI[HI<1]
RGI['HI'] = HI

# Convert to gdf
RGI_gdf = gpd.GeoDataFrame(
    RGI.drop(['Lon','Lat'], axis=1), 
    geometry=gpd.points_from_xy(
        RGI['Lon'], RGI['Lat']), 
    crs="EPSG:4326")

# %% Extract HAR data at glacier locations

def get_nearest(
    src_points, candidates, k_neighbors=1):
    """
    Find nearest neighbors for all source points from a set of candidate points.
    src_points {pandas.core.frame.DataFrame}: Source locations to match to nearest neighbor in search set, with variables for longitude ('lon') and latitude ('lat'). Both should be prescribed in radians instead of degrees.
    candidates {pandas.core.frame.DataFrame}: Candidate locations in which to search for nearest neighbors, with variables for longitude ('lon') and latitude ('lat'). Both should be prescribed in radians rather than degrees.
    k_neighbors {int}: How many neighbors to return (defaults to 1 per source point).
    """

    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15, metric='haversine')

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Transpose to get distances and indices into arrays
    distances = np.squeeze(distances.transpose())
    indices = np.squeeze(indices.transpose())

    return indices, distances


def extract_at_pts(
    xr_ds, gdf_pts, coord_names=['lon','lat'], 
    return_dist=False, planet_radius=6371000):
    """
    Function where, given an xr-dataset and a Point-based geodataframe, 
    extract all values of variables in xr-dataset at pixels nearest 
    the given points in the geodataframe.
    xr_ds {xarray.core.dataset.Dataset}: Xarray dataset containing variables to extract.
    gdf_pts {geopandas.geodataframe.GeoDataFrame} : A Points-based geodataframe containing the locations at which to extract xrarray variables.
    coord_names {list}: The names of the longitude and latitude coordinates within xr_ds.
    return_dist {bool}: Whether function to append the distance (in meters) between the given queried points and the nearest raster pixel centroids. 
    NOTE: This assumes the xr-dataset includes lon/lat in the coordinates 
    (although they can be named anything, as this can be prescribed in the `coord_names` variable).
    """
    # Convert xr dataset to df and extract coordinates
    xr_df = xr_ds.to_dataframe().reset_index()
    xr_coord = xr_df[coord_names]

    # Ensure gdf_pts is in lon/lat and extract coordinates
    crs_end = gdf_pts.crs 
    gdf_pts.to_crs(epsg=4326, inplace=True)
    pt_coord = pd.DataFrame(
        {'Lon': gdf_pts.geometry.x, 
        'Lat': gdf_pts.geometry.y}).reset_index(drop=True)

    # Convert lon/lat points to RADIANS for both datasets
    xr_coord = xr_coord*np.pi/180
    pt_coord = pt_coord*np.pi/180

    # Find xr data nearest given points
    xr_idx, xr_dist = get_nearest(pt_coord, xr_coord)

    # Drop coordinate data from xr (leaves raster values)
    cols_drop = list(dict(xr_ds.coords).keys())
    xr_df_filt = xr_df.iloc[xr_idx].drop(
        cols_drop, axis=1).reset_index(drop=True)
    
    # Add raster values to geodf
    gdf_return = gdf_pts.reset_index(
        drop=True).join(xr_df_filt)
    
    # Add distance between raster center and points to gdf
    if return_dist:
        gdf_return['dist_m'] = xr_dist * planet_radius
    
    # Reproject results back to original projection
    gdf_return.to_crs(crs_end, inplace=True)

    return gdf_return

# Get climate variables from xarray dataset
gdf_clim = extract_at_pts(ds_season, RGI_gdf)

# Recalculate T_amp based on difference between
# mean summer T and mean winter T (addresses
# issue of single bad values biasing values)
gdf_clim['T_amp'] = gdf_clim['temp_JJA'] -  gdf_clim['temp_DJF']

# %% Convert seasonal precip to fractional precipitation
# Convert seasonal precipitation to fraction of total
gdf_clim['prcp_DJF'] = gdf_clim.apply(
    lambda row: row.prcp_DJF/row.P_tot, axis=1)
gdf_clim['prcp_JJA'] = gdf_clim.apply(
    lambda row: row.prcp_JJA/row.P_tot, axis=1)
gdf_clim['fracP_WARM'] = gdf_clim.apply(
    lambda row: row.prcp_WARM/row.P_tot, axis=1)

# %%

# gdf_plt = gdf_clim.sample(10000)
# clipping = {'min': 'red', 'max': 'orange'}

# # Mass balance map
# mb_min = np.quantile(gdf_clim.mb_mwea, 0.01)
# mb_max = np.quantile(gdf_clim.mb_mwea, 0.99)
# mb_plt = gv.Points(
#     data=gdf_plt, vdims=['mb_mwea']).opts(
#         color='mb_mwea', colorbar=True, cmap='coolwarm_r', 
#         symmetric=True, size=3, tools=['hover'], 
#         bgcolor='silver', 
#         width=600, height=500).redim.range(
#             mb_mwea=(mb_min,mb_max))

# # Temperature maps
# Tmu_min = np.quantile(gdf_clim.T_mu, 0.01)
# Tmu_max = np.quantile(gdf_clim.T_mu, 0.99)
# Tmu_plt = gv.Points(
#         data=gdf_plt, 
#         vdims=['T_mu']).opts(
#             color='T_mu', colorbar=True, 
#             cmap='bmy', clipping_colors=clipping, 
#             size=5, tools=['hover'], bgcolor='silver', 
#             width=600, height=500).redim.range(
#                 T_mu=(Tmu_min, Tmu_max))

# # Temperature difference map
# Tamp_min = np.quantile(gdf_clim.T_amp, 0.01)
# Tamp_max = np.quantile(gdf_clim.T_amp, 0.99)
# Tamp_plt = gv.Points(
#         data=gdf_plt, 
#         vdims=['T_amp']).opts(
#             color='T_amp', colorbar=True, 
#             cmap='fire', clipping_colors=clipping, 
#             size=5, tools=['hover'], bgcolor='silver', 
#             width=600, height=500).redim.range(
#                 T_amp=(Tamp_min, Tamp_max))

# # Elevation plot
# zmed_min = np.quantile(gdf_clim.z_med, 0.01)
# zmed_max = np.quantile(gdf_clim.z_med, 0.99)
# Zmed_plt = gv.Points(
#         data=gdf_plt, 
#         vdims=['z_med']).opts(
#             color='z_med', colorbar=True, 
#             cmap='bgyw', clipping_colors=clipping, 
#             size=5, tools=['hover'], bgcolor='silver', 
#             width=600, height=500).redim.range(
#                 z_med=(zmed_min, zmed_max))

# # Total precip plot
# P_max = np.quantile(gdf_clim.P_tot, 0.99)
# P_min = np.quantile(gdf_clim.P_tot, 0.01)
# Ptot_plt = gv.Points(
#     data=gdf_plt, 
#     vdims=['P_tot']).opts(
#         color='P_tot', colorbar=True, bgcolor='silver', 
#         cmap='viridis', size=5, tools=['hover'], 
#         width=600, height=500).redim.range(
#             P_tot=(0,P_max))

# # Winter precip plot
# DJFprcp_plt = gv.Points(
#     data=gdf_plt, 
#     vdims=['prcp_DJF']).opts(
#         color='prcp_DJF', colorbar=True, 
#         cmap='plasma', clipping_colors=clipping, 
#         size=5, tools=['hover'], bgcolor='silver', 
#         width=600, height=500).redim.range(prcp_DJF=(0,1))

# # Summer precip plot
# JJAprcp_plt = gv.Points(
#     data=gdf_plt, 
#     vdims=['prcp_JJA']).opts(
#         color='prcp_JJA', colorbar=True, 
#         cmap='plasma', clipping_colors=clipping, 
#         size=5, tools=['hover'], bgcolor='silver', 
#         width=600, height=500).redim.range(prcp_JJA=(0,1))


# # Temperature maps
# warmT_min = np.quantile(gdf_clim.T_mu, 0.01)
# warmT_max = np.quantile(gdf_clim.T_mu, 0.99)
# warmT_plt = gv.Points(
#         data=gdf_plt, 
#         vdims=['temp_WARM']).opts(
#             color='temp_WARM', colorbar=True, 
#             cmap='bmy', clipping_colors=clipping, 
#             size=5, tools=['hover'], bgcolor='silver', 
#             width=600, height=500).redim.range(
#                 T_mu=(warmT_min, warmT_max))

# # Fractional warm P plot
# warmPf_plt = gv.Points(
#     data=gdf_plt, 
#     vdims=['fracP_WARM']).opts(
#         color='fracP_WARM', colorbar=True, 
#         cmap='plasma', clipping_colors=clipping, 
#         size=5, tools=['hover'], bgcolor='silver', 
#         width=600, height=500).redim.range(fracP_WARM=(0,1))

# (
#     mb_plt  + warmT_plt + Tamp_plt + 
#     Zmed_plt + Ptot_plt + warmPf_plt).cols(3)

# %%

gdf_clim['area_km2'] = gdf_clim['area_m2'] / 1000**2

# Select variables of interest in modeling
gdf_glacier = gdf_clim.loc[:,
    ['RGIId', 'mb_mwea', 'geometry', 'area_km2', 
    'z_med', 'har_elev', 'z_slope', 'z_aspect', 
    'HI', 'T_mu', 'T_amp', 'temp_JJA', 'temp_WARM', 
    'P_tot', 'prcp_JJA', 'prcp_WARM', 'fracP_WARM']]

# %% Correct climate data based on per-cluster lapse rates

def correct_lapse(
    geodf, x_name, y_name, xTrue_name, y_others=None, 
    show_plts=False):
    """

    """

    # Find best fit temperature lapse rate
    X = geodf[x_name].to_numpy()
    y = geodf[y_name].to_numpy()
    reg = TheilSenRegressor().fit(X.reshape(-1,1), y)
    
    # Define variable lapse rate
    lapse_rate = reg.coef_[0]

    # Correct data based on lapse rate
    y_correct = y + lapse_rate*(
        geodf[xTrue_name].to_numpy() - X)
    
    # Add corrected values to new gdf
    new_df = geodf.copy()
    new_df[y_name] = y_correct

    if show_plts:
        # Diagnostic plot
        x_lin = np.linspace(X.min(), X.max())
        y_lin = reg.predict(x_lin.reshape(-1,1))
        plt.scatter(X,y, alpha=0.25)
        plt.plot(x_lin, y_lin, color='red')
        plt.xlabel(xTrue_name)
        plt.ylabel(y_name)
        plt.show()

        # Diagnostic plot
        plt.scatter(y, y_correct, alpha=0.25)
        plt.plot(
            [y_correct.min(), y.max()], 
            [y_correct.min(), y.max()], 
            color='black')
        plt.xlabel(y_name)
        plt.ylabel(y_name+' corrected')
        plt.show()

        print(f"Calculated lapse rate: {1000*lapse_rate:.3f} K/km")
    
    if y_others:
        for name in y_others:
            y = geodf[name].to_numpy()

            # Correct data based on lapse rate
            y_correct = y + lapse_rate*(
                geodf[xTrue_name].to_numpy() - X)

            # Add corrected values to geodf
            new_df[name] = y_correct

    return new_df

# %% Initial k-clustering to determine groups for lapse rates

# # Select climate features of interest for clustering
# clust_df = pd.DataFrame(gdf_glacier[
#     ['har_elev', 'T_mu', 'T_amp', 'temp_DJF', 
#     'temp_JJA', 'P_tot', 'prcp_JJA']])
# clust_df['Lon'] = gdf_glacier.geometry.x
# clust_df['Lat'] = gdf_glacier.geometry.y


# # Perform PCA
# pca = PCA()
# pca.fit(clust_df)

# # Select results that cumulatively explain at least 95% of variance
# pc_var = pca.explained_variance_ratio_.cumsum()
# pc_num = np.arange(
#     len(pc_var))[pc_var >= 0.95][0] + 1
# pca_df = pd.DataFrame(
#     pca.fit_transform(clust_df)).iloc[:,0:pc_num]

# # Cluster predictions
# k0 = 4
# grp_pred = KMeans(n_clusters=k0).fit_predict(clust_df)

# # Add cluster numbers to gdf
# clust_gdf = gdf_glacier.copy()
# clust_gdf['cluster'] = grp_pred

# # Reassign clusters to consistent naming convention
# # (KMeans randomly assigned cluster value)
# A_val = A_val = ord('A')
# alpha_dict = dict(
#     zip(np.arange(k0), 
#     [chr(char) for char in np.arange(A_val, A_val+k0)]))
# clust_alpha = [alpha_dict.get(item,item)  for item in grp_pred]

# # Add cluster groups to gdf
# clust_gdf['cluster'] = clust_alpha

# my_cmap = {
#     'A': '#e41a1c', 'B': '#377eb8', 'C': '#4daf4a', 
#     'D': '#984ea3', 'E': '#ff7f00', 'F': '#ffff33'}
# cluster0_plt = gv.Points(
#     data=clust_gdf.sample(10000), 
#     vdims=['cluster']).opts(
#         color='cluster', colorbar=True, 
#         cmap='Category10', 
#         # cmap=my_cmap, 
#         legend_position='bottom_left', 
#         size=5, tools=['hover'], width=750,
#         height=500)
# cluster0_plt

# %%

# vars_correct = [
#     'T_mu', 'temp_DJF', 'temp_JJA']
# clust_correct = clust_gdf.copy()
# for var in vars_correct:
#     print(f"Results for {var}...")
#     clust_correct = clust_correct.groupby('cluster').apply(
#         lambda x: correct_lapse(
#             x, x_name='har_elev', y_name=var, 
#             xTrue_name='z_med', show_plts=True))

# %%

# groups = clust_gdf.groupby('cluster')

# fig, ax = plt.subplots()
# for name, group in groups:
#     ax.plot(group.har_elev, group.T_mu, marker='o', linestyle='', ms=4, label=name, alpha=0.05)
# ax.legend()
# ax.set_xlabel('Har elevation')
# ax.set_ylabel('Mean annual T')

# fig, ax = plt.subplots()
# for name, group in groups:
#     ax.plot(group.har_elev, group.temp_DJF, marker='o', linestyle='', ms=4, label=name, alpha=0.05)
# ax.legend()
# ax.set_xlabel('Har elevation')
# ax.set_ylabel('Mean winter T')

# fig, ax = plt.subplots()
# for name, group in groups:
#     ax.plot(group.har_elev, group.temp_JJA, marker='o', linestyle='', ms=4, label=name, alpha=0.05)
# ax.legend()
# ax.set_xlabel('Har elevation')
# ax.set_ylabel('Mean summer T')

# %%

# # Determine if seasonal lapse rates differ from annual
# vars_correct = ['T_mu', 'temp_JJA', 'temp_DJF']
# clust_correct = gdf_glacier.copy()
# for var in vars_correct:
#     clust_correct = correct_lapse(
#             clust_correct, x_name='har_elev', y_name=var, 
#             xTrue_name='z_med', show_plts=True)

# %%[markdown]
# Based on these analyses, the cluster groups have minimal impact on determining temperature lapse rates.
# Slightly more important would be seasonal temperature lapse rates, but these are also fairly minor.
# For the time being, I will therefore simply use the mean annual temperature and the "warm season" mean temperature in modeling, and these are the only elevation corrections needed.
# 
# %%

gdf_correct = correct_lapse(
    gdf_clim, x_name='har_elev', y_name='T_mu', 
    xTrue_name='z_med')
gdf_correct = correct_lapse(
    gdf_correct, x_name='har_elev', y_name='T_mu', 
    xTrue_name='z_med')

# Drop deprecated variables
gdf_glacier = gdf_correct.loc[:,
    ['RGIId', 'mb_mwea', 'geometry', 'area_km2', 
    'z_med', 'z_slope', 'z_aspect', 
    'HI', 'T_mu', 'T_amp', 'temp_WARM', 
    'P_tot', 'prcp_JJA', 'prcp_WARM', 'fracP_WARM']]

# %%

gdf_plt = gdf_glacier.sample(10000)
clipping = {'min': 'red', 'max': 'orange'}

# Mass balance map
mb_min = np.quantile(gdf_glacier.mb_mwea, 0.01)
mb_max = np.quantile(gdf_glacier.mb_mwea, 0.99)
mb_plt = gv.Points(
    data=gdf_plt, vdims=['mb_mwea']).opts(
        color='mb_mwea', colorbar=True, cmap='coolwarm_r', 
        symmetric=True, size=3, tools=['hover'], 
        bgcolor='silver', 
        width=600, height=500).redim.range(
            mb_mwea=(mb_min,mb_max))

# Temperature maps
Tmu_min = np.quantile(gdf_glacier.T_mu, 0.01)
Tmu_max = np.quantile(gdf_glacier.T_mu, 0.99)
Tmu_plt = gv.Points(
        data=gdf_plt, 
        vdims=['T_mu']).opts(
            color='T_mu', colorbar=True, 
            cmap='bmy', clipping_colors=clipping, 
            size=5, tools=['hover'], bgcolor='silver', 
            width=600, height=500).redim.range(
                T_mu=(Tmu_min, Tmu_max))

Tw_min = np.quantile(gdf_glacier.temp_WARM, 0.01)
Tw_max = np.quantile(gdf_glacier.temp_WARM, 0.99)
Tw_plt = gv.Points(
        data=gdf_plt, 
        vdims=['temp_WARM']).opts(
            color='temp_WARM', colorbar=True, 
            cmap='bmy', clipping_colors=clipping, 
            size=5, tools=['hover'], bgcolor='silver', 
            width=600, height=500).redim.range(
                temp_WARM=(Tw_min, Tw_max))

# Temperature difference map
Tamp_min = np.quantile(gdf_glacier.T_amp, 0.01)
Tamp_max = np.quantile(gdf_glacier.T_amp, 0.99)
Tamp_plt = gv.Points(
        data=gdf_plt, 
        vdims=['T_amp']).opts(
            color='T_amp', colorbar=True, 
            cmap='fire', clipping_colors=clipping, 
            size=5, tools=['hover'], bgcolor='silver', 
            width=600, height=500).redim.range(
                T_amp=(Tamp_min, Tamp_max))

# Elevation plot
zmed_min = np.quantile(gdf_glacier.z_med, 0.01)
zmed_max = np.quantile(gdf_glacier.z_med, 0.99)
Zmed_plt = gv.Points(
        data=gdf_plt, 
        vdims=['z_med']).opts(
            color='z_med', colorbar=True, 
            cmap='bgyw', clipping_colors=clipping, 
            size=5, tools=['hover'], bgcolor='silver', 
            width=600, height=500).redim.range(
                z_med=(zmed_min, zmed_max))

# Total precip plot
P_max = np.quantile(gdf_glacier.P_tot, 0.99)
P_min = np.quantile(gdf_glacier.P_tot, 0.01)
Ptot_plt = gv.Points(
    data=gdf_plt, 
    vdims=['P_tot']).opts(
        color='P_tot', colorbar=True, bgcolor='silver', 
        cmap='viridis', size=5, tools=['hover'], 
        width=600, height=500).redim.range(
            P_tot=(0,P_max))

# WARM precip plot
P_max = np.quantile(gdf_clim.prcp_WARM, 0.99)
P_min = np.quantile(gdf_clim.prcp_WARM, 0.01)
warmP_plt = gv.Points(
    data=gdf_plt, 
    vdims=['prcp_WARM']).opts(
        color='prcp_WARM', colorbar=True, 
        cmap='plasma', clipping_colors=clipping, 
        size=5, tools=['hover'], bgcolor='silver', 
        width=600, height=500).redim.range(
            prcp_WARM=(P_min,P_max))

# WARM precip plot
WARMprcp_plt = gv.Points(
    data=gdf_plt, 
    vdims=['fracP_WARM']).opts(
        color='fracP_WARM', colorbar=True, 
        cmap='plasma', clipping_colors=clipping, 
        size=5, tools=['hover'], bgcolor='silver', 
        width=600, height=500).redim.range(fracP_WARM=(0,1))

(
    mb_plt  + Tw_plt + Tamp_plt + 
    Zmed_plt + Ptot_plt + WARMprcp_plt).cols(3)


# %%[markdown]
# ## Clustering data for improved spatial modeling
# 
# I need to determine if coherent spatial regions (based on location, topography, and climate attributes) can reasonably be discerned in the other attributes in order to justify using cluster groups to improve INLA modeling
# 
# %% PCA for dimensionality reduction of climate clusters

# Normalize data variables
norm_df = pd.DataFrame(gdf_glacier)
norm_df = norm_df.drop(['RGIId', 'geometry'], axis=1)
norm_df['Lon'] = gdf_glacier.geometry.x
norm_df['Lat'] = gdf_glacier.geometry.y
norm_df = (
    norm_df-norm_df.mean())/norm_df.std()

# Select only climate variables for pca
pca_clim = norm_df[
    ['temp_WARM', 'T_amp', 'P_tot', 'fracP_WARM', 
    'z_med', 'Lon', 'Lat']] 
    # 'prcp_JJA', 'T_mu', 'P_tot']] #No additional info added with these

# Perform PCA
pca = PCA()
pca.fit(pca_clim)

# Select results that cumulatively explain at least 95% of variance
pc_var = pca.explained_variance_ratio_.cumsum()
pc_num = np.arange(
    len(pc_var))[pc_var >= 0.99][0] + 1
pca_df = pd.DataFrame(
    pca.fit_transform(pca_clim)).iloc[:,0:pc_num]

# df of how features correlate with PCs
feat_corr = pd.DataFrame(
    pca.components_.T, index=pca_clim.columns)

# %%[markdown]
# Because the pca didn't sufficiently reduce the required dimensions, I will perform clustering on the raw normalized variables rather than on principle components
# 
# %%

# data_samp = gdf_glacier
data_samp = gdf_glacier.reset_index().sample(
    frac=0.33, random_state=777)

t0 = time.time()
Z = linkage(pca_clim.loc[data_samp.index,:], method='ward')
# Z = linkage(pca_df.loc[data_samp.index,:], method='ward')
t_end = time.time()
print(f"Agglomerative clustering time: {t_end-t0:.0f}s")

# %%

plt.figure(figsize=(30, 12))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='level', p=6, 
    color_threshold=175, #k=4
    leaf_font_size=12., 
    leaf_rotation=90.,  # rotates the x axis labels
)
plt.show()

# %%

# # Split results into desired clusters
# k = 5
# grp_pred = fcluster(Z, k, criterion='maxclust') - 1

# # Reassign clusters to consistent naming convention
# A_val = A_val = ord('A')
# alpha_dict = dict(
#     zip(np.arange(k), 
#     [chr(char) for char in np.arange(A_val, A_val+k)]))
# clust_alpha = [alpha_dict.get(item,item)  for item in grp_pred]
# data_samp['Group'] = clust_alpha

# # Plot of clusters
# clust_plt = gv.Points(
#     data=clust_gdf.sample(7000), 
#     vdims=['Group']).opts(
#         color='Group', colorbar=True, 
#         cmap='Category10', size=5, tools=['hover'], 
#         legend_position='bottom_left', 
#         bgcolor='silver', width=600, height=500)
# clust_plt

# %%

# Cluster predictions
k = 5
grp_pred = KMeans(n_clusters=k).fit_predict(pca_clim)
# grp_pred = KMeans(n_clusters=k).fit_predict(pca_df)

# Reassign clusters to consistent naming convention
A_val = A_val = ord('A')
alpha_dict = dict(
    zip(np.arange(k), 
    [chr(char) for char in np.arange(A_val, A_val+k)]))
clust_alpha = [alpha_dict.get(item,item)  for item in grp_pred]
gdf_glacier['Group'] = clust_alpha

# %%

# Rename groups based on mass balance
glacier_grps = gdf_glacier.groupby('Group')
mb_groups = glacier_grps.median()['z_med'].sort_values()
alpha_new = ['C', 'D', 'E', 'A', 'B']
A_dict = dict(zip(mb_groups.index, alpha_new))
gdf_glacier["Group"].replace(A_dict, inplace=True)

glacier_grps = gdf_glacier.groupby('Group')
cnt_ALL = glacier_grps.count() / gdf_glacier.shape[0]
print(cnt_ALL['RGIId'])

glac_feat = [
    'mb_mwea', 'z_med', 'HI', 'z_slope', 
    'z_aspect', 'area_km2']
clim_feat = [
    'temp_WARM', 'T_amp', 
    'P_tot', 'fracP_WARM']
print(glacier_grps.median()[glac_feat])
print(glacier_grps.median()[clim_feat])

# %%

import matplotlib.cm as cm
cm_tab10 = cm.get_cmap('tab10').colors
cm_alpha = [
    chr(el) for el in np.arange(A_val, A_val+len(cm_tab10))]
cat_cmap = dict(zip(cm_alpha, cm_tab10))

def var_plts(gdf_data, grouping_var, var_list, my_cmap):
    """
    Blah.
    """
    print(f"Cluster plots for {grouping_var} scheme")
    cluster_groups = gdf_data.groupby(grouping_var)

    nplt = len(var_list)
    ncol = 3
    nrow = int(np.floor(nplt/ncol))
    if nplt % ncol:
        nrow += 1

    fig, axs = plt.subplots(nrows=nrow, ncols=ncol)

    for i, var in enumerate(var_list):
        ax = axs.reshape(-1)[i]
        for key, group in cluster_groups:
            group[var].plot(ax=ax, kind='kde', 
                label=key, color=my_cmap[key], 
                legend=True)
        ax.set_xlim(
            (np.quantile(gdf_data[var], 0.005), 
            np.quantile(gdf_data[var], 0.995)))
        ax.set_xlabel(var)
    
    fig.set_size_inches((35,55))
    plt.show()

# %%

# Plot of clusters
clust_plt = gv.Points(
    data=gdf_glacier.sample(10000), 
    vdims=['Group']).opts(
        color='Group', colorbar=True, 
        cmap='Category10', size=5, tools=['hover'], 
        legend_position='bottom_left', 
        bgcolor='silver', width=600, height=500)
clust_plt

# %%

plt_vars = [
    'mb_mwea', 'z_med', 'HI', 
    'area_km2', 'z_slope', 'z_aspect', 
    'temp_WARM', 'P_tot', 'fracP_WARM']
var_plts(gdf_glacier, 'Group', plt_vars, cat_cmap)

# %%

# Save corrected data to disk
gdf_glacier.to_file(DATA_DIR.joinpath(
    'glacier-corrected.geojson'), driver='GeoJSON')

# %%
