# This script prepares data for importing into R for use in INLA modeling.

# %% Set environment

# Import modules
import re
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
vars_2d = ['t2', 'prcp', 'netrad']
var_names = ['temp', 'prcp', 'netrad']

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
MAM = np.arange(60,152)
JJA = np.arange(152,244)
SON = np.arange(244,335)

# Create xr-arrays for single-valued variables 
# (temp amplitude, seasonal means/totals, etc.)
das_season = []
seasons = [DJF, MAM, JJA, SON]
name_season = ['winter', 'spring', 'summer', 'autumn']
for i, season in enumerate(seasons):
    # Calculate mean seasonal air temperature
    da_T = ds['temp'].sel(day=season).mean(dim='day')
    da_T.attrs =  {
        'long_name': 'Mean '+name_season[i]+' 2-m air temperature',
        'units': ds['temp'].units}
    das_season.append(da_T)

    # Calculate mean seasonal net radiation
    da_R = ds['netrad'].sel(day=season).mean(dim='day')
    da_R.attrs =  {
        'long_name': 'Mean '+name_season[i]+' net radiation',
        'units': ds['netrad'].units}
    das_season.append(da_R)

    # Calculate total seasonal precipitation
    da_P = ds['prcp'].sel(day=season).sum(dim='day')
    da_P.attrs = {
        'long_name': 'Total '+name_season[i]+' precipitation', 
        'units': ds['prcp'].units}
    das_season.append(da_P)

# Combine seasonal arrays to dataset
var_seasons = [
    'temp_DJF', 'rad_DJF', 'prcp_DJF', 
    'temp_MAM', 'rad_MAM', 'prcp_MAM',  
    'temp_JJA', 'rad_JJA', 'prcp_JJA', 
    'temp_SON', 'rad_SON', 'prcp_SON']
ds_season = xr.Dataset(dict(zip(var_seasons, das_season)))

# Calculate mean in annual air temperature 
T_mu = ds['temp'].mean(dim='day')
T_mu.attrs = {
    'long_name': 'Mean annual 2-m air temperature', 
    'units': ds['temp'].units}
ds_season['T_mu'] = T_mu

# Calculate seasonal amplitude in daily air temperature 
# (and add to seasonal dataset)
# NOTE: This needs to be refined to not simply use max/min
T_amp = ds['temp'].max(dim='day') - ds['temp'].min(dim='day')
T_amp.attrs = {
    'long_name': 'Amplitude in annual 2-m air temperature', 
    'units': ds['temp'].units}
ds_season['T_amp'] = T_amp

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

# # Remove 1% of glaciers that are smallest/largest in area
# # (eliminates tiny cluster of massive glaciers)
# a_min = np.quantile(RGI.area_m2, 0.001)
# a_max = np.quantile(RGI.area_m2, 0.995)
# RGI.query(
#     'area_m2 >= @a_min & area_m2 <= @a_max', 
#     inplace=True)

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

# Remove glaciers with 0 annual prcp (bad modeling)
gdf_clim.query('P_tot > 0', inplace=True)

# %% Convert seasonal precip to fractional precipitation
# Convert seasonal precipitation to fraction of total
gdf_clim['prcp_DJF'] = gdf_clim.apply(
    lambda row: row.prcp_DJF/row.P_tot, axis=1)
gdf_clim['prcp_MAM'] = gdf_clim.apply(
    lambda row: row.prcp_MAM/row.P_tot, axis=1)
gdf_clim['prcp_JJA'] = gdf_clim.apply(
    lambda row: row.prcp_JJA/row.P_tot, axis=1)
gdf_clim['prcp_SON'] = gdf_clim.apply(
    lambda row: row.prcp_SON/row.P_tot, axis=1)

# %%

gdf_plt = gdf_clim.sample(10000)
clipping = {'min': 'red', 'max': 'orange'}

# Mass balance map
mb_min = np.quantile(gdf_clim.mb_mwea, 0.01)
mb_max = np.quantile(gdf_clim.mb_mwea, 0.99)
mb_plt = gv.Points(
    data=gdf_plt, vdims=['mb_mwea']).opts(
        color='mb_mwea', colorbar=True, cmap='coolwarm_r', 
        symmetric=True, size=3, tools=['hover'], 
        bgcolor='silver', 
        width=600, height=500).redim.range(
            mb_mwea=(mb_min,mb_max))

# Temperature maps
Tmu_min = np.quantile(gdf_clim.T_mu, 0.01)
Tmu_max = np.quantile(gdf_clim.T_mu, 0.99)
Tmu_plt = gv.Points(
        data=gdf_plt, 
        vdims=['T_mu']).opts(
            color='T_mu', colorbar=True, 
            cmap='bmy', clipping_colors=clipping, 
            size=5, tools=['hover'], bgcolor='silver', 
            width=600, height=500).redim.range(
                T_mu=(Tmu_min, Tmu_max))

# Temperature difference map
Tamp_min = np.quantile(gdf_clim.T_amp, 0.01)
Tamp_max = np.quantile(gdf_clim.T_amp, 0.99)
Tamp_plt = gv.Points(
        data=gdf_plt, 
        vdims=['T_amp']).opts(
            color='T_amp', colorbar=True, 
            cmap='fire', clipping_colors=clipping, 
            size=5, tools=['hover'], bgcolor='silver', 
            width=600, height=500).redim.range(
                T_amp=(Tamp_min, Tamp_max))

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

# Total precip plot
P_max = np.quantile(gdf_clim.P_tot, 0.99)
P_min = np.quantile(gdf_clim.P_tot, 0.01)
Ptot_plt = gv.Points(
    data=gdf_plt, 
    vdims=['P_tot']).opts(
        color='P_tot', colorbar=True, bgcolor='silver', 
        cmap='viridis', size=5, tools=['hover'], 
        width=600, height=500).redim.range(
            P_tot=(0,P_max))

# Winter precip plot
DJFprcp_plt = gv.Points(
    data=gdf_plt, 
    vdims=['prcp_DJF']).opts(
        color='prcp_DJF', colorbar=True, 
        cmap='plasma', clipping_colors=clipping, 
        size=5, tools=['hover'], bgcolor='silver', 
        width=600, height=500).redim.range(prcp_DJF=(0,1))

# Summer precip plot
JJAprcp_plt = gv.Points(
    data=gdf_plt, 
    vdims=['prcp_JJA']).opts(
        color='prcp_JJA', colorbar=True, 
        cmap='plasma', clipping_colors=clipping, 
        size=5, tools=['hover'], bgcolor='silver', 
        width=600, height=500).redim.range(prcp_JJA=(0,1))

(
    mb_plt  + Tmu_plt + Tamp_plt
    + Ptot_plt + DJFprcp_plt + JJAprcp_plt).cols(3)

# %%

gdf_clim['area_km2'] = gdf_clim['area_m2'] / 1000**2

# Select variables of interest in modeling
gdf_glacier = gdf_clim.loc[:,
    ['RGIId', 'mb_mwea', 'geometry', 'area_km2', 
    'z_med', 'har_elev', 'z_slope', 'z_aspect', 
    'HI', 'T_mu', 'T_amp', 'temp_DJF', 'temp_JJA', 
    'P_tot', 'prcp_JJA']]

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

# Select climate features of interest for clustering
clust_df = pd.DataFrame(gdf_glacier[
    ['har_elev', 'T_mu', 'T_amp', 'temp_DJF', 
    'temp_JJA', 'P_tot', 'prcp_JJA']])
clust_df['Lon'] = gdf_glacier.geometry.x
clust_df['Lat'] = gdf_glacier.geometry.y


# # Perform PCA
# pca = PCA()
# pca.fit(clust_df)

# # Select results that cumulatively explain at least 95% of variance
# pc_var = pca.explained_variance_ratio_.cumsum()
# pc_num = np.arange(
#     len(pc_var))[pc_var >= 0.95][0] + 1
# pca_df = pd.DataFrame(
#     pca.fit_transform(clust_df)).iloc[:,0:pc_num]

# Cluster predictions
k0 = 4
grp_pred = KMeans(n_clusters=k0).fit_predict(clust_df)

# Add cluster numbers to gdf
clust_gdf = gdf_glacier.copy()
clust_gdf['cluster'] = grp_pred

# Reassign clusters to consistent naming convention
# (KMeans randomly assigned cluster value)
A_val = A_val = ord('A')
alpha_dict = dict(
    zip(np.arange(k0), 
    [chr(char) for char in np.arange(A_val, A_val+k0)]))
clust_alpha = [alpha_dict.get(item,item)  for item in grp_pred]

# Add cluster groups to gdf
clust_gdf['cluster'] = clust_alpha

my_cmap = {
    'A': '#e41a1c', 'B': '#377eb8', 'C': '#4daf4a', 
    'D': '#984ea3', 'E': '#ff7f00', 'F': '#ffff33'}
cluster0_plt = gv.Points(
    data=clust_gdf.sample(10000), 
    vdims=['cluster']).opts(
        color='cluster', colorbar=True, 
        cmap='Category10', 
        # cmap=my_cmap, 
        legend_position='bottom_left', 
        size=5, tools=['hover'], width=750,
        height=500)
cluster0_plt

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

groups = clust_gdf.groupby('cluster')

fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.har_elev, group.T_mu, marker='o', linestyle='', ms=4, label=name, alpha=0.05)
ax.legend()
ax.set_xlabel('Har elevation')
ax.set_ylabel('Mean annual T')

fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.har_elev, group.temp_DJF, marker='o', linestyle='', ms=4, label=name, alpha=0.05)
ax.legend()
ax.set_xlabel('Har elevation')
ax.set_ylabel('Mean winter T')

fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.har_elev, group.temp_JJA, marker='o', linestyle='', ms=4, label=name, alpha=0.05)
ax.legend()
ax.set_xlabel('Har elevation')
ax.set_ylabel('Mean summer T')

# %%

# Determine if seasonal lapse rates differ from annual
vars_correct = ['T_mu', 'temp_JJA', 'temp_DJF']
clust_correct = gdf_glacier.copy()
for var in vars_correct:
    clust_correct = correct_lapse(
            clust_correct, x_name='har_elev', y_name=var, 
            xTrue_name='z_med', show_plts=True)

# %%[markdown]
# Based on these analyses, the cluster groups have minimal impact on determining temperature lapse rates.
# Slightly more important would be seasonal temperature lapse rates, but these are also fairly minor.
# For the time being, I will therefore simply use the mean annual temperature in modeling, and that is the only variable needing correction.
# 
# %%

# Select variables of interest in modeling
gdf_glacier = gdf_clim.loc[:,
    ['RGIId', 'mb_mwea', 'geometry', 'area_km2', 
    'z_med', 'har_elev', 'z_slope', 'z_aspect', 
    'HI', 'T_mu', 'T_amp', 'P_tot', 'prcp_JJA']]

gdf_correct = correct_lapse(
    gdf_glacier, x_name='har_elev', y_name='T_mu', 
    xTrue_name='z_med')

# Drop deprecated variables
gdf_correct.drop(['har_elev'], axis=1, inplace=True)

# %%

gdf_plt = gdf_correct.sample(10000)
clipping = {'min': 'red', 'max': 'orange'}

# Mass balance map
mb_min = np.quantile(gdf_correct.mb_mwea, 0.01)
mb_max = np.quantile(gdf_correct.mb_mwea, 0.99)
mb_plt = gv.Points(
    data=gdf_plt, vdims=['mb_mwea']).opts(
        color='mb_mwea', colorbar=True, cmap='coolwarm_r', 
        symmetric=True, size=3, tools=['hover'], 
        bgcolor='silver', 
        width=600, height=500).redim.range(
            mb_mwea=(mb_min,mb_max))

# Temperature maps
Tmu_min = np.quantile(gdf_correct.T_mu, 0.01)
Tmu_max = np.quantile(gdf_correct.T_mu, 0.99)
Tmu_plt = gv.Points(
        data=gdf_plt, 
        vdims=['T_mu']).opts(
            color='T_mu', colorbar=True, 
            cmap='bmy', clipping_colors=clipping, 
            size=5, tools=['hover'], bgcolor='silver', 
            width=600, height=500).redim.range(
                T_mu=(Tmu_min, Tmu_max))

# Temperature difference map
Tamp_min = np.quantile(gdf_correct.T_amp, 0.01)
Tamp_max = np.quantile(gdf_correct.T_amp, 0.99)
Tamp_plt = gv.Points(
        data=gdf_plt, 
        vdims=['T_amp']).opts(
            color='T_amp', colorbar=True, 
            cmap='fire', clipping_colors=clipping, 
            size=5, tools=['hover'], bgcolor='silver', 
            width=600, height=500).redim.range(
                T_amp=(Tamp_min, Tamp_max))

# Elevation plot
zmed_min = np.quantile(gdf_correct.z_med, 0.01)
zmed_max = np.quantile(gdf_correct.z_med, 0.99)
Zmed_plt = gv.Points(
        data=gdf_plt, 
        vdims=['z_med']).opts(
            color='z_med', colorbar=True, 
            cmap='bgyw', clipping_colors=clipping, 
            size=5, tools=['hover'], bgcolor='silver', 
            width=600, height=500).redim.range(
                z_med=(zmed_min, zmed_max))

# Total precip plot
P_max = np.quantile(gdf_correct.P_tot, 0.99)
P_min = np.quantile(gdf_correct.P_tot, 0.01)
Ptot_plt = gv.Points(
    data=gdf_plt, 
    vdims=['P_tot']).opts(
        color='P_tot', colorbar=True, bgcolor='silver', 
        cmap='viridis', size=5, tools=['hover'], 
        width=600, height=500).redim.range(
            P_tot=(0,P_max))

# Summer precip plot
JJAprcp_plt = gv.Points(
    data=gdf_plt, 
    vdims=['prcp_JJA']).opts(
        color='prcp_JJA', colorbar=True, 
        cmap='plasma', clipping_colors=clipping, 
        size=5, tools=['hover'], bgcolor='silver', 
        width=600, height=500).redim.range(prcp_JJA=(0,1))

(
    mb_plt  + Zmed_plt + Ptot_plt
    + Tmu_plt + Tamp_plt + JJAprcp_plt).cols(3)

# %%
