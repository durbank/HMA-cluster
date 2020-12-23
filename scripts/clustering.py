# Script to perform clustering of climate variables for HMA

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
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import geoviews as gv
gv.extension('bokeh')
import holoviews as hv
hv.extension('bokeh')

# Environment setup
ROOT_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT_DIR.joinpath('data')

# %% Import and format HAR climate data

# Get list of all downloaded climate files
har_fn = [
    p.name for p in list(DATA_DIR.joinpath('har-data').glob('*.nc'))]

# Assign 2D variables of interest (and names to use)
vars_2d = ['t2', 'prcp']
var_names = ['temp', 'prcp']

# Format 2D data into mean daily value xr arrays
das = []
for var in vars_2d:

    # Subset files to those matching regex of current variable
    var_regex = r".+_" + re.escape(var) + r"_(\d)+.+"
    files = sorted([fn for fn in har_fn if re.search(var_regex, fn)])

    # Intialize first year as new array with days-of-year dim 
    tmp = xr.open_dataarray(DATA_DIR.joinpath('har-data', files[0]))
    da = xr.DataArray(
        data=tmp[0:365,:,:].data, 
        coords={
            'day': np.arange(0,365), 'south_north': tmp.south_north, 
            'west_east': tmp.west_east, 'lat':tmp.lat, 'lon':tmp.lon}, 
        dims=['day', 'south_north', 'west_east'], attrs=tmp.attrs)

    # Concatenate additional years of data into array along 'year' dim
    for file in files[1:]:
        tmp = xr.open_dataarray(DATA_DIR.joinpath('har-data', file))
        tmp = xr.DataArray(
            data=tmp[0:365,:,:].data, 
            coords={
                'day': np.arange(0,365), 'south_north': tmp.south_north, 
                'west_east': tmp.west_east, 'lat':tmp.lat, 'lon':tmp.lon}, 
            dims=['day', 'south_north', 'west_east'], attrs=tmp.attrs)
        da = xr.concat([da, tmp], 'year')

    # Compute mean daily value across all years
    da_clim = da.mean(dim='year')

    # Assign attributes
    da_clim.attrs = da.attrs
    da_clim.day.attrs = {
        'long_name': 'day of (365-day) year', 'units': '24-hour day'}

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

    # Calculate total seasonal precipitation
    da_P = ds['prcp'].sel(day=season).sum(dim='day')
    da_P.attrs = {
        'long_name': 'Total '+name_season[i]+' precipitation', 
        'units': ds['prcp'].units}
    das_season.append(da_P)

# Combine seasonal arrays to dataset
var_seasons = [
    'temp_DJF', 'prcp_DJF', 'temp_MAM', 'prcp_MAM', 
    'temp_JJA', 'prcp_JJA', 'temp_SON', 'prcp_SON']
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

# %% Import and format HAR static data

# Assign static variables of interest (and names to use)
vars_static = ['hgt']
static_names = ['har_elev']

# Import static variables
das_static = []
for var in vars_static:

    # Subset files to those matching regex of current variable
    var_regex = r".+_" + "static_" + re.escape(var)
    file = sorted([fn for fn in har_fn if re.search(var_regex, fn)])

    # Import xarray
    da = xr.open_dataarray(DATA_DIR.joinpath('har-data', file[0]))

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

# # Load glacier outlines
# RGI_files = list(DATA_DIR.joinpath('RGI-data').glob('*/*.shp'))
# gdfs = [gpd.read_file(path) for path in RGI_files]
# RGI_poly = gpd.GeoDataFrame(
#     pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)

# # Extract glacier features of interest
# RGI = pd.DataFrame(RGI_poly[
#     ['RGIId', 'GLIMSId', 'CenLon', 'CenLat', 
#     'Area', 'Zmed', 'Slope', 'Aspect', 'Lmax']])

# # Remove results outside the bounds of the HAR data
# RGI.query(
#     'CenLat <= @ds_season.lat.max().values' 
#     + '& CenLat >= @ds_season.lat.min().values', 
#     inplace=True)
# RGI.query(
#     'CenLon <= @ds_season.lon.max().values' 
#     + '& CenLon >= @ds_season.lon.min().values', 
#     inplace=True)

# RGI_gdf = gpd.GeoDataFrame(
#     RGI.drop(['CenLon','CenLat'], axis=1), 
#     geometry=gpd.points_from_xy(
#         RGI['CenLon'], RGI['CenLat']), 
#     crs="EPSG:4326")

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

# %% Perform initial clustering to determine k

# Normalize data variables
norm_df = pd.DataFrame(gdf_clim[
    ['z_med', 'z_min', 'z_max', 'z_slope', 
    'z_aspect', 'dhdt_ma', 'mb_mwea', 'area_m2', 
    'mb_m3wea', 'temp_DJF', 'prcp_DJF', 'temp_MAM', 
    'prcp_MAM', 'temp_JJA', 'prcp_JJA', 'temp_SON', 
    'prcp_SON', 'T_mu', 'T_amp', 'P_tot', 
    'har_elev']])
# norm_df = pd.DataFrame(
#     gdf_clim.drop(
#         ['RGIId', 'geometry'], 
#         axis=1))
norm_df['Lon'] = gdf_clim.geometry.x
norm_df['Lat'] = gdf_clim.geometry.y
norm_df = (
    norm_df-norm_df.mean())/norm_df.std()

## Perform k-means clustering on glacier data

# Select climate features of interest
clust_df = norm_df[
    ['T_mu', 'P_tot', 'temp_DJF', 'prcp_DJF', 
    'temp_MAM', 'prcp_MAM', 'temp_JJA', 'prcp_JJA', 
    'temp_SON', 'prcp_SON', 'Lat', 'Lon', 'z_med']]

# Perform PCA
pca = PCA()
pca.fit(clust_df)

# Select results that cumulatively explain at least 95% of variance
pc_var = pca.explained_variance_ratio_.cumsum()
pc_num = np.arange(
    len(pc_var))[pc_var >= 0.95][0] + 1
pca_df = pd.DataFrame(
    pca.fit_transform(clust_df)).iloc[:,0:pc_num]



ks = range(1,11)
scores = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit_predict(pca_df)
    scores.append(-model.score(pca_df))

plt.plot(ks, scores)
plt.ylabel('Total intra-cluster distance')
plt.xlabel('k')
plt.show()

# %% Initial k-clustering to determine groups for lapse rates

# Cluster predictions
grp_pred = KMeans(n_clusters=4).fit_predict(pca_df)

# Add cluster numbers to gdf
clust_gdf = gdf_clim.copy()
clust_gdf['cluster'] = grp_pred

# Reassign clusters to consistent naming convention
# (KMeans randomly assigned cluster value)
clust_num = clust_gdf.cluster.values
tmp = clust_gdf.groupby('cluster').mean()
clust_alpha = np.repeat('NA', len(clust_num))
clust_alpha[clust_num == tmp['z_med'].idxmax()] = 'A'
clust_alpha[
    clust_num == tmp['area_m2'].idxmax()] = 'B'
clust_alpha[
    clust_num == tmp['area_m2'].idxmin()] = 'D'
clust_alpha[clust_alpha == 'NA'] = 'C'
clust_gdf['cluster'] = clust_alpha

my_cmap = {
    'A': '#e41a1c', 'B': '#377eb8', 'C': '#4daf4a', 'D': '#984ea3', 'E': '#ff7f00', 'F': '#ffff33'}
# my_cmap = {
#     'A': '#66C2A5', 'B': '#FC8D62', 'C': '#8DA0CB', 'D': '#E78AC3'}
cluster0_plt = gv.Points(
    data=clust_gdf.sample(10000), vdims=['cluster']).opts(
        color='cluster', colorbar=True, cmap=my_cmap, 
        size=5, tools=['hover'], width=750,
        height=500)
cluster0_plt


# %% Compare HAR elev to RGI elev to determine biases

Z_res = gdf_clim.har_elev - gdf_clim.z_med
# Z_res.plot(kind='density')
print(Z_res.describe())
gdf_Zres = gpd.GeoDataFrame(
    data={'Z_res': Z_res}, geometry=gdf_clim.geometry, 
    crs=gdf_clim.crs)
elevRES_plt = gv.Points(
    data=gdf_Zres.sample(15000), vdims=['Z_res']).opts(
        color='Z_res', cmap='gwv_r', colorbar=True, 
        size=5, tools=['hover'], width=750,
        height=500).redim.range(Z_res=(-2000,2000))
elevRES_plt

# %% Additional plot for per-cluster biases

one_to_one = hv.Curve(
    data=pd.DataFrame(
        {'x':[0,7850], 'y':[0,7850]}))
scatt_elev = hv.Points(
    data=pd.DataFrame(clust_gdf), 
    kdims=['z_med', 'har_elev'], 
    vdims=['cluster']).groupby('cluster')
(
    one_to_one.opts(color='black') 
    * scatt_elev.opts(
        xlabel='RGI elevation (m)', 
        ylabel='HAR elevation (m)'))

# %% Correct climate data based on per-cluster lapse rates

def correct_lapse(
    geodf, x_name, y_name, xTrue_name, y_others=None, 
    show_plts=False):
    """

    """

    # Find best fit temperature lapse rate
    X = geodf[x_name].to_numpy()
    y = geodf[y_name].to_numpy()
    reg = LinearRegression().fit(X.reshape(-1,1), y)
    
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
        plt.scatter(X,y)
        plt.plot(x_lin, y_lin, color='red')
        plt.xlabel(xTrue_name)
        plt.ylabel(y_name)
        plt.show()

        # Diagnostic plot
        plt.scatter(y, y_correct)
        plt.plot(
            [y_correct.min(), y.max()], 
            [y_correct.min(), y.max()], 
            color='black')
        plt.xlabel(y_name)
        plt.ylabel(y_name+' corrected')
        plt.show()
    
    if y_others:
        for name in y_others:
            y = geodf[name].to_numpy()

            # Correct data based on lapse rate
            y_correct = y + lapse_rate*(
                geodf[xTrue_name].to_numpy() - X)

            # Add corrected values to geodf
            new_df[name] = y_correct

    return new_df


vars_correct = [
    'T_mu', 'temp_DJF', 'temp_MAM', 
    'temp_JJA', 'temp_SON']
# vars_correct = [
#     'T_mu', 'P_tot', 'temp_DJF', 'temp_MAM', 
#     'temp_JJA', 'temp_SON', 'prcp_DJF', 'prcp_MAM', 
#     'prcp_JJA', 'prcp_SON']
clust_correct = clust_gdf.copy()
for var in vars_correct:

    clust_correct = clust_correct.groupby('cluster').apply(
        lambda x: correct_lapse(
            x, x_name='har_elev', y_name=var, 
            xTrue_name='z_med'))


# Recalculate T_amp based on difference between
# corrected mean summer T and corrected 
# mean winter T
clust_correct['T_amp'] = (
    clust_correct['temp_JJA'] 
    - clust_correct['temp_DJF'])

# Drop deprecated variables
clust_correct.drop(
    ['har_elev', 'cluster'], axis=1, inplace=True)

# %% PCA for dimensionality reduction of climate clusters

# Normalize data variables
norm_df = pd.DataFrame(gdf_clim[
    ['z_med', 'z_min', 'z_max', 'z_slope', 
    'z_aspect', 'dhdt_ma', 'mb_mwea', 'area_m2', 
    'mb_m3wea', 'temp_DJF', 'prcp_DJF', 'temp_MAM', 
    'prcp_MAM', 'temp_JJA', 'prcp_JJA', 'temp_SON', 
    'prcp_SON', 'T_mu', 'T_amp', 'P_tot']])
norm_df['Lon'] = gdf_clim.geometry.x
norm_df['Lat'] = gdf_clim.geometry.y
norm_df = (
    norm_df-norm_df.mean())/norm_df.std()

# Select only climate variables for pca
pca_clim = norm_df[
    ['T_mu', 'T_amp', 'P_tot', 'temp_DJF', 
    'prcp_DJF', 'temp_MAM', 'prcp_MAM', 'temp_JJA', 
    'prcp_JJA', 'temp_SON', 'prcp_SON', 
    'z_med', 'Lon', 'Lat']]

# Perform PCA
pca = PCA()
pca.fit(pca_clim)

# Select results that cumulatively explain at least 95% of variance
pc_var = pca.explained_variance_ratio_.cumsum()
pc_num = np.arange(
    len(pc_var))[pc_var >= 0.95][0] + 1
pca_df = pd.DataFrame(
    pca.fit_transform(pca_clim)).iloc[:,0:pc_num]

# df of how features correlate with PCs
feat_corr = pd.DataFrame(
    pca.components_.T, index=pca_clim.columns)


# %%

from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(
    distance_threshold=0, n_clusters=None, linkage='ward')


data_samp = clust_correct.reset_index().sample(frac=0.50, random_state=101)
model = model.fit(pca_df.loc[data_samp.index,:])






# %%

ks = range(1,11)
scores = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit_predict(pca_df)
    scores.append(-model.score(pca_df))

plt.plot(ks, scores)
plt.ylabel('Total intra-cluster distance')
plt.xlabel('k')
plt.show()

# %%

# Cluster predictions based on PCs
model = KMeans(n_clusters=4)
grp_pred = model.fit_predict(pca_df)
score = -model.score(pca_df)

# Add cluster numbers to gdf
clust_gdf = clust_correct.copy()
clust_gdf['cluster'] = grp_pred

# Reassign clusters to consistent naming convention
# (KMeans randomly assigned cluster value)
clust_num = clust_gdf.cluster.values
tmp = clust_gdf.groupby('cluster').mean()
clust_alpha = np.repeat('NA', len(clust_num))
clust_alpha[clust_num == tmp['z_med'].idxmax()] = 'A'
clust_alpha[clust_num == tmp['z_med'].idxmin()] = 'B'
clust_alpha[clust_num == tmp['P_tot'].idxmax()] = 'C'
clust_alpha[clust_alpha == 'NA'] = 'D'
# clust_alpha[clust_num == tmp['z_med'].idxmax()] = 'A'
# clust_alpha[clust_num == tmp['z_med'].idxmin()] = 'B'
# clust_alpha[clust_num == tmp['area_m2'].idxmax()] = 'C'
# clust_alpha[clust_num == tmp['area_m2'].idxmin()] = 'D'
# clust_alpha[clust_num == tmp['P_tot'].idxmax()] = 'E'
# clust_alpha[clust_alpha == 'NA'] = 'F'
clust_gdf['cluster'] = clust_alpha

# Plot of clusters based on climate-only PCs
clim_clust_plt = gv.Points(
    data=clust_gdf.sample(10000), vdims=['cluster']).opts(
        color='cluster', colorbar=True, cmap=my_cmap, 
        size=5, tools=['hover'], width=750,
        height=500)

# %% Generate PCs and clusters based on glacier data

pca_glacier = norm_df[
    ['z_med', 'z_max', 'z_min', 'z_slope', 
    'z_aspect', 'mb_mwea', 'area_m2', 'Lon', 'Lat']]
# pca_glacier = norm_df[
#     ['z_med', 'z_min', 'z_max', 'z_slope', 'z_aspect', 
#     'dhdt_ma', 'mb_mwea', 'area_m2', 'mb_m3wea', 'Lon', 'Lat']]

# Perform PCA
pca2 = PCA()
pca2.fit(pca_glacier)

# Select results that cumulatively explain at least 95% of variance
pc_var2 = pca2.explained_variance_ratio_.cumsum()
pc_num2 = np.arange(
    len(pc_var2))[pc_var2 >= 0.95][0] + 1
pca_df2 = pd.DataFrame(
    pca2.fit_transform(pca_glacier)).iloc[:,0:pc_num2]

# %%

ks = range(1,11)
scores = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit_predict(pca_df2)
    scores.append(-model.score(pca_df2))

plt.plot(ks, scores)
plt.ylabel('Total intra-cluster distance')
plt.xlabel('k')
plt.show()

# %%

model2 = KMeans(n_clusters=4)
grp_pred2 = model2.fit_predict(pca_df2)
score2 = -model2.score(pca_df2)

# Add cluster numbers to gdf
clust_gdf2 = clust_correct.copy()
clust_gdf2['cluster'] = grp_pred2
# Reassign clusters to consistent naming convention
# (KMeans randomly assigned cluster value)
clust_num = clust_gdf2.cluster.values
tmp = clust_gdf2.groupby('cluster').mean()
clust_alpha = np.repeat('NA', len(clust_num))
clust_alpha[clust_num == tmp['mb_mwea'].idxmax()] = 'A'
clust_alpha[clust_num == tmp['mb_mwea'].idxmin()] = 'C'
clust_alpha[clust_num == tmp['z_med'].idxmax()] = 'D'
clust_alpha[clust_alpha == 'NA'] = 'B'
# clust_alpha[clust_num == tmp['z_med'].idxmax()] = 'A'
# clust_alpha[clust_num == tmp['z_med'].idxmin()] = 'B'
# clust_alpha[clust_num == tmp['area_m2'].idxmax()] = 'C'
# clust_alpha[clust_num == tmp['area_m2'].idxmin()] = 'D'
# clust_alpha[clust_num == tmp['dhdt_ma'].idxmin()] = 'E'
# clust_alpha[clust_alpha == 'NA'] = 'F'
clust_gdf2['cluster'] = clust_alpha

glacier_clust_plt = gv.Points(
    data=clust_gdf2.sample(10000), vdims=['cluster']).opts(
        color='cluster', colorbar=True, cmap=my_cmap, 
        size=5, tools=['hover'], width=750,
        height=500)

(clim_clust_plt + glacier_clust_plt)

# %% Cluster statistics and exploration

# Display cluster stats
clust_groups = clust_gdf.groupby('cluster')
# print(clust_groups1.mean().drop(cols_drop, axis=1))
clust_groups2 = clust_gdf2.groupby('cluster')
# print(clust_groups2.mean().drop(cols_drop, axis=1))

clust_res = (
    pd.DataFrame(clust_groups2.median()) 
    - pd.DataFrame(clust_groups.median())
    ) / pd.DataFrame(clust_groups.median())
print(clust_res)

cnt_1 = clust_groups.count() / clust_gdf.shape[0]
cnt_2 = clust_groups2.count() / clust_gdf2.shape[0]
grp_perc = pd.concat([cnt_1.iloc[:,1], cnt_2.iloc[:,1]], axis=1)
grp_perc.columns = ['CLIM', 'GLACIER']
print(grp_perc)

# plt_vars = clust_df.columns
plt_vars = [
    'z_med', 'prcp_DJF', 'prcp_MAM', 'prcp_JJA', 'prcp_SON', 
    'P_tot', 'mb_mwea']
for var in plt_vars:
    fig, ax = plt.subplots()
    for key, group in clust_groups:
        group[var].plot(ax=ax, kind='kde', 
            label=key, color=my_cmap[key], 
            legend=True)
    for key, group in clust_groups2:
        group[var].plot(ax=ax, kind='kde', 
        label=key, color=my_cmap[key], 
        linestyle='--', legend=False)
    ax.set_xlim(
        (np.min(clust_groups.min()[var]), 
        np.max(clust_groups.max()[var])))
    ax.set_xlabel(var)
    fig.set_size_inches((12,8))
    plt.show()

# clust_groups['T_mu'].plot(kind='kde', legend=True)
# clust_groups['P_tot'].plot(kind='kde', legend=True)


# %%

# Find fraction of P_tot for each season
Pfrac_gdf = clust_gdf.copy()
Pfrac_gdf['prcp_DJF'] = clust_gdf.apply(
    lambda row: row.prcp_DJF/row.P_tot, axis=1)
Pfrac_gdf['prcp_MAM'] = clust_gdf.apply(
    lambda row: row.prcp_MAM/row.P_tot, axis=1)
Pfrac_gdf['prcp_JJA'] = clust_gdf.apply(
    lambda row: row.prcp_JJA/row.P_tot, axis=1)
Pfrac_gdf['prcp_SON'] = clust_gdf.apply(
    lambda row: row.prcp_SON/row.P_tot, axis=1)


clipping = {'min': 'red'}

DJF_frac_plt = gv.Points(
    data=Pfrac_gdf.sample(10000), 
    vdims=['prcp_DJF']).opts(
        color='prcp_DJF', colorbar=True, 
        cmap='viridis', clipping_colors=clipping, 
        size=5, tools=['hover'], width=750, height=500)

DJF_frac_plt.redim.range(prcp_DJF=(0,1))

# %%
MAM_frac_plt = gv.Points(
    data=Pfrac_gdf.sample(10000), 
    vdims=['prcp_MAM']).opts(
        color='prcp_MAM', colorbar=True, 
        cmap='viridis', clipping_colors=clipping, 
        size=5, tools=['hover'], width=750, height=500)
MAM_frac_plt.redim.range(prcp_MAM=(0,1))

# %%
JJA_frac_plt = gv.Points(
    data=Pfrac_gdf.sample(10000), 
    vdims=['prcp_JJA']).opts(
        color='prcp_JJA', colorbar=True, 
        cmap='viridis', clipping_colors=clipping, 
        size=5, tools=['hover'], width=750, height=500)
JJA_frac_plt.redim.range(prcp_JJA=(0,1))

# %%
SON_frac_plt = gv.Points(
    data=Pfrac_gdf.sample(10000), 
    vdims=['prcp_SON']).opts(
        color='prcp_SON', colorbar=True, 
        cmap='viridis', clipping_colors=clipping, 
        size=5, tools=['hover'], width=750, height=500)
SON_frac_plt.redim.range(prcp_SON=(0,1))

# %%
P_max = np.quantile(Pfrac_gdf.P_tot, 0.99)
P_min = np.quantile(Pfrac_gdf.P_tot, 0.01)
Ptot_plt = gv.Points(
    data=Pfrac_gdf.sample(10000), 
    vdims=['P_tot']).opts(
        color='P_tot', colorbar=True, 
        cmap='viridis', size=5, tools=['hover'], 
        width=750, height=500).redim.range(
            P_tot=(P_min,P_max))
Ptot_plt




# %% Clustering based on all numeric features

# Perform PCA
pca_ALL = PCA()
pca_ALL.fit(norm_df)

# Select results that cumulatively explain at least 95% of variance
pc_var_ALL = pca_ALL.explained_variance_ratio_.cumsum()
pc_num_ALL = np.arange(
    len(pc_var_ALL))[pc_var_ALL >= 0.95][0] + 1
pcaALL_df = pd.DataFrame(
    pca_ALL.fit_transform(norm_df)).iloc[:,0:pc_num]

# df of how features correlate with PCs
feat_corr = pd.DataFrame(
    pca_ALL.components_.T, index=norm_df.columns)

# Cluster predictions based on PCs (best k could be 3,4,6, or 7)
model = KMeans(n_clusters=4)
grp_pred = model.fit_predict(pcaALL_df)
score = -model.score(pcaALL_df)

# Add cluster numbers to gdf
clust_gdf_ALL = clust_correct.copy()
clust_gdf_ALL['cluster'] = grp_pred

# Reassign clusters to consistent naming convention
# (KMeans randomly assigned cluster value)
clust_num = clust_gdf_ALL.cluster.values
tmp = clust_gdf_ALL.groupby('cluster').mean()
clust_alpha = np.repeat('NA', len(clust_num))
clust_alpha[clust_num == tmp['z_med'].idxmax()] = 'A'
clust_alpha[clust_num == tmp['z_med'].idxmin()] = 'B'
clust_alpha[clust_num == tmp['P_tot'].idxmax()] = 'C'
clust_alpha[clust_alpha == 'NA'] = 'D'
clust_gdf_ALL['cluster'] = clust_alpha

# Plot of clusters based on climate-only PCs
clust_ALL_plt = gv.Points(
    data=clust_gdf_ALL.sample(10000), vdims=['cluster']).opts(
        color='cluster', colorbar=True, cmap=my_cmap, 
        size=5, tools=['hover'], width=750,
        height=500)






# %%

# Cluster A: 
# - Represents 40% of glaciers in the dataset
# - High elevation, cold, and dry
# - Moderate-sized glaciers
# - Mostly confined to western HMA
# - Greatest precip in spring (closely followed by winter)

# Cluster B:
# - Represents 22% of glaciers in dataset
# - Low elevation, mid-temperature, wetter
# - Larger glaciers
# - Westernmost cluster of glaciers
# - Greatest precip in winter, closely followed by spring

# Cluster C:
# - Represents 6% of glaciers in dataset
# - Mid elevation, warmest, wettest
# - NOTE: HAR has this group as the lowest elevation (although looking at both distributions, the bulk of samples are pretty comparable)
# - Small seasonal cycle in temperature
# - Moderate-sized glaciers
# - Consists of southern band of HMA
# - Precip dominated by summer contribution

# Cluster D:
# - Represents 32% of glaciers in dataset
# - Higher elevation, modest temperature, dry
# - Small glaciers
# - Mostly eastern HMA, but scattered throughout
# - Precip dominated by summer contribution










# import seaborn as sns

# clust_sns_df = clust_gdf[
#     ['Area', 'Zmed', 'temp_DJF', 'temp_JJA', 'T_mu', 'T_amp', 
#     'prcp_DJF', 'prcp_JJA', 'P_tot', 'cluster']]
# plt_hist = sns.pairplot(
#     clust_sns_df.sample(frac=0.25), kind='kde', 
#     hue='cluster', corner=True)

# customPalette = sns.set_palette(sns.color_palette(list(my_cmap.values())))
# sns.kdeplot(
#     data=clust_gdf, x='Zmed', hue='cluster', 
#     fill=True)


# from scipy.stats import probplot

# # Exploration of general data
# pplt_geo = sns.pairplot(
#     norm_df[
#         ['Lon', 'Lat', 'Zmed', 'T_mu', 
#         'T_amp', 'P_tot']].sample(frac=0.10), 
#     kind="kde", corner=True)

# # Exploration of temp data
# # probplot(norm_df['T_amp'], plot=plt)
# pplt_temp = sns.pairplot(
#     norm_df[
#         ['T_mu', 'T_amp', 'temp_DJF', 'temp_MAM', 
#         'temp_JJA', 'temp_SON']].sample(frac=0.10), 
#     kind="kde", corner=True)

# # Exploration of prcp data
# ppt_prcp = sns.pairplot(
#     norm_df[
#         ['P_tot', 'prcp_DJF', 'prcp_MAM', 
#         'prcp_JJA', 'prcp_SON']].sample(frac=0.10), 
#     kind="kde", corner=True)
