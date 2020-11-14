# Script to perform clustering of climate variables for HMA

import re
import pandas as pd
import geopandas as gpd
import xarray as xr
import numpy as np
from pathlib import Path

# Environment setup
ROOT_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT_DIR.joinpath('data')

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

# Convert prcp from rainfall intensity to total precipitation
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
    'temp_DJF', 'temp_MAM', 'temp_JJA', 'temp_SON', 
    'prcp_DJF', 'prcp_MAM', 'prcp_JJA', 'prcp_SON']
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



# Load glacier outlines
RGI_files = list(DATA_DIR.joinpath('RGI-data').glob('*/*.shp'))
gdfs = [gpd.read_file(path) for path in RGI_files]

RGI_poly = gpd.GeoDataFrame(
    pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)

RGI = pd.DataFrame(RGI_poly[
    ['RGIId', 'GLIMSId', 'CenLon', 'CenLat', 
    'Area', 'Zmed', 'Slope', 'Aspect', 'Lmax']])

# Remove results outside the bounds of the HAR data
RGI.query(
    'CenLat <= @ds_season.lat.max().values' 
    + '& CenLat >= @ds_season.lat.min().values', 
    inplace=True)
RGI.query(
    'CenLon <= @ds_season.lon.max().values' 
    + '& CenLon >= @ds_season.lon.min().values', 
    inplace=True)

RGI_gdf = gpd.GeoDataFrame(
    RGI.drop(['CenLon','CenLat'], axis=1), 
    geometry=gpd.points_from_xy(
        RGI['CenLon'], RGI['CenLat']), 
    crs="EPSG:4326")

# Functions for nearest neighbor search in xarray
from sklearn.neighbors import BallTree
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

## Compare HAR elev to RGI elev to determine biases

Z_res = gdf_clim.har_elev - gdf_clim.Zmed
Z_res.plot(kind='density')
print(Z_res.describe())
gdf_Zres = gpd.GeoDataFrame(
    data={'Z_res': Z_res}, geometry=gdf_clim.geometry, 
    crs=gdf_clim.crs)
# gdf_Zres.plot(column='Z_res', legend=True)


# import geoviews as gv
# gv.extension('bokeh')
# gv.Points(
#     data=gdf_Zres, vdims=['Z_res']).opts(
#         color='Z_res', cmap='gwv_r', colorbar=True, 
#         size=5, tools=['hover'], width=750,
#         height=500).redim.range(Z_res=(-2000,2000))


## Explore dimensionality reduction to select variables

from sklearn import decomposition as decomp

# Normalize data variables
norm_df = pd.DataFrame(
    gdf_clim.drop(
        ['RGIId', 'GLIMSId', 'geometry', 'dist_m'], 
        axis=1))
norm_df['Lon'] = gdf_clim.geometry.x
norm_df['Lat'] = gdf_clim.geometry.y
norm_df = (
    norm_df-norm_df.mean())/norm_df.std()


pca = decomp.PCA()
pca.fit(norm_df)
X = pca.transform(norm_df)


## Perform k-means clustering on glacier data

# Additional module loading
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt


clust_df = norm_df[
    ['Zmed', 'T_mu', 'T_amp', 'temp_DJF', 'temp_JJA', 
    'P_tot', 'prcp_DJF', 'prcp_JJA', 'Lon', 'Lat']]
# clust_df.drop(['Lon', 'Lat'], axis=1, inplace=True)

ks = range(1,11)
scores = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit_predict(clust_df)
    scores.append(-model.score(clust_df))

plt.plot(ks, scores)
plt.ylabel('Total intra-cluster distance')
plt.xlabel('k')
plt.show()



grp_pred = KMeans(n_clusters=4).fit_predict(clust_df)

clust_gdf = gdf_clim.copy()
clust_gdf['cluster'] = grp_pred

clust_gdf.sample(10000).plot(column='cluster')










import seaborn as sns

from scipy.stats import probplot

# Exploration of general data
pplt_geo = sns.pairplot(
    norm_df[
        ['Lon', 'Lat', 'Zmed', 'T_mu', 
        'T_amp', 'P_tot']].sample(frac=0.10), 
    kind="kde", corner=True)

# Exploration of temp data
# probplot(norm_df['T_amp'], plot=plt)
pplt_temp = sns.pairplot(
    norm_df[
        ['T_mu', 'T_amp', 'temp_DJF', 'temp_MAM', 
        'temp_JJA', 'temp_SON']].sample(frac=0.10), 
    kind="kde", corner=True)

# Exploration of prcp data
ppt_prcp = sns.pairplot(
    norm_df[
        ['P_tot', 'prcp_DJF', 'prcp_MAM', 
        'prcp_JJA', 'prcp_SON']].sample(frac=0.10), 
    kind="kde", corner=True)





## Exploratory dimensionality reduction with PCA

# New variable subset based on exploration
pca_df = norm_df[
    ['Lon', 'Lat', 'Zmed', 'T_mu', 'T_amp', 'P_tot', 
    'temp_DJF', 'temp_MAM', 'temp_JJA', 'temp_SON', 
    'prcp_DJF', 'prcp_MAM', 'prcp_JJA', 'prcp_SON']]
