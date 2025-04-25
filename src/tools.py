import re
import os 
import xarray as xr 
import pandas as pd
import numpy as np
from typing import Pattern, List
import gcpy.constants as gcon
import glob
import gcgridobj 
import regionmask 
import matplotlib.pyplot as plt 
from shapely.geometry import Point, Polygon

# DEFINE DIRECTORIES HERE
MODEL_OUTPUT_DIR = "/home/eleroy/proj-dirs/OZCLIM/data/analysis_data/GCHP_CAM_public/"
CASTNET_DIR = "/home/eleroy/proj-dirs/OZCLIM/data/ExtData/CASTNET/"
CNEMC_DIR = "/home/eleroy/proj-dirs/OZCLIM/data/ExtData/CNEMC/"
AIRBASE_DIR = '/home/eleroy/proj-dirs/OZCLIM/data/ExtData/EEA/'
GPWv4_DIR = "/home/eleroy/proj-dirs/OZCLIM/data/ExtData/GPWv4/"

# import population data
population_data = xr.open_rasterio(f'{GPWv4_DIR}/gpw_v4_population_density_rev11_2015_1_deg.tif')
    
# import example c48 grid
dir = f'{MODEL_OUTPUT_DIR}/tools/cs48_example.nc4'
cs48_example = xr.open_dataset(dir)
dst_grid = gcgridobj.cstools.extract_grid(cs48_example)

# import example c90 grid
dir = f'{MODEL_OUTPUT_DIR}/tools/cs90_example.nc4'
cs90_example = xr.open_dataset(dir)
dst_grid_c90 = gcgridobj.cstools.extract_grid(cs90_example)

# import GCHP variables to skip
skip_vars = gcon.skip_these_vars


def set_matplotlib_font(font_family: str):
    """Set the matplotlib font family.
    Args:
        font_family (str): the font family
    """
    supported_fonts: List[str] = {'Andale Mono', 'Arial', 'Arial Black',
                               'Comic Sans MS', 'Courier New', 'Georgia',
                               'Impact', 'Times New Roman', 'Trebuchet MS',
                               'Verdana', 'Webdings', 'Amiri', 'Lato'}

    assert font_family in supported_fonts, f'Font {font_family} not supported.'
    plt.rcParams['font.family'] = font_family
    plt.rcParams["mathtext.fontset"] = 'stixsans'


def initialize_directory(directory_path: str):
    """Make a directory if it doesn't already exist.

    Args:
        directory_path (str): path to directory to be created

    """

    if os.path.exists(directory_path):
        print("Directory " + directory_path + " already exists!")

    else:
        os.makedirs(directory_path)
        print("Initialized directory " + directory_path)

        
def get_file_list(directory_path: str, compiled_regex: Pattern):
    """Return a list of file paths in a directory matching the regex pattern.

    Args:
        directory_path (str): path to directory where to look for files
        compiled_regex (pattern): compiled regex pattern to match

    Returns:
        file_list (list[str]): list of file paths

    """

    file_list = []
    for file_name in os.listdir(directory_path):
        if compiled_regex.match(file_name):
            file_list.append(os.path.join(directory_path, file_name))

    # Important!!! Sort to concatenate chronologically
    file_list.sort()

    return file_list


def open_multifile_ds(file_list: List[str], compat_check=False):
    """Open multiple files (or a single file) as a single dataset.

    WARNING: compatibility checks are overrided for speed-up!

    Args:
        file_list (list(str): list of file paths

    Returns:
        ds (xr.DataSet): a single concatenated xr.Dataset

    """

    v = xr.__version__.split(".")

    
    if int(v[0]) == 0 and int(v[1]) >= 15:
        if compat_check is False:
            ds = xr.open_mfdataset(file_list,
                                drop_variables=skip_vars,
                                combine='nested',
                                concat_dim='time',
                                engine='netcdf4',
                                chunks='auto',
                                parallel=True,
                                data_vars='minimal',
                                coords='minimal',
                                compat='override',
                                autoclose=True)
        else:
            ds = xr.open_mfdataset(file_list,
                                drop_variables=skip_vars,
                                #combine='nested',
                                #concat_dim='time',
                                engine='netcdf4',
                                #chunks='auto',
                                parallel=True,
                                #data_vars='minimal',
                                #coords='minimal',
                                #compat='override',
                                autoclose=True)            
    else:
        ds = xr.open_mfdataset(file_list, drop_variables=skip_vars, autoclose=True)

    return ds


def get_ds(my_simulation: str, variable_type: str):
    """ Function returns dataset for a given simulation for one of 
        three variable types ("Emissions", "SpeciesConc", or "MDA8_O3").

    Args:
        my_simulation(str): example "w10_ref"

        variable_type(str): "Emissions", "SpeciesConc", or "MDA8_O3"
    """

    directory = MODEL_OUTPUT_DIR

    if variable_type=="MDA8_O3":
        regex_pattern = re.compile(fr"GCHP.{my_simulation}.{variable_type}")
    elif variable_type[:4]=="Spec":
        regex_pattern = re.compile(fr"GCHP.{my_simulation}.SpeciesConc")
    elif variable_type[:4]=="Emis":
        regex_pattern = re.compile(fr"GCHP.{my_simulation}.Emissions")
 
    file_path = get_file_list(directory, regex_pattern)
    print(file_path)
    ds = open_multifile_ds(file_path, compat_check=True)
    
    # Chunk these large multi-year datasets into 1-year chunks
    ds = ds.chunk({'time': (366, 365, 365, 365, 366, 365, 365, 365, 366, 365, 365, 365, 366, 365, 365, 365)})

    return ds



def get_cams_ds(file_path: str):
    """
    Open a single file while dropping GCHP variables that should not be read.

    Args:
        file_path (str): path to file to be opened

    """
    ds = xr.open_dataset(file_path)
    if 'latitude' in ds.dims and 'longitude' in ds.dims:
        ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'})
    # Check if latitudes are monotonically increasing:
    if np.all(np.diff(ds.lat) < 0):
        ds = ds.reindex(lat=list(reversed(ds['lat'])))
    
    if any(x > 180 for x in ds.lon):
        ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180
        ds = ds.sortby(ds.lon)

    return ds


def get_ensemble_ds(sim, variable_type):
    """For a given variable type ("Emissions", "SpeciesConc", "MDA8_O3"), 
    return a concatenated dataset with all 5 realizations/ensemble members along dimensions 'sim'
    """
    w10 = get_ds(f"w10_{sim}_c48", variable_type)
    w13 = get_ds(f"w13_{sim}_c48", variable_type)
    w14 = get_ds(f"w14_{sim}_c48", variable_type)
    w26 = get_ds(f"w26_{sim}_c48", variable_type)
    w28 = get_ds(f"w28_{sim}_c48", variable_type)

    ds = xr.concat([w10, w13, w14, w26, w28], dim="sim")
    
    return ds 


def _read_and_process_castnet_data(file_path, qa_options, month):
    df = pd.read_csv(file_path)
    valid_df = df[df['QA_CODE'].isin(qa_options)]
    valid_df['DATE_TIME'] = pd.to_datetime(valid_df['DATE_TIME'], infer_datetime_format=True)
    return valid_df[valid_df['DATE_TIME'].dt.month == month]


def _read_and_process_cnemc_data(directory_path, pattern):
    matching_files = glob.glob(f'{directory_path}/{pattern}')
    combined_df = pd.concat([pd.read_csv(file) for file in matching_files], ignore_index=True)
    o3_df = combined_df[combined_df['type'] == 'O3']
    melted_df = pd.melt(o3_df, id_vars=['date', 'hour', 'type'], var_name='SITE_ID', value_name='OZONE')
    melted_df['OZONE'] = melted_df['OZONE'] * 0.467 # Convert ug/m3 to ppbv according to CNEMC reference (273K and 1013 hPa)
    # Add a 'DATE_TIME' column (all in Beijing Standard Time)
    melted_df['DATE_TIME'] = pd.to_datetime(melted_df['date'].astype(str) + melted_df['hour'].astype(str).str.zfill(2), format='%Y%m%d%H')
    return melted_df[~melted_df['OZONE'].isna()]


def _read_and_process_eea_data(file_path, qa_options, month):
    df = pd.read_csv(file_path, on_bad_lines='skip')
    df = df[(df['AveragingTime'] == 'hour') & (df['UnitOfMeasurement'] == 'µg/m3')]
    df['OZONE'] = df['Concentration'] * 0.501 # Convert ug/m3 to ppbv according to EU Directive (293K and 1013 hPa)
    valid_df = df[df['Validity'].isin(qa_options) & df['Verification'].isin([1])]
    valid_df = valid_df.rename(columns={"AirQualityStationEoICode": "SITE_ID"})

    # Convert UTC + Offset to Local Time
    valid_df['DatetimeBegin'] = pd.to_datetime(valid_df['DatetimeBegin'])
    valid_df['DATE_TIME'] = valid_df['DatetimeBegin'].dt.tz_localize(None) + pd.Timedelta(hours=1) #note that DatetimeBegin includes tz of +0100

    return valid_df[valid_df['DATE_TIME'].dt.month == month]


def _get_ozone_observations(region, month):
    if region == "ENA":
        file_path =  f'{CASTNET_DIR}/ozone_2014.csv'
        df_month_valid = _read_and_process_castnet_data(file_path, [3], month)

    elif region == "EAS":
        directory_path = f'{CNEMC_DIR}'
        pattern = f'china_sites_2014{str(month).zfill(2)}*.csv' if month in [6, 8, 12] else None
        if not pattern:
            raise Exception("Sorry, only December (12) or June (6) or August (8) accepted for China")
        df_month_valid = _read_and_process_cnemc_data(directory_path, pattern)

    elif region == "WCE":
        #file_path = f'{AIRBASE_DIR}/allcountries_O3_2014.csv'
        file_path = f'{AIRBASE_DIR}/allcountries_O3_2014_JuneAugust.csv'
        if month not in [6,8]:
            raise Exception("Sorry, only June (6) or August (8) accepted for Europe")
        df_month_valid = _read_and_process_eea_data(file_path, [1, 2, 3], month)

    else:
        raise Exception("Sorry, only ENA, EAS, or WCE regions accepted")
    df_month_valid['DATE'] = pd.to_datetime(df_month_valid['DATE_TIME'], infer_datetime_format=True).dt.date
    return df_month_valid


def _get_all_site_locations(region):
    # Define file paths and column names to rename for each region
    region_files = {
        "ENA": (f"{CASTNET_DIR}/activesuspendedcastnetsites.xlsx", 'Site ID and Webpage Link'),
        "EAS": (f"{CNEMC_DIR}/SiteList.csv", 'Site'),
        "WCE": (f"{AIRBASE_DIR}/DataExtract.csv", 'Air Quality Station EoI Code'),
    }
    
    if region not in region_files:
        raise Exception("Sorry, only ENA, EAS, or WCE regions accepted")
    
    file_path, column_name = region_files[region]
    
    # Load data and rename the specified column to 'SITE_ID'
    if region == "ENA":
        site_locations_df = pd.read_excel(file_path)
    else:
        site_locations_df = pd.read_csv(file_path)
    
    site_locations_df.rename(columns={column_name: 'SITE_ID'}, inplace=True)
    
    if region == "WCE":
        site_locations_df = site_locations_df[site_locations_df['Air Pollutant']=="O3"]
    
    if region == "EAS":
        # Get population data for each location in site_locations_df
        population_list = []
        for _, row in site_locations_df.iterrows():
            site_id = row["SITE_ID"]
            lon = row["Longitude"]
            lat = row["Latitude"]
            try:
                pop_value = population_data.sel(x=lon, y=lat, method="nearest").values.item()
            except KeyError:
                pop_value = np.nan  # Handle out-of-bounds cases
        
            population_list.append({"SITE_ID": site_id, "Population": pop_value})

        site_populations = pd.DataFrame(population_list)
        rural_sites = site_populations[site_populations['Population']<500].SITE_ID

        # Only select non-urban site locations (pop. less than 2500 people per square km; ref: Wang et al., 2022 ACP)
        site_locations_df = site_locations_df[site_locations_df.SITE_ID.isin(rural_sites)]

    return site_locations_df 


def _filter_sites_by_observation_availability(region, month, criteria=90):
    
    df = _get_ozone_observations(region, month)
    
    # Define the number of days in each month of interest
    days_in_month_dict = {
    1: 31,  # January
    2: 28,  # February (29 in a leap year)
    3: 31,  # March
    4: 30,  # April
    5: 31,  # May
    6: 30,  # June
    7: 31,  # July
    8: 31,  # August
    9: 30,  # September
    10: 31, # October
    11: 30, # November
    12: 31  # December
}
    
    # Calculate the total possible observations for the given month (days * 24 hours)
    total_possible_observations = days_in_month_dict[month] * 24
    
    # Group by SITE_ID and calculate the percentage of available observations
    df_clean = df[~df.OZONE.isin([np.nan])]
    percentage_obs_per_site = df_clean.groupby('SITE_ID').size() / total_possible_observations * 100
    
    # Identify sites with at least the specified criteria percentage of availability
    eligible_sites = percentage_obs_per_site[percentage_obs_per_site >= criteria].index
    
    # Filter and return the DataFrame for these eligible sites

    available_sites = df[df['SITE_ID'].isin(eligible_sites)]

    if region=="WCE":

        # we do not have MetaData for these sites
        ineligible_sites = [
            "RO0094A",
            "RO0196A",
            "RO0095A",
            "RO0128A",
            "RO0136A",
            "RO0085A"]
        available_sites = available_sites[~available_sites['SITE_ID'].isin(ineligible_sites)]
    return available_sites


def _get_mda8o3_daily_data(region, month, criteria=90):
    """ For the given region and month, calculate the MDA8 O3 for each day in the month
    for sites that meet the availability criteria.

    Args:
        region(str): three-character IPCC AR6 region abreviation (ENA, WCE, or EAS)
        month(int): i.e., 6=June, 7=July, 12=December
        criteria(int): minimum percentage of available observations (i.e., 90)
    
    """

    df = _filter_sites_by_observation_availability(region, month, criteria)

    # Convert OZONE concentrations to numeric
    df['OZONE'] = pd.to_numeric(df['OZONE'], errors='coerce')
    
    # Calculate 8-hour rolling mean and shift the result to center the window
    df['O3_mda8'] = df['OZONE'].rolling(window=8, min_periods=2, center=True).mean()

    # Calculate the daily max of the 8-hour average ozone for each site and date
    mda8o3_daily_max = df.groupby(['SITE_ID', 'DATE'])['O3_mda8'].max().reset_index()

    return mda8o3_daily_max


def _get_mda8o3_daily_data_afternoon(region, month, criteria=90, min_hour=12, max_hour=17):
    # min_hour=12, max_hour=17)
    # min_hour=7, max_hour=23)

    df = _filter_sites_by_observation_availability(region, month, criteria)

    # Convert OZONE concentrations to numeric
    df['OZONE'] = pd.to_numeric(df['OZONE'], errors='coerce')
    
    # Calculate 8-hour rolling mean and shift the result to center the window

    # CASTNET datetime is Local Standard Time
    # EEA datetime is UTC + Time Zone
    # CNEMC is Local Time

    # Set DATE_TIME as index
    df.set_index('DATE_TIME', inplace=True)

    # Filter between min hour and max hour
    df = df.between_time(min_hour, max_hour)

    # Calculate 8-hour rolling average for OZONE
    df['O3_mda8'] = df['OZONE'].rolling(window=8, min_periods=2, center=True).mean()

    # Reset index to group by date
    df.reset_index(inplace=True)

    # Add a date column
    df['DATE'] = df['DATE_TIME'].dt.date

    # Group by date and find the maximum 8-hour average
    O3_MDA8_daily_max = df.groupby(['SITE_ID', 'DATE'])['O3_mda8'].max().reset_index()

    return O3_MDA8_daily_max


def _get_complete_valid_site_info(region, month, criteria=90):

    df_valid = _filter_sites_by_observation_availability(region, month, criteria)
    site_locations_df = _get_all_site_locations(region)
    
    # Merge updated site locations with df_dec_selected
    if region=="WCE":
        
        merged_df = pd.merge(df_valid, site_locations_df,
            how='left', left_on=['SamplingPoint', 'SITE_ID'], right_on=['Sampling Point Id', 'SITE_ID'])

        merged_df = merged_df[merged_df['Air Quality Station Type']=="background"]
        merged_df = merged_df[merged_df['Air Quality Station Area']!="urban"]
        merged_df = merged_df[merged_df["Duration Unit"]=="hour"]
        merged_df = merged_df[merged_df["Cadence Unit"]=="hour"]

        if month == 6:
            merged_df = merged_df[merged_df['DATE_TIME']=="2014-06-06 07:00:00"]
        if month == 8:
            merged_df = merged_df[merged_df['DATE_TIME']=="2014-08-06 07:00:00"]
            
    else:                      
        merged_df = pd.merge(df_valid[['SITE_ID']].drop_duplicates(), site_locations_df, on='SITE_ID', how='left')
    # note if you get CRSError: Invalid projection: +proj=gnom +lat_0=nan +lon_0=nan +type=crs: (Internal Proj Error: proj_create: invalid value for lat_0)
    # this means lat/lon is NaN for some sites
    if region=="ENA" and month!=12:
        # locate the row to update
        row_index = merged_df.loc[merged_df['SITE_ID'] == 'HOW191'].index[0]

        # update the row with new values
        merged_df.loc[row_index, 'Latitude'] = 45.203963
        merged_df.loc[row_index, 'Longitude'] = -68.740041

    if region=="EAS":
        # Drop sites with nan Longitude or Latitude (cannot find)
        merged_df = merged_df.dropna(subset=['Longitude', 'Latitude'])

    
    def point_in_polygon_shapely(lon, lat, polygon_coords):
        """Check if given lat/lon coordinates lie within the polygon defined by polygon_coords."""
        polygon = Polygon(polygon_coords)
        point = Point(lon, lat)
        return polygon.contains(point)

    # Define functions for region and grid index calculations, assuming required libraries and objects are defined
    def calculate_region(lat, lon):
        """Given lat/lon coordinates, return the IPCC AR6 region those coordinates lie in"""
        lat_array = np.atleast_1d(lat)
        lon_array = np.atleast_1d(lon)
        
        #new_EAS = np.array([[117,45], [109,37], [109,30], [134,30], [134,30], [150,45]])
        new_EAS = np.array([[117,45], [109,37], [109,30], [126,30],[142,45]])
        inside = point_in_polygon_shapely(lon, lat, new_EAS)
        if inside is True:
            return 58
        else:
            return regionmask.defined_regions.ar6.land.mask(lon_array, lat_array).values.item()

    def get_c48_index(lat, lon, index_type, grid=dst_grid):
        """Given lat/lon coordinates, return c48 coordinates of a given type ('nf', 'Ydim', or 'Xdim')"""
        [nf, Ydim, Xdim] = gcgridobj.cstools.find_index(lat, lon, grid, jitter_size=0.0)
        if index_type == "nf":
            return nf.item()
        elif index_type == "Ydim":
            return Ydim.item()
        elif index_type == "Xdim":
            return Xdim.item()

    # Apply calculations for region and grid indices
    for index_type in ["nf", "Ydim", "Xdim"]:
        merged_df[f'{index_type}_idx'] = merged_df.apply(lambda x: get_c48_index(x['Latitude'], x['Longitude'], index_type=index_type), axis=1)

    merged_df['Region'] = merged_df.apply(lambda x: calculate_region(x['Latitude'], x['Longitude']), axis=1)

    # Create final DataFrame with all site information
    site_info_data = {
        'SITE_ID': merged_df['SITE_ID'].tolist(),
        'Latitude': merged_df['Latitude'].tolist(),
        'Longitude': merged_df['Longitude'].tolist(),
        'IPCC AR6 region': merged_df['Region'].tolist(),
        'nf': merged_df['nf_idx'].tolist(),
        'Ydim': merged_df['Ydim_idx'].tolist(),
        'Xdim': merged_df['Xdim_idx'].tolist()
    }
    site_info = pd.DataFrame(site_info_data)

    return site_info


def get_observed_daily_mda8o3_ar6(region, month, criteria=90):

    ar6_region = {
        "ENA": 5,
        "EAS": 58,
        "WCE": 17,
    }

    O3_mda8_data = _get_mda8o3_daily_data(region, month, criteria)
    site_info = _get_complete_valid_site_info(region, month, criteria)
    mda8o3_ar6 = pd.merge(O3_mda8_data, site_info,
                    how='left', left_on=['SITE_ID'], right_on=['SITE_ID'])
                    
    mda8o3_ar6 = mda8o3_ar6[mda8o3_ar6['IPCC AR6 region']==ar6_region[region]]

    return mda8o3_ar6


def get_observed_daily_mda8o3_ar6_afternoon(region, month, criteria=90, min_hour='12:00', max_hour='17:00'):
    
    ar6_region = {
        "ENA": 5,
        "EAS": 35,
        "WCE": 17,
    }

    O3_mda8_data = _get_mda8o3_daily_data_afternoon(region, month, criteria, min_hour=min_hour, max_hour=max_hour)
    site_info = _get_complete_valid_site_info(region, month, criteria)
    mda8o3_ar6 = pd.merge(O3_mda8_data, site_info,
                    how='left', left_on=['SITE_ID'], right_on=['SITE_ID'])
                    
    mda8o3_ar6 = mda8o3_ar6[mda8o3_ar6['IPCC AR6 region']==ar6_region[region]]

    return mda8o3_ar6


def get_observation_mask(sitemean_mda8o3_ar6):
    """From a list of observations with c48 coordinates, create an xarray 
    mask where these observations exist.

    Args:
        sitemean_mda8o3_ar6 (pd.DataFrame): pandas dataframe with site information (incl. cube-sphere coordinates)

    Returns:
        observation_mask (xr.Dataset): mask of observations

    """

    cs48_example_clean = cs48_example['SpeciesConc_O3'].isel(lev=0,time=0)

    # Create an empty xarray dataset
    nf = np.asarray(cs48_example_clean.nf)
    Ydim = np.asarray(cs48_example_clean.Ydim)
    Xdim = np.asarray(cs48_example_clean.Xdim)

    observation_mask = xr.Dataset(
        coords={"nf": nf, "Ydim": Ydim, "Xdim": Xdim}
    )

    # Initialize variables for each type of data
    observation_mask["O3_mda8"] = (("nf", "Ydim", "Xdim"), np.full((len(nf), len(Ydim), len(Xdim)), np.nan))

    # Iterate over each row in the sitemean dataframe
    for index, row in sitemean_mda8o3_ar6.iterrows():
        observation_mask["O3_mda8"][int(row['nf']), int(row['Ydim']), int(row['Xdim'])] = row["O3_mda8"]

    return observation_mask


def get_masked_model_mda8o3(ds_ref, observation_mask, month):
    """Select GCHP model values where there are observations        
    """
    da_ref = ds_ref['SpeciesConc_O3'].sel(time=ds_ref.time.dt.month.isin(month))
    if "lev" in da_ref.dims:
        da_ref = da_ref.isel(lev=0, drop=True)
    da_ref = da_ref.where(observation_mask['O3_mda8']>0) 
    return da_ref


def get_model_mda8o3(ds_ref, month):
    da_ref = ds_ref['SpeciesConc_O3'].sel(time=ds_ref.time.dt.month.isin(month))
    if "lev" in da_ref.dims:
        da_ref = da_ref.isel(lev=0, drop=True)
    return da_ref


def crop_regionmask_ar6_c48(c48_da, region_num):
    if region_num==58:
        mask_c48 = xr.open_dataarray(f"{MODEL_OUTPUT_DIR}/tools/regionmask.defined_regions.ar6.EAS.modifiedv2.mask_3D.c48.nc")
        c48_da_masked = c48_da.where(mask_c48)
    else:
        mask_c48 = xr.open_dataarray(f"{MODEL_OUTPUT_DIR}/tools/regionmask.defined_regions.ar6.all.mask_3D.c48.nc")
        c48_da_masked = c48_da.where(mask_c48.sel(region=region_num))
    return c48_da_masked


def crop_regionmask_ar6_c90(c90_da, region_num):
    if region_num==58:
        mask_c90 = xr.open_dataarray(f"{MODEL_OUTPUT_DIR}/tools/regionmask.defined_regions.ar6.all.mask_3D.c90.nc")
        c90_da_masked = c90_da.where(mask_c90)
    else:
        mask_c90 = xr.open_dataarray(f"{MODEL_OUTPUT_DIR}/tools/regionmask.defined_regions.ar6.all.mask_3D.c90.nc")
        c90_da_masked = c90_da.where(mask_c90.sel(region=region_num))
    return c90_da_masked


def mask_ocean_c48(c48_da):
    mask_c48 = xr.open_dataarray(f"{MODEL_OUTPUT_DIR}/tools/regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask_3D.c48.nc")
    c48_da_masked = c48_da.where((mask_c48))
    return c48_da_masked


def mask_ocean_c90(c90_da):
    mask_c90 = xr.open_dataarray(f"{MODEL_OUTPUT_DIR}/tools/regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask_3D.c90.nc")
    c90_da_masked = c90_da.where((mask_c90))
    return c90_da_masked


def add_time_local(ds0, grid_obj=dst_grid):
    """Take as input a c48 dataset and add local time as a data array.
    Basic steps:
    1. Convert longitudes from grid_obj from 0-360 to -180-180.
    2. Calculate time offsets from UTC in hours as a function of longitude.
    3. Apply offsets to UTC time.
    """
    ds0_original = ds0.copy()
    # Function to convert longitudes from 0-360 to -180 to 180
    def convert_longitudes(longitude):
        return ((longitude + 180) % 360) - 180

    # Convert the longitudes from 0-360 to -180 to 180
    longitudes = grid_obj['lon']
    longitudes_converted = convert_longitudes(longitudes)

    # Calculate time offsets in hours
    offset_hours = longitudes_converted / 15.0

    # Convert the time offsets
    time_offsets = np.array([pd.Timedelta(hours=offset) for offset in offset_hours.values.flatten()])
    time_offsets = time_offsets.reshape(offset_hours.shape)

    # Convert xarray time to pandas datetime index
    time_utc = ds0.time.to_index()

    # Create a DataArray of time_utc repeated for each grid cell
    time_utc_da = xr.DataArray(time_utc, dims=['time'], coords={'time': ds0.time})

    # Expand time_utc_da to match the dimensions of ds0
    #time_utc_da_expanded = time_utc_da.expand_dims({'nf': ds0.dims['nf'], 'Ydim': ds0.dims['Ydim'], 'Xdim': ds0.dims['Xdim']}).transpose('time', 'nf', 'Ydim', 'Xdim')

    # Apply the time offsets to the UTC time using broadcasting
    #time_local = time_utc_da_expanded.copy()
    #for nf in range(ds0.dims['nf']):
    #    for y in range(ds0.dims['Ydim']):
    #        for x in range(ds0.dims['Xdim']):
    #            time_local[:, nf, y, x] += time_offsets[nf, y, x]

    ds0_original['time_local'] = time_utc_da

    return ds0_original

def get_original_values_diff(variable, sim, sim_SNOx, conversion_factor=1, resolution='c48'):
    ds = xr.open_dataset(fr"{MODEL_OUTPUT_DIR}/GCHP.{sim}.MDA8_O3.april-august.nc4")
    ds_SNOx = xr.open_dataset(fr"{MODEL_OUTPUT_DIR}/GCHP.{sim_SNOx}.MDA8_O3.april-august.nc4")

    ds_diff = ds_SNOx-ds

    da = ds_diff[variable]*conversion_factor
    
    if "lev" in da.dims:
        da = da.isel(lev=0)
    da_monthly = da.resample(time='1M').mean(dim='time')
    da_monthly_noland = mask_ocean_c48(da_monthly)
    return da_monthly_noland


def get_sensitivity_diff_values(variable, simulation, simulation_SNOx, year, conversion_factor=1, resolution='c48'):
    ds = xr.open_dataset(fr"{MODEL_OUTPUT_DIR}/GCHP.{simulation}.MDA8_O3.april-august.{year}.nc4")
    ds_SNOx = xr.open_dataset(fr"{MODEL_OUTPUT_DIR}/GCHP.{simulation_SNOx}.MDA8_O3.april-august.{year}.nc4")
    
    ds_diff = ds_SNOx-ds

    da = ds_diff[variable]*conversion_factor
    if "lev" in da.dims:
        da = da.isel(lev=0)
    da_monthly = da.resample(time='1M').mean(dim='time')
    if resolution=='c90':
        da_monthly_noland = mask_ocean_c90(da_monthly)
    elif resolution=='c48':
        da_monthly_noland = mask_ocean_c48(da_monthly)
    return da_monthly_noland


def get_original_values_ratio(variable, sim, sim_SNOx, resolution='c48'):
    ds = xr.open_dataset(fr"{MODEL_OUTPUT_DIR}/GCHP.{sim}.MDA8_O3.april-august.nc4")
    ds_SNOx = xr.open_dataset(fr"{MODEL_OUTPUT_DIR}/GCHP.{sim_SNOx}.MDA8_O3.april-august.nc4")

    ds_ratio = ds_SNOx/ds

    da = ds_ratio[variable]
    
    if "lev" in da.dims:
        da = da.isel(lev=0)

    da_noland = mask_ocean_c48(da)
    return da_noland


def get_original_values(variable, sim, conversion_factor=1, resolution='c48'):
    
    ds = xr.open_dataset(fr"{MODEL_OUTPUT_DIR}/GCHP.{sim}.Emissions.april-august.nc4")
    da = ds[variable]*conversion_factor
    
    if "lev" in da.dims:
        da = da.isel(lev=0)
    da_monthly = da.resample(time='1M').mean(dim='time')
    da_monthly_noland = mask_ocean_c48(da_monthly)
    return da_monthly_noland


def get_sensitivity_values(variable, simulation, year, conversion_factor=1, resolution='c48'):
    
    ds = xr.open_dataset(fr"{MODEL_OUTPUT_DIR}/GCHP.{simulation}.{variable}.april-august.{year}.nc4")
    da = ds[variable]*conversion_factor
    
    if "lev" in da.dims:
        da = da.isel(lev=0)
    da_monthly = da.resample(time='1M').mean(dim='time')
    if resolution=='c90':
        da_monthly_noland = mask_ocean_c90(da_monthly)
    elif resolution=='c48':
        da_monthly_noland = mask_ocean_c48(da_monthly)
    return da_monthly_noland


def quantile_error(data):
    central = np.mean(data)  # Central value (mean)
    lower = np.quantile(data, 0.025)
    upper = np.quantile(data, 0.975)
    return [central - lower, upper - central]  # Negative and positive errors


def calculate_regional_mean_std(da, months, region, resolution='c48'):
    """
    Calculate the mean and standard deviation for a specific region and month.
    """
    if resolution == 'c48':
        cropped_da = crop_regionmask_ar6_c48(da, region_num=ar6_region[region])
    elif resolution == 'c90':
        cropped_da = crop_regionmask_ar6_c90(da, region_num=ar6_region[region])
    elif resolution == 'latlon':
        raise NotImplementedError

    subset = cropped_da.sel(time=cropped_da.time.dt.month.isin(months))
    seasonal_mean = subset.resample(time='1Y').mean()

    if 'nf' in seasonal_mean.dims:
        regional_mean = seasonal_mean.mean(dim=['nf', 'Ydim', 'Xdim'])
    if 'lat' in seasonal_mean.dims:
        regional_mean = seasonal_mean.mean(dim=['lat', 'lon'])
    mean = regional_mean.mean().values
    std = quantile_error(regional_mean.values)
    return mean, std


ar6_region = {
        "ENA": 5,
        "EAS": 58,
        "WCE": 17,
    }



def reshape_data(data):
    """
    Reshapes a list of pairs into a 2D array with separate lists for each index.

    Parameters:
    data (list of list): A list of pairs, where each pair is [x, y].

    Returns:
    list of list: A 2D array where the first row contains all first elements of the pairs
                  and the second row contains all second elements.
    """
    # Transpose the data using zip
    reshaped = [list(i) for i in zip(*data)]
    return reshaped