"""
General utility routines. Contains many useful functions for
helping to manage xarray and numpy objects used in GCPy.
"""

import os
import warnings
from textwrap import wrap
from yaml import safe_load as yaml_safe_load
import numpy as np
import xarray as xr
from PyPDF2 import PdfFileWriter, PdfFileReader

# ======================================================================
# %%%%% METHODS %%%%%
# ======================================================================

def convert_lon(
        data,
        dim='lon',
        format='atlantic',
        neg_dateline=True
):
    """
    Convert longitudes from -180..180 to 0..360, or vice-versa.

    Args:
        data: DataArray or Dataset
             The container holding the data to be converted; the dimension
             indicated by 'dim' must be associated with this container

    Keyword Args (optional):
        dim: str
             Name of dimension holding the longitude coordinates
             Default value: 'lon'
        format: str
             Control whether or not to shift from -180..180 to 0..360 (
             ('pacific') or from 0..360 to -180..180 ('atlantic')
             Default value: 'atlantic'
        neg_dateline: logical
             If True, then the international dateline is set to -180
             instead of 180.
             Default value: True

    Returns:
        data, with dimension 'dim' altered according to conversion rule
    """

    data_copy = data.copy()

    lon = data_copy[dim].values
    new_lon = np.empty_like(lon)

    # Tweak offset for rolling the longitudes later
    offset = 0 if neg_dateline else 1

    if format not in ['atlantic', 'pacific']:
        msg = f"Cannot convert longitudes for format '{format}'; "
        msg += "please choose one of 'atlantic' or 'pacific'"
        raise ValueError(msg)

    # Create a mask to decide how to mutate the longitude values
    if format == 'atlantic':
        mask = lon >= 180 if neg_dateline else lon > 180

        new_lon[mask] = -(360. - lon[mask])
        new_lon[~mask] = lon[~mask]

        roll_len = len(data[dim]) // 2 - offset

    elif format == 'pacific':
        mask = lon < 0.

        new_lon[mask] = lon[mask] + 360.
        new_lon[~mask] = lon[~mask]

        roll_len = -len(data[dim]) // 2 - offset

    # Copy mutated longitude values into copied data container
    data_copy[dim].values = new_lon
    data_copy = data_copy.roll(**{dim: roll_len})

    return data_copy


def add_bookmarks_to_pdf(
        pdfname,
        varlist,
        remove_prefix="",
        verbose=False
):
    """
    Adds bookmarks to an existing PDF file.

    Args:
        pdfname: str
            Name of an existing PDF file of species or emission plots
            to which bookmarks will be attached.
        varlist: list
            List of variables, which will be used to create the
            PDF bookmark names.

    Keyword Args (optional):
        remove_prefix: str
            Specifies a prefix to remove from each entry in varlist
            when creating bookmarks.  For example, if varlist has
            a variable name "SpeciesConcVV_NO", and you specify
            remove_prefix="SpeciesConcVV_", then the bookmark for
            that variable will be just "NO", etc.
         verbose: bool
            Set this flag to True to print extra informational output.
            Default value: False
    """

    # Setup
    pdfobj = open(pdfname, "rb")
    input_pdf = PdfFileReader(pdfobj, overwriteWarnings=False)
    output_pdf = PdfFileWriter()

    for i, varname in enumerate(varlist):
        bookmarkname = varname.replace(remove_prefix, "")
        if verbose:
            print(f"Adding bookmark for {varname} with name {bookmarkname}")
        output_pdf.addPage(input_pdf.getPage(i))
        output_pdf.addBookmark(bookmarkname, i)
        output_pdf.setPageMode("/UseOutlines")

    # Write to temp file
    pdfname_tmp = pdfname + "_with_bookmarks.pdf"
    outputstream = open(pdfname_tmp, "wb")
    output_pdf.write(outputstream)
    outputstream.close()

    # Rename temp file with the target name
    os.rename(pdfname_tmp, pdfname)
    pdfobj.close()


def add_nested_bookmarks_to_pdf(
        pdfname,
        category,
        catdict,
        warninglist,
        remove_prefix=""
):
    """
    Add nested bookmarks to PDF.

    Args:
        pdfname: str
            Path of PDF to add bookmarks to
        category: str
            Top-level key name in catdict that maps to contents of PDF
        catdict: dictionary
            Dictionary containing key-value pairs where one top-level
            key matches category and has value fully describing pages
            in PDF. The value is a dictionary where keys are level 1
            bookmark names, and values are lists of level 2 bookmark
            names, with one level 2 name per PDF page.  Level 2 names
            must appear in catdict in the same order as in the PDF.
        warninglist: list of strings
            Level 2 bookmark names to skip since not present in PDF.

    Keyword Args (optional):
        remove_prefix: str
            Prefix to be remove from warninglist names before comparing with
            level 2 bookmark names in catdict.
            Default value: empty string (warninglist names match names
            in catdict)
    """

    # ==================================================================
    # Setup
    # ==================================================================
    pdfobj = open(pdfname, "rb")
    input_pdf = PdfFileReader(pdfobj, overwriteWarnings=False)
    output_pdf = PdfFileWriter()
    warninglist = [k.replace(remove_prefix, "") for k in warninglist]

    # ==================================================================
    # Loop over the subcategories in this category; make parent bookmark
    # ==================================================================
    i = -1
    for subcat in catdict[category]:

        # First check that there are actual variables for
        # this subcategory; otherwise skip
        numvars = 0
        if catdict[category][subcat]:
            for varname in catdict[category][subcat]:
                if varname in warninglist:
                    continue
                numvars += 1
        else:
            continue
        if numvars == 0:
            continue

        # There are non-zero variables to plot in this subcategory
        i = i + 1
        output_pdf.addPage(input_pdf.getPage(i))
        parent = output_pdf.addBookmark(subcat, i)
        output_pdf.setPageMode("/UseOutlines")
        first = True

        # Loop over variables in this subcategory; make children bookmarks
        for varname in catdict[category][subcat]:
            if varname in warninglist:
                print(f"Warning: skipping {varname}")
                continue
            if first:
                output_pdf.addBookmark(varname, i, parent)
                first = False
            else:
                i = i + 1
                output_pdf.addPage(input_pdf.getPage(i))
                output_pdf.addBookmark(varname, i, parent)
                output_pdf.setPageMode("/UseOutlines")

    # ==================================================================
    # Write to temp file
    # ==================================================================
    pdfname_tmp = pdfname + "_with_bookmarks.pdf"
    outputstream = open(pdfname_tmp, "wb")
    output_pdf.write(outputstream)
    outputstream.close()

    # Rename temp file with the target name
    os.rename(pdfname_tmp, pdfname)
    pdfobj.close()


def reshape_MAPL_CS(
        darr,
        multi_index_lat=True
):
    """
    Reshapes data if contains dimensions indicate MAPL v1.0.0+ output

    Args:
        darr: xarray DataArray
            Data array variable

    Keyword Args (Optional):
        multi_index_lat : bool
            Determines if the returned "lat" index of the DataArray
            object will be a MultiIndex (true) or a simple list of
            latitude values (False).
            Default value: True

    Returns:
        darr: xarray DataArray
            Data with dimensions renamed and transposed to match old MAPL format
    """
    # Suppress annoying future warnings for now
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Only do the following for DataArray objects
    # (otherwise just fall through and return the original argument as-is)
    if isinstance(darr, xr.DataArray):
        with xr.set_options(keep_attrs=True):
            if "nf" in darr.dims and "Xdim" in darr.dims and "Ydim" in darr.dims:
                darr = darr.stack(lat=("nf", "Ydim"))
                darr = darr.rename({"Xdim": "lon"})
                # NOTE: The darr.stack operation will return the darr.lat
                # dimension as a MultiIndex.  In other words, each
                # element of darr.lat is a tuple (face number, latitude
                # in degrees).  To disable this behavior, set keyword
                # argument multi_index_lat=False.  This will return
                # darr.lat as an array of latitude values, which is
                # needed for backwards compatibility.
                #  -- Bob Yantosca (07 Jul 2023)
                if not multi_index_lat:
                    darr = darr.assign_coords(
                        {"lat": [list(tpl)[1] for tpl in darr.lat.values]}
                    )
            if "lev" in darr.dims and "time" in darr.dims:
                darr = darr.transpose("time", "lev", "lat", "lon")
            elif "lev" in darr.dims:
                darr = darr.transpose("lev", "lat", "lon")
            elif "time" in darr.dims:
                darr = darr.transpose("time", "lat", "lon")
            else:
                darr = darr.transpose("lat", "lon")
    return darr


def slice_by_lev_and_time(
        dset,
        varname,
        itime,
        ilev,
        flip
):
    """
    Given a Dataset, returns a DataArray sliced by desired time and level.

    Args:
        dset: xarray Dataset
            Dataset containing GEOS-Chem data.
        varname: str
            Variable name for data variable to be sliced
        itime: int
            Index of time by which to slice
        ilev: int
            Index of level by which to slice
        flip: bool
            Whether to flip ilev to be indexed from ground or top of atmosphere

    Returns:
        darr: xarray DataArray
            DataArray of data variable sliced according to ilev and itime
    """
    verify_variable_type(dset, xr.Dataset)

    # used in compare_single_level and compare_zonal_mean to get dataset slices
    if not varname in dset.data_vars.keys():
        msg="Could not find 'varname' in dset!"
        raise ValueError(msg)

    # NOTE: isel no longer seems to work on a Dataset, so
    # first createthe DataArray object, then use isel on it.
    #  -- Bob Yantosca (19 Jan 2023)
    darr = dset[varname]
    vdims = darr.dims
    if ("time" in vdims and darr.time.size > 0) and "lev" in vdims:
        if flip:
            fliplev=len(darr['lev']) - 1 - ilev
            return darr.isel(time=itime, lev=fliplev)
        return darr.isel(time=itime, lev=ilev)
    if ("time" not in vdims or itime == -1) and "lev" in vdims:
        if flip:
            fliplev= len(darr['lev']) - 1 - ilev
            return darr.isel(lev=fliplev)
        return darr.isel(lev=ilev)
    if ("time" in vdims and darr.time.size > 0 and itime != -1) and \
       "lev" not in vdims:
        return darr.isel(time=itime)
    return darr


def rename_and_flip_gchp_rst_vars(
        dset
):
    '''
    Transforms a GCHP restart dataset to match GCC names and level convention

    Args:
        dset: xarray Dataset
            Dataset containing GCHP restart file data, such as variables
            SPC_{species}, BXHEIGHT, DELP_DRY, and TropLev, with level
            convention down (level 0 is top-of-atmosphere).

    Returns:
        dset: xarray Dataset
            Dataset containing GCHP restart file data with names and level
            convention matching GCC restart. Variables include
            SpeciesRst_{species}, Met_BXHEIGHT, Met_DELPDRY, and Met_TropLev,
            with level convention up (level 0 is surface).
    '''
    for var in dset.data_vars.keys():
        if var.startswith('SPC_'):
            spc = var.replace('SPC_', '')
            dset = dset.rename({var: 'SpeciesRst_' + spc})
        elif var == 'DELP_DRY':
            dset = dset.rename({"DELP_DRY": "Met_DELPDRY"})
        elif var == 'BXHEIGHT':
            dset = dset.rename({"BXHEIGHT": "Met_BXHEIGHT"})
        elif var == 'TropLev':
            dset = dset.rename({"TropLev": "Met_TropLev"})
    dset = dset.sortby('lev', ascending=False)
    return dset


def dict_diff(
        dict0,
        dict1
):
    """
    Function to take the difference of two dict objects.
    Assumes that both objects have the same keys.

    Args:
        dict0, dict1: dict
            Dictionaries to be subtracted (dict1 - dict0)

    Returns:
        result: dict
            Key-by-key difference of dict1 - dict0
    """
    result = {}
    for key, _ in dict0.items():
        result[key] = dict1[key] - dict0[key]

    return result


def compare_stats(
        refdata,
        refstr,
        devdata,
        devstr,
        varname
):
    """
    Prints out global statistics (array sizes, mean, min, max, sum)
    from two xarray Dataset objects.

    Args:
        refdata: xarray Dataset
            The first Dataset to be compared.
            (This is often referred to as the "Reference" Dataset.)
        refstr: str
            Label for refdata to be used in the printout
        devdata: xarray Dataset
            The second Dataset to be compared.
            (This is often referred to as the "Development" Dataset.)
        devstr: str
            Label for devdata to be used in the printout
        varname: str
            Variable name for which global statistics will be printed out.
    """

    refvar = refdata[varname]
    devvar = devdata[varname]
    units = refdata[varname].units
    print("Data units:")
    print(f"    {refstr}:  {units}")
    print(f"    {devstr}:  {units}")
    print("Array sizes:")
    print(f"    {refstr}:  {refvar.shape}")
    print(f"    {devstr}:  {devvar.shape}")
    print("Global stats:")
    print("  Mean:")
    print(f"    {refstr}:  {np.round(refvar.values.mean(), 20)}")
    print(f"    {devstr}:  {np.round(devvar.values.mean(), 20)}")
    print("  Min:")
    print(f"    {refstr}:  {np.round(refvar.values.min(), 20)}")
    print(f"    {devstr}:  {np.round(devvar.values.min(), 20)}")
    print("  Max:")
    print(f"    {refstr}:  {np.round(refvar.values.max(), 20)}")
    print(f"    {devstr}:  {np.round(devvar.values.max(), 20)}")
    print("  Sum:")
    print(f"    {refstr}:  {np.round(refvar.values.sum(), 20)}")
    print(f"    {devstr}:  {np.round(devvar.values.sum(), 20)}")


def convert_bpch_names_to_netcdf_names(
        dset,
        verbose=False
):
    """
    Function to convert the non-standard bpch diagnostic names
    to names used in the GEOS-Chem netCDF diagnostic outputs.

    Args:
        dset: xarray Dataset
            The xarray Dataset object whose names are to be replaced.

    Keyword Args (optional):
        verbose: bool
            Set this flag to True to print informational output.
            Default value: False

    Returns:
        dset_new: xarray Dataset
            A new xarray Dataset object all of the bpch-style
            diagnostic names replaced by GEOS-Chem netCDF names.

    Remarks:
        To add more diagnostic names, edit the dictionary contained
        in the bpch_to_nc_names.yml.
    """

    # Names dictionary (key = bpch id, value[0] = netcdf id,
    # value[1] = action to create full name using id)
    # Now read from YAML file (bmy, 4/5/19)
    names = read_config_file(
        os.path.join(
            os.path.dirname(__file__),
            "bpch_to_nc_names.yml"
        ),
        quiet=True
    )

    # define some special variable to overwrite above
    special_vars = {
        "Met_AIRNUMDE": "Met_AIRNUMDEN",
        "Met_UWND": "Met_U",
        "Met_VWND": "Met_V",
        "Met_CLDTOP": "Met_CLDTOPS",
        "Met_GWET": "Met_GWETTOP",
        "Met_PRECON": "Met_PRECCON",
        "Met_PREACC": "Met_PRECTOT",
        "Met_PBL": "Met_PBLH",
    }

    # Tags for the UVFlux* diagnostics
    uvflux_tags = [
        "187nm",
        "191nm",
        "193nm",
        "196nm",
        "202nm",
        "208nm",
        "211nm",
        "214nm",
        "261nm",
        "267nm",
        "277nm",
        "295nm",
        "303nm",
        "310nm",
        "316nm",
        "333nm",
        "380nm",
        "574nm",
    ]

    # Python dictionary for variable name replacement
    old_to_new = {}

    # Loop over all variable names in the data set
    for variable_name in dset.data_vars.keys():

        # Save the original variable name, since this is the name
        # that we actually need to replace in the dataset.
        original_variable_name = variable_name

        # Replace "__" with "_", in variable name (which will get tested
        # against the name sin the YAML file.  This will allow us to
        # replace variable names in files created with BPCH2COARDS.
        if "__" in variable_name:
            variable_name = variable_name.replace("__", "_")

        # Check if name matches anything in dictionary. Give warning if not.
        oldid = ""
        newid = ""
        idaction = ""
        for key in names:
            if key in variable_name:
                if names[key][1] == "skip":
                    # Verbose output
                    if verbose:
                        print(f"WARNING: skipping {key}")
                else:
                    oldid = key
                    newid = names[key][0]
                    idaction = names[key][1]
                break

        # Go to the next line if no definition was found
        if oldid == "" or newid == "" or idaction == "":
            continue

        # If fullname replacement:
        if idaction == "replace":
            newvar = newid

            # Update the dictionary of names with this pair
            # Use the original variable name.
            old_to_new.update({original_variable_name: newvar})

        # For all the rest:
        else:
            linearr = variable_name.split("_")
            varstr = linearr[-1]

            # These categories use append
            if oldid in [
                    "IJ_AVG_S_",
                    "RN_DECAY_",
                    "WETDCV_S_",
                    "WETDLS_S_",
                    "BXHGHT_S_",
                    "DAO_3D_S_",
                    "PL_SUL_",
                    "CV_FLX_S_",
                    "EW_FLX_S_",
                    "NS_FLX_S_",
                    "UP_FLX_S_",
                    "MC_FRC_S_",
            ]:
                newvar = newid + "_" + varstr

            # DAO_FLDS
            # Skip certain fields that will cause conflicts w/ netCDF
            elif oldid in "DAO_FLDS_":
                if oldid in ["DAO_FLDS_PS_PBL", "DAO_FLDS_TROPPRAW"]:

                    # Verbose output
                    if verbose:
                        print(f"Skipping: {oldid}")
                else:
                    newvar = newid + "_" + varstr

            # Special handling for J-values: The bpch variable names all
            # begin with "J" (e.g. JNO, JACET), so we need to strip the first
            # character of the variable name manually (bmy, 4/8/19)
            elif oldid == "JV_MAP_S_":
                newvar = newid + "_" + varstr[1:]

            # IJ_SOA_S_
            elif oldid == "IJ_SOA_S_":
                newvar = newid + varstr

            # DRYD_FLX_, DRYD_VEL_
            elif "DRYD_" in oldid:
                newvar = newid + "_" + varstr[:-2]

            # BIOBSRCE_, BIOFSRCE_, BIOGSRCE_. ANTHSRCE_
            elif oldid in ["BIOBSRCE_", "BIOFSRCE_", "BIOGSRCE_", "ANTHSRCE_"]:
                newvar = "Emis" + varstr + "_" + newid

            # Special handling for UV radiative flux diagnostics:
            # We need to append the bin descriptor to the new name.
            elif "FJX_FLXS" in oldid:
                uvind = int(original_variable_name[-2:]) - 1
                newvar = newid + "_" + uvflux_tags[uvind]

            # If nothing found...
            else:

                # Verbose output
                if verbose:
                    print(f"WARNING: Nothing defined for: {variable_name}")
                continue

            # Overwrite certain variable names
            if newvar in special_vars:
                newvar = special_vars[newvar]

            # Update the dictionary of names with this pair
            old_to_new.update({original_variable_name: newvar})

    # Verbose output
    if verbose:
        print("\nList of bpch names and netCDF names")
        for key in old_to_new:
            print(f"{key : <25} ==> {old_to_new[key] : <40}")

    # Rename the variables in the dataset
    if verbose:
        print("\nRenaming variables in the data...")
    with xr.set_options(keep_attrs=True):
        dset = dset.rename(name_dict=old_to_new)

    # Return the dataset
    return dset


def filter_names(
        names,
        text=""):
    """
    Returns elements in a list that match a given substring.
    Can be used in conjnction with compare_varnames to return a subset
    of variable names pertaining to a given diagnostic type or species.

    Args:
        names: list of str
            Input list of names.
        text: str
            Target text string for restricting the search.

    Returns:
        filtered_names: list of str
            Returns all elements of names that contains the substring
            specified by the "text" argument.  If "text" is omitted,
            then the original contents of names will be returned.
    """
    verify_variable_type(text, str)

    if text != "":
        return [name for name in names if text in name]

    return names


def divide_dataset_by_dataarray(
        dset,
        darr,
        varlist=None
):
    """
    Divides variables in an xarray Dataset object by a single DataArray
    object.  Will also make sure that the Dataset variable attributes
    are preserved.
    This method can be useful for certain types of model diagnostics
    that have to be divided by a counter array.  For example, local
    noontime J-value variables in a Dataset can be divided by the
    fraction of time it was local noon in each grid box, etc.

    Args:
        dset: xarray Dataset
            The Dataset object containing variables to be divided.
        darr: xarray DataArray
            The DataArray object that will be used to divide the
            variables of dset.

    Keyword Args (optional):
        varlist: list of str
            If passed, then only those variables of dset that are listed
            in varlist will be divided by darr.  Otherwise, all variables
            of dset will be divided by darr.
            Default value: None
    Returns:
        dset_new: xarray Dataset
            A new xarray Dataset object with its variables divided by darr.
    """
    # -----------------------------
    # Check arguments
    # -----------------------------
    verify_variable_type(dset, xr.Dataset)
    verify_variable_type(darr, xr.DataArray)
    if varlist is None:
        varlist = dset.data_vars.keys()

    # -----------------------------
    # Do the division
    # -----------------------------

    # Keep all Dataset attributes
    with xr.set_options(keep_attrs=True):

        # Loop over variables
        for var in varlist:

            # Divide each variable of dset by darr
            dset[var] = dset[var] / darr

    return dset


def get_shape_of_data(
        data,
        vertical_dim="lev",
        return_dims=False
):
    """
    Convenience routine to return a the shape (and dimensions, if
    requested) of an xarray Dataset, or xarray DataArray.  Can also
    also take as input a dictionary of sizes (i.e. {'time': 1,
    'lev': 72, ...} from an xarray Dataset or xarray Datarray object.

    Args:
        data: xarray Dataset, xarray DataArray, or dict
            The data for which the size is requested.

    Keyword Args (optional):
        vertical_dim: str
            Specify the vertical dimension that you wish to
            return: lev or ilev.
            Default value: 'lev'
        return_dims: bool
            Set this switch to True if you also wish to return a list of
            dimensions in the same order as the tuple of dimension sizes.
            Default value: False

    Returns:
        shape: tuple of int
            Tuple containing the sizes of each dimension of darr in order:
            (time, lev|ilev, nf, lat|YDim, lon|XDim).
        dims: list of str
            If return_dims is True, then dims will contain a list of
            dimension names in the same order as shape
            (['time', 'lev', 'lat', 'lon'] for GEOS-Chem "Classic",
             or ['time', 'lev', 'nf', 'Ydim', 'Xdim'] for GCHP.
    """

    # Validate the data argument
    if isinstance(data, (xr.Dataset, xr.DataArray)):
        sizelist = data.sizes
    elif isinstance(data, dict):
        sizelist = data
    else:
        msg = (
            'The "dataset" argument must be either an xarray Dataset, '
            + " xarray DataArray, or a dictionary!"
        )
        raise ValueError(msg)

    # Initialize
    dimlist = ["time", vertical_dim, "lat", "nf", "Ydim", "lon", "Xdim"]
    shape = ()
    dims = []

    # Return a tuple with the shape of each dimension (and also a
    # list of each dimension if return_dims is True).
    for dim in dimlist:
        if dim in sizelist:
            shape += (sizelist[dim],)
            dims.append(dim)

    if return_dims:
        return shape, dims
    return shape


def get_area_from_dataset(
        dset
):
    """
    Convenience routine to return the area variable (which is
    usually called "AREA" for GEOS-Chem "Classic" or "Met_AREAM2"
    for GCHP) from an xarray Dataset object.

    Args:
        dset: xarray Dataset
            The input dataset.
    Returns:
        area_m2: xarray DataArray
            The surface area in m2, as found in dset.
    """

    if "Met_AREAM2" in dset.data_vars.keys():
        return dset["Met_AREAM2"]
    if "AREA" in dset.data_vars.keys():
        return dset["AREA"]
    msg = (
        'An area variable ("AREA" or "Met_AREAM2" is missing'
        + " from this dataset!"
    )
    raise ValueError(msg)


def get_variables_from_dataset(
        dset,
        varlist
):
    """
    Convenience routine to return multiple selected DataArray
    variables from an xarray Dataset.  All variables must be
    found in the Dataset, or else an error will be raised.

    Args:
        dset: xarray Dataset
            The input dataset.
        varlist: list of str
            List of DataArray variables to extract from dset.

    Returns:
        dset_subset: xarray Dataset
            A new data set containing only the variables
            that were requested.

    Remarks:
    Use this routine if you absolutely need all of the requested
    variables to be returned.  Otherwise
    """

    dset_subset = xr.Dataset()
    for var in varlist:
        if var in dset.data_vars.keys():
            dset_subset = xr.merge([dset_subset, dset[var]])
        else:
            msg = f"{var} was not found in this dataset!"
            raise ValueError(msg)

    return dset_subset


def create_blank_dataarray(
        name,
        sizes,
        coords,
        attrs,
        fill_value=np.nan,
        fill_type=np.float64,
        vertical_dim="lev"
):
    """
    Given an xarray DataArray darr, returns a DataArray object with
    the same dimensions, coordinates, attributes, and name, but
    with its data set to missing values (default=NaN) everywhere.
    This is useful if you need to plot or compare two DataArray
    variables, and need to represent one as missing or undefined.

    Args:
    name: str
        The name for the DataArray object that will contain NaNs.
    sizes: dict of int
        Dictionary of the dimension names and their sizes (e.g.
        {'time': 1 ', 'lev': 72, ...} that will be used to create
        the DataArray of NaNs.  This can be obtained from an
        xarray Dataset as dset.sizes.
    coordset: dict of lists of float
        Dictionary containing the coordinate variables that will
        be used to create the DataArray of NaNs.  This can be obtained
        from an xarray Dataset with dset.coordset.
    attrs: dict of str
        Dictionary containing the DataArray variable attributes
        (such as "units", "long_name", etc.).  This can be obtained
        from an xarray Dataset with darr.attrs.
    fill_value: np.nan or numeric type
        Value with which the DataArray object will be filled.
        Default value: np.nan
    fill_type: numeric type
        Specifies the numeric type of the DataArray object.
        Default value: np.float64 (aka "double")
    vertical_dim: str
        Specifies the name of the vertical dimension (e.g. "lev", "ilev")
        Default: "lev"

    Returns:
    darr: xarray DataArray
        The output DataArray object, which will be set to the value
        specified by the fill_value argument everywhere.
    """

    # Save dims and coords into local variables
    # NOTE: Cast to type dict so that we can delete keys and values
    new_sizes = dict(sizes)
    new_coords = dict(coords)

    # Only keep one of the vertical dimensions (lev or ilev)
    if vertical_dim == "lev":
        if "ilev" in new_sizes:
            del new_sizes["ilev"]
            del new_coords["ilev"]
    elif vertical_dim == "ilev":
        if "lev" in new_sizes:
            del new_sizes["lev"]
            del new_coords["lev"]
    else:
        msg = 'The "vertical_lev" argument must be either "lev" or "ilev"!'
        raise ValueError(msg)

    # Get the names and sizes of the dimensions
    # after discarding one of "lev" or "ilev"
    [new_shape, new_dims] = get_shape_of_data(new_sizes, return_dims=True)

    # Create an array full of NaNs of the required size
    fill_arr = np.empty(new_shape, dtype=fill_type)
    fill_arr.fill(fill_value)

    # Create a DataArray of NaN's
    return xr.DataArray(
        fill_arr,
        name=name,
        dims=new_dims,
        coords=new_coords,
        attrs=attrs
    )


def check_for_area(
        dset,
        gcc_area_name="AREA",
        gchp_area_name="Met_AREAM2"
):
    """
    Makes sure that a dataset has a surface area variable contained
    within it.
    GEOS-Chem Classic files all contain surface area as variable AREA.
    GCHP files do not and area must be retrieved from the met-field
    collection from variable Met_AREAM2. To simplify comparisons,
    the GCHP area name will be appended to the dataset under the
    GEOS-Chem "Classic" area name if it is present.

    Args:
        dset: xarray Dataset
            The Dataset object that will be checked.

    Keyword Args (optional):
        gcc_area_name: str
            Specifies the name of the GEOS-Chem "Classic" surface
            area varaible
            Default value: "AREA"
        gchp_area_name: str
            Specifies the name of the GCHP surface area variable.
            Default value: "Met_AREAM2"

    Returns:
        dset: xarray Dataset
            The modified Dataset object
    """

    found_gcc = gcc_area_name in dset.data_vars.keys()
    found_gchp = gchp_area_name in dset.data_vars.keys()

    if (not found_gcc) and (not found_gchp):
        msg = f"Could not find {gcc_area_name} or {gchp_area_name} in the dataset!"
        raise ValueError(msg)

    if found_gchp:
        dset[gcc_area_name] = dset[gchp_area_name]

    return dset


def extract_pathnames_from_log(
        filename,
        prefix_filter=""
):
    """
    Returns a list of pathnames from a GEOS-Chem log file.
    This can be used to get a list of files that should be
    downloaded from gcgrid or from Amazon S3.

    Args:
        filename: str
            GEOS-Chem standard log file
        prefix_filter (optional): str
            Restricts the output to file paths starting with
            this prefix (e.g. "/home/ubuntu/ExtData/HEMCO/")
            Default value: ''
    Returns:
        data list: list of str
            List of full pathnames of data files found in
            the log file.
    Author:
        Jiawei Zhuang (jiaweizhuang@g.harvard.edu)
    """

    # Initialization
    prefix_len = len(prefix_filter)
    data_list = set()  # only keep unique files

    # Open file
    with open(filename, "r", encoding='UTF-8') as input_file:

        # Read data from the file line by line.
        # Add file paths to the data_list set.
        line = input_file.readline()
        while line:
            upcaseline = line.upper()
            if (": OPENING" in upcaseline) or (": READING" in upcaseline):
                data_path = line.split()[-1]
                # remove common prefix
                if data_path.startswith(prefix_filter):
                    trimmed_path = data_path[prefix_len:]
                    data_list.add(trimmed_path)

            # Read next line
            line = input_file.readline()

        # Close file and return
        input_file.close()

    data_list = sorted(list(data_list))
    return data_list


def get_nan_mask(
        data
):
    """
    Create a mask with NaN values removed from an input array

    Args:
        data: numpy array
            Input array possibly containing NaNs

    Returns:
        new_data: numpy array
            Original array with NaN values removed
    """

    # remove NaNs
    fill = np.nanmax(data) + 100000
    new_data = np.where(np.isnan(data), fill, data)
    new_data = np.ma.masked_where(data == fill, data)
    return new_data


def all_zero_or_nan(
        dset
):
    """
    Return whether dset is all zeros, or all nans

    Args:
        dset: numpy array
            Input GEOS-Chem data
    Returns:
        all_zero, all_nan: bool, bool
            all_zero is whether dset is all zeros,
            all_nan  is whether dset is all NaNs
    """

    return not np.any(dset), np.isnan(dset).all()


def dataset_mean(
        dset,
        dim="time",
        skipna=True
):
    """
    Convenience wrapper for taking the mean of an xarray Dataset.

    Args:
       dset : xarray Dataset
           Input data

    Keyword Args:
       dim : str
           Dimension over which the mean will be taken.
           Default: "time"
       skipna : bool
           Flag to omit missing values from the mean.
           Default: True

    Returns:
       dset_mean : xarray Dataset or None
           Dataset containing mean values
           Will return None if dset is not defined
    """
    if dset is None:
        return dset

    if not isinstance(dset, xr.Dataset):
        raise ValueError("Argument dset must be None or xarray.Dataset!")

    with xr.set_options(keep_attrs=True):
        return dset.mean(dim=dim, skipna=skipna)


def dataset_reader(
        multi_files,
        verbose=False
):
    """
    Returns a function to read an xarray Dataset.

    Args:
        multi_files : bool
            Denotes whether we will be reading multiple files into
            an xarray Dataset.
            Default value: False

    Returns:
         reader : either xr.open_mfdataset or xr.open_dataset
    """
    if multi_files:
        reader = xr.open_mfdataset
        if verbose:
            print('Reading data via xarray open_mfdataset\n')
    else:
        reader = xr.open_dataset
        if verbose:
            print('Reading data via xarray open_dataset\n')

    return reader


def read_config_file(config_file, quiet=False):
    """
    Reads configuration information from a YAML file.
    """
    # Read the configuration file in YAML format
    try:
        if not quiet:
            print(f"Using configuration file {config_file}")
        config = yaml_safe_load(open(config_file))
    except Exception as err:
        msg = f"Error reading configuration in {config_file}: {err}"
        raise Exception(msg) from err

    return config


def unique_values(
        this_list,
        drop=None,
):
    """
    Given a list, returns a sorted list of unique values.

    Args:
    -----
    this_list : list
        Input list (may contain duplicate values)

    drop: list of str
        List of variable names to exclude

    Returns:
    --------
    unique: list
        List of unique values from this_list
    """
    verify_variable_type(this_list, list)
    verify_variable_type(drop, list)

    unique = list(set(this_list))

    if drop is not None:
        for drop_val in drop:
            if drop_val in unique:
                unique.remove(drop_val)

    unique.sort()

    return unique


def wrap_text(
        text,
        width=80
):
    """
    Wraps text so that it fits within a certain line width.

    Args:
    -----
    text: str or list of str
        Input text to be word-wrapped.
    width: int
        Line width, in characters.
        Default value: 80

    Returns:
    --------
    Original text reformatted so that it fits within lines
    of 'width' characters or less.
    """
    if not isinstance(text, str):
        if isinstance(text, list):
            text = ' '.join(text)  # List -> str conversion
        else:
            raise ValueError("Argument 'text' must be either str or list!")

    text = wrap(text, width=width)
    text = '\n'.join(text)

    return text


def insert_text_into_file(
        filename,
        search_text,
        replace_text,
        width=80
):
    """
    Convenience routine to insert text into a file.  The best way
    to do this is to read the contents of the file, manipulate the
    text, and then overwrite the file.

    Args:
    -----
    filename: str
        The file with text to be replaced.
    search_text: str
        Text string in the file that will be replaced.
    replace_text: str or list of str
        Text that will replace 'search_text'
    width: int
        Will "word-wrap" the text in 'replace_text' to this width
    """
    verify_variable_type(search_text, str)
    verify_variable_type(replace_text, (str, list))

    # Word-wrap the replacement text
    # (does list -> str conversion if necessary)
    replace_text = wrap_text(
        replace_text,
        width=width
    )

    with open(filename, "r", encoding="UTF-8") as input_file:
        filedata = input_file.read()
        input_file.close()

    filedata = filedata.replace(
        search_text,
        replace_text
    )

    with open(filename, "w", encoding="UTF-8") as output_file:
        output_file.write(filedata)
        output_file.close()


def array_equals(
        refdata,
        devdata,
        dtype=np.float64
):
    """
    Tests two arrays for equality.  Useful for checking which
    species have nonzero differences in benchmark output.

    Args:
    -----
    refdata: xarray DataArray or numpy ndarray
        The first array to be checked.
    devdata: xarray DataArray or numpy ndarray
        The second array to be checked.
    dtype : np.float32 or np.float64
        The precision that will be used to make the evaluation.
        Default: np.float64

    Returns:
    --------
    True if both arrays are equal; False if not
    """
    if not isinstance(refdata, np.ndarray):
        if isinstance(refdata, xr.DataArray):
            refdata = refdata.values
        else:
            raise ValueError(
            "Argument 'refdata' must be an xarray DataArray or numpy ndarray!"
            )
    if not isinstance(devdata, np.ndarray):
        if isinstance(devdata, xr.DataArray):
            devdata = devdata.values
        else:
            raise ValueError(
            "Argument 'devdata' must be an xarray DataArray or numpy ndarray!"
            )

    # This method will work if the arrays hve different dimensions
    # but an element-by-element search will not!
    refsum = np.nansum(refdata, dtype=dtype)
    devsum = np.nansum(devdata, dtype=dtype)
    return (not np.abs(devsum - refsum) > dtype(0.0))


def make_directory(
        dir_name,
        overwrite
):
    """
    Creates a directory where benchmark plots/tables will be placed.

    Args:
    -----
    dir_name : str
        Name of the directory to be created.
    overwrite : bool
        Set to True if you wish to overwrite prior contents in
        the directory 'dir_name'
    """
    verify_variable_type(dir_name, str)

    if os.path.isdir(dir_name) and not overwrite:
        msg = f"Directory {dir_name} exists!\n"
        msg += "Pass overwrite=True to overwrite files in that directory."
        raise ValueError(msg)

    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def verify_variable_type(
        var,
        var_type
):
    """
    Convenience routine that will raise a TypeError if a variable's
    type does not match a list of expected types.

    Args:
    -----
    var : variable of any type
        The variable to check.

    var_type : type or tuple of types
        A single type definition (list, str, pandas.Series, etc.)
        or a tuple of type definitions.
    """
    if isinstance(var, var_type):
        return
    raise TypeError( f"{var} is not of type: {var_type}!")
