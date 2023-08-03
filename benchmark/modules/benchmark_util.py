"""
Utility routines used in benchmark plotting routines.
(These were split off from gcpy.util.)
"""

import os
import shutil
import numpy as np
import xarray as xr
from gcpy import util as gcpy_util
import benchmark_config_files as bcf

# ======================================================================
# %%%%% METHODS %%%%%
# ======================================================================

def get_emissions_varnames(
        commonvars,
        template=None
):
    """
    Will return a list of emissions diagnostic variable names that
    contain a particular search string.

    Args:
        commonvars: list of strs
            A list of commmon variable names from two data sets.
            (This can be obtained with method compare_varnames)
        template: str
            String template for matching variable names corresponding
            to emission diagnostics by sector
            Default Value: None
    Returns:
        varnames: list of strs
            A list of variable names corresponding to emission
            diagnostics for a given species and sector
    """
    gcpy_util.verify_variable_type(commonvars, list)

    # Make sure the commonvars list has at least one element
    if len(commonvars) == 0:
        raise ValueError("No valid variable names were passed!")

    # Define template for emission diagnostics by sector
    if template is None:
        raise ValueError("The template argument was not passed!")

    # Find all emission diagnostics for the given species
    varnames = gcpy_util.filter_names(commonvars, template)

    return varnames


def create_display_name(
        diagnostic_name
):
    """
    Converts a diagnostic name to a more easily digestible name
    that can be used as a plot title or in a table of totals.

    Args:
        diagnostic_name: str
            Name of the diagnostic to be formatted

    Returns:
        display_name: str
            Formatted name that can be used as plot titles or in tables
            of emissions totals.

    Remarks:
        Assumes that diagnostic names will start with either "Emis"
        (for emissions by category) or "Inv" (for emissions by inventory).
        This should be an OK assumption to make since this routine is
        specifically geared towards model benchmarking.
    """
    gcpy_util.verify_variable_type(diagnostic_name, str)

    # Initialize
    display_name = diagnostic_name

    # For restart files, just split at the first underscore and return
    # the text followiong the underscore.  This will preserve certain
    # species names, such as the TransportTracers species CO_25, etc.
    if "SpeciesRst" in display_name:
        return display_name.split("_", 1)[1]

    # Special handling for Inventory totals
    if "INV" in display_name.upper():
        display_name = display_name.replace("_", " ")

    # Replace text
    for var in ["Emis", "EMIS", "emis", "Inv", "INV", "inv"]:
        display_name = display_name.replace(var, "")

    # Replace only the first underscore with a space
    display_name = display_name.replace("_", " ", 1)

    return display_name


def format_number_for_table(
        number,
        max_thresh=1.0e8,
        min_thresh=1.0e-6,
        f_fmt="18.6f",
        e_fmt="18.8e"
):
    """
    Returns a format string for use in the "print_totals" routine.
    If the number is greater than a maximum threshold or smaller
    than a minimum threshold, then use scientific notation format.
    Otherwise use floating-piont format.

    Special case: do not convert 0.0 to exponential notation.

    Args:
    -----
    number : float
        Number to be printed

    max_thresh, min_thresh: float
        If |number| > max_thresh, use scientific notation.
        If |number| < min_thresh, use scientific notation

    f_fmt, e_fmt : str
        The default floating point string and default scientific
        notation string.
        Default values: 18.6f, 18.6e

    Returns:
    --------
    fmt_str : str
        Formatted string that can be inserted into the print
        statement in print_totals.
    """
    abs_number = np.abs(number)

    if not abs_number > 1e-60:
        return f"{number:{f_fmt}}"

    if abs_number > max_thresh or abs_number < min_thresh:
        return f"{number:{e_fmt}}"
    return f"{number:{f_fmt}}"


def print_totals(
        ref,
        dev,
        output_file,
        diff_list,
        masks=None,
):
    """
    Computes and prints Ref and Dev totals (as well as the difference
    Dev - Ref) for two xarray DataArray objects.

    Args:
        ref: xarray DataArray
            The first DataArray to be compared (aka "Reference")
        dev: xarray DataArray
            The second DataArray to be compared (aka "Development")
        output_file: file
            File object denoting a text file where output will be directed.

    Keyword Args (optional):
        masks: dict of xarray DataArray
            Dictionary containing the tropospheric mask arrays
            for Ref and Dev.  If this keyword argument is passed,
            then print_totals will print tropospheric totals.
            Default value: None (i.e. print whole-atmosphere totals)

    Remarks:
        This is an internal method.  It is meant to be called from method
        create_total_emissions_table or create_global_mass_table instead of
        being called directly.
    """

    # ==================================================================
    # Initialization and error checks
    # ==================================================================
    gcpy_util.verify_variable_type(ref, xr.DataArray)
    gcpy_util.verify_variable_type(dev, xr.DataArray)
    gcpy_util.verify_variable_type(diff_list, list)

    # Determine if either Ref or Dev have all NaN values:
    ref_is_all_nan = np.isnan(ref.values).all()
    dev_is_all_nan = np.isnan(dev.values).all()

    # If Ref and Dev do not contain all NaNs, then make sure
    # that Ref and Dev have the same units before proceeding.
    if (not ref_is_all_nan) and (not dev_is_all_nan):
        if ref.units != dev.units:
            msg = f"Ref has units {ref.units}, but Dev has units {dev.units}!"
            raise ValueError(msg)

    # ==================================================================
    # Get the diagnostic name and units
    # ==================================================================
    diagnostic_name = dev.name
    if dev_is_all_nan:
        diagnostic_name = ref.name

    # Create the display name for the table
    display_name = create_display_name(diagnostic_name)

    # Get the species name from the display name
    species_name = display_name
    char = species_name.find(" ")
    if char > 0:
        species_name = display_name[0:char]

    # Special handling for totals
    if "_TOTAL" in diagnostic_name.upper():
        print("-"*90, file=output_file)

    # ==================================================================
    # Sum the Ref array (or set to NaN if missing)
    # ==================================================================
    refarr = ref.values
    if ref_is_all_nan:
        total_ref = np.nan
    else:
        if masks is not None:
            refarr = np.ma.masked_array(refarr, masks["Ref_TropMask"])
        total_ref = np.sum(refarr, dtype=np.float64)

    # ==================================================================
    # Sum the Dev array (or set to NaN if missing)
    # ==================================================================
    devarr = dev.values
    if dev_is_all_nan:
        total_dev = np.nan
    else:
        if masks is not None:
            devarr = np.ma.masked_array(devarr, masks["Dev_TropMask"])
        total_dev = np.sum(devarr, dtype=np.float64)

    # ==================================================================
    # Compute differences (or set to NaN if missing)
    # ==================================================================
    if ref_is_all_nan or dev_is_all_nan:
        diff = np.nan
    else:
        diff = total_dev - total_ref
    has_diffs = abs(diff) > np.float64(0.0)

    # Append to the list of differences.  If no differences then append
    # None.  Duplicates can be stripped out in the calling routine.
    if has_diffs:
        diff_str = " * "
        diff_list.append(species_name)
    else:
        diff_str = ""
        diff_list.append(None)

    # ==================================================================
    # Compute % differences (or set to NaN if missing)
    # If ref is very small, near zero, also set the % diff to NaN
    # ==================================================================
    if np.isnan(total_ref) or np.isnan(total_dev):
        pctdiff = np.nan
    else:
        pctdiff = ((total_dev - total_ref) / total_ref) * 100.0
        if np.abs(total_ref) < 1.0e-15:
            pctdiff = np.nan

    # ==================================================================
    # Write output to file and return
    # ==================================================================
    ref_fmt = format_number_for_table(total_ref)
    dev_fmt = format_number_for_table(total_dev)
    diff_fmt = format_number_for_table(
        diff,
        max_thresh=1.0e4,
        min_thresh=1.0e-4,
        f_fmt="12.3f",
        e_fmt="12.4e"
    )
    pctdiff_fmt = format_number_for_table(
        pctdiff,
        max_thresh=1.0e3,
        min_thresh=1.0e-3,
        f_fmt="8.3f",
        e_fmt="8.1e"
    )

    print(f"{display_name[0:19].ljust(19)}: {ref_fmt}  {dev_fmt}  {diff_fmt}  {pctdiff_fmt}  {diff_str}", file=output_file)

    return diff_list


def add_missing_variables(
        refdata,
        devdata,
        verbose=False,
        **kwargs
):
    """
    Compares two xarray Datasets, "Ref", and "Dev".  For each variable
    that is present  in "Ref" but not in "Dev", a DataArray of missing
    values (i.e. NaN) will be added to "Dev".  Similarly, for each
    variable that is present in "Dev" but not in "Ref", a DataArray
    of missing values will be added to "Ref".
    This routine is mostly intended for benchmark purposes, so that we
    can represent variables that were removed from a new GEOS-Chem
    version by missing values in the benchmark plots.
    NOTE: This function assuming incoming datasets have the same sizes and
    dimensions, which is not true if comparing datasets with different grid
    resolutions or types.

    Args:
        refdata: xarray Dataset
            The "Reference" (aka "Ref") dataset
        devdata: xarray Dataset
            The "Development" (aka "Dev") dataset

    Keyword Args (optional):
        verbose: bool
            Toggles extra debug print output
            Default value: False

    Returns:
        refdata, devdata: xarray Datasets
            The returned "Ref" and "Dev" datasets, with
            placeholder missing value variables added
    """
    # ==================================================================
    # Initialize
    # ==================================================================
    gcpy_util.verify_variable_type(refdata, xr.Dataset)
    gcpy_util.verify_variable_type(devdata, xr.Dataset)

    # Find common variables as well as variables only in one or the other
    vardict = compare_varnames(refdata, devdata, quiet=True)
    refonly = vardict["refonly"]
    devonly = vardict["devonly"]
    # Don't clobber any DataArray attributes
    with xr.set_options(keep_attrs=True):

        # ==============================================================
        # For each variable that is in refdata but not in devdata,
        # add a new DataArray to devdata with the same sizes but
        # containing all NaN's.  This will allow us to represent those
        # variables as missing values # when we plot against refdata.
        # ==============================================================
        devlist = [devdata]
        for var in refonly:
            if verbose:
                print(f"Creating array of NaN in devdata for: {var}")
            darr = gcpy_util.create_blank_dataarray(
                name=refdata[var].name,
                sizes=devdata.sizes,
                coords=devdata.coords,
                attrs=refdata[var].attrs,
                **kwargs
            )
            devlist.append(darr)
        devdata = xr.merge(devlist)

        # ==============================================================
        # For each variable that is in devdata but not in refdata,
        # add a new DataArray to refdata with the same sizes but
        # containing all NaN's.  This will allow us to represent those
        # variables as missing values # when we plot against devdata.
        # ==================================================================
        reflist = [refdata]
        for var in devonly:
            if verbose:
                print(f"Creating array of NaN in refdata for: {var}")
            darr = gcpy_util.create_blank_dataarray(
                name=devdata[var].name,
                sizes=refdata.sizes,
                coords=refdata.coords,
                attrs=devdata[var].attrs,
                **kwargs
            )
            reflist.append(darr)
        refdata = xr.merge(reflist)

    return refdata, devdata


def get_diff_of_diffs(
        ref,
        dev
):
    """
    Generate datasets containing differences between two datasets

    Args:
        ref: xarray Dataset
            The "Reference" (aka "Ref") dataset.
        dev: xarray Dataset
            The "Development" (aka "Dev") dataset

    Returns:
         absdiffs: xarray Dataset
            Dataset containing dev-ref values
         fracdiffs: xarray Dataset
            Dataset containing dev/ref values
    """
    gcpy_util.verify_variable_type(ref, xr.Dataset)
    gcpy_util.verify_variable_type(dev, xr.Dataset)

    # get diff of diffs datasets for 2 datasets
    # limit each pair to be the same type of output (GEOS-Chem Classic or GCHP)
    # and same resolution / extent
    vardict = compare_varnames(ref, dev, quiet=True)
    varlist = vardict["commonvars"]
    # Select only common fields between the Ref and Dev datasets
    ref = ref[varlist]
    dev = dev[varlist]
    if 'nf' not in ref.dims and 'nf' not in dev.dims:
        # if the coords do not align then set time dimensions equal
        try:
            xr.align(dev, ref, join='exact')
        except:
            ref.coords["time"] = dev.coords["time"]
        with xr.set_options(keep_attrs=True):
            absdiffs = dev - ref
            fracdiffs = dev / ref
            for var in dev.data_vars.keys():
                # Ensure the diffs Dataset includes attributes
                absdiffs[var].attrs = dev[var].attrs
                fracdiffs[var].attrs = dev[var].attrs
    elif 'nf' in ref.dims and 'nf' in dev.dims:

        # Include special handling if cubed sphere grid dimension names are different
        # since they changed in MAPL v1.0.0.
        if "lat" in ref.dims and "Xdim" in dev.dims:
            ref_newdimnames = dev.copy()
            for var in dev.data_vars.keys():
                if "Xdim" in dev[var].dims:
                    ref_newdimnames[var].values = ref[var].values.reshape(
                        dev[var].values.shape)
                # NOTE: the reverse conversion is gchp_dev[v].stack(lat=("nf","Ydim")).transpose(
                # "time","lev","lat","Xdim").values

        with xr.set_options(keep_attrs=True):
            absdiffs = dev.copy()
            fracdiffs = dev.copy()
            for var in dev.data_vars.keys():
                if "Xdim" in dev[var].dims or "lat" in dev[var].dims:
                    absdiffs[var].values = dev[var].values - ref[var].values
                    fracdiffs[var].values = dev[var].values / ref[var].values
                    # NOTE: The diffs Datasets are created without variable
                    # attributes; we have to reattach them
                    absdiffs[var].attrs = dev[var].attrs
                    fracdiffs[var].attrs = dev[var].attrs
    else:
        print('Diff-of-diffs plot supports only identical grid types (lat/lon or cubed-sphere)' + \
              ' within each dataset pair')
        raise ValueError

    return absdiffs, fracdiffs


def compare_varnames(
        refdata,
        devdata,
        refonly=None,
        devonly=None,
        quiet=False):
    """
    Finds variables that are common to two xarray Dataset objects.

    Args:
        refdata: xarray Dataset
            The first Dataset to be compared.
            (This is often referred to as the "Reference" Dataset.)
        devdata: xarray Dataset
            The second Dataset to be compared.
            (This is often referred to as the "Development" Dataset.)

    Keyword Args (optional):
        quiet: bool
            Set this flag to True if you wish to suppress printing
            informational output to stdout.
            Default value: False

    Returns:
        vardict: dict of lists of str
            Dictionary containing several lists of variable names:
            Key              Value
            -----            -----
            commonvars       List of variables that are common to
                             both refdata and devdata
            commonvarsOther  List of variables that are common
                             to both refdata and devdata, but do
                             not have lat, lon, and/or level
                             dimensions (e.g. index variables).
            commonvars2D     List of variables that are common to
                             common to refdata and devdata, and that
                             have lat and lon dimensions, but not level.
            commonvars3D     List of variables that are common to
                             refdata and devdata, and that have lat,
                             lon, and level dimensions.
            commonvarsData   List of all commmon 2D or 3D data variables,
                             excluding index variables.  This is the
                             list of "plottable" variables.
            refonly          List of 2D or 3D variables that are only
                             present in refdata.
            devonly          List of 2D or 3D variables that are only
                             present in devdata
    """
    gcpy_util.verify_variable_type(refdata, xr.Dataset)
    gcpy_util.verify_variable_type(devdata, xr.Dataset)

    refvars = list(refdata.data_vars.keys())
    devvars = list(devdata.data_vars.keys())
    commonvars = sorted(list(set(refvars).intersection(set(devvars))))
    refonly = [v for v in refvars if v not in devvars]
    devonly = [v for v in devvars if v not in refvars]
    dimmismatch = [v for v in commonvars if refdata[v].ndim != devdata[v].ndim]
    # Assume plottable data has lon and lat
    # This is OK for purposes of benchmarking
    #  -- Bob Yantosca (09 Feb 2023)
    commonvars_data = [
        var for var in commonvars if (
            ("lat" in refdata[var].dims or "Ydim" in refdata[var].dims)
            and
            ("lon" in refdata[var].dims or "Xdim" in refdata[var].dims)
        )
    ]
    commonvars_other = [
        var for var in commonvars if (
           var not in commonvars_data
        )
    ]
    commonvars_2d = [
        var for var in commonvars if (
            (var in commonvars_data) and ("lev" not in refdata[var].dims)
        )
    ]
    commonvars_3d = [
        var for var in commonvars if (
            (var in commonvars_data) and ("lev" in refdata[var].dims)
        )
    ]

    # Print information on common and mismatching variables,
    # as well as dimensions
    if not quiet:
        print("\nComparing variable names in compare_varnames")
        print(f"{len(commonvars)} common variables")
        if len(refonly) > 0:
            print(f"{len(refonly)} variables in ref only (skip)")
            print(f"   Variable names: {refonly}")
        else:
            print("0 variables in ref only")
            if len(devonly) > 0:
                print(f"len({devonly} variables in dev only (skip)")
                print(f"   Variable names: {devonly}")
            else:
                print("0 variables in dev only")
                if len(dimmismatch) > 0:
                    print(f"{dimmismatch} common variables have different dimensions")
                    print(f"   Variable names: {dimmismatch}")
                else:
                    print("All variables have same dimensions in ref and dev")

    # For safety's sake, remove the 0-D and 1-D variables from
    # commonvarsData, refonly, and devonly.  This will ensure that
    # these lists will only contain variables that can be plotted.
    commonvars_data = [var for var in commonvars if var not in commonvars_other]
    refonly = [var for var in refonly if var not in commonvars_other]
    devonly = [var for var in devonly if var not in commonvars_other]

    return {
        "commonvars": commonvars,
        "commonvars2D": commonvars_2d,
        "commonvars3D": commonvars_3d,
        "commonvarsData": commonvars_data,
        "commonvarsOther": commonvars_other,
        "refonly": refonly,
        "devonly": devonly
    }


def get_filepath(
        datadir,
        col,
        date,
        is_gchp=False,
        gchp_res="c00",
        gchp_is_pre_14_0=False
):
    """
    Routine to return file path for a given GEOS-Chem "Classic"
    (aka "GCC") or GCHP diagnostic collection and date.

    Args:
        datadir: str
            Path name of the directory containing GCC or GCHP data files.
        col: str
            Name of collection (e.g. Emissions, SpeciesConc, etc.)
            for which file path will be returned.
        date: numpy.datetime64
            Date for which file paths are requested.

    Keyword Args (optional):
        is_gchp: bool
            Set this switch to True to obtain file pathnames to
            GCHP diagnostic data files. If False, assumes GEOS-Chem "Classic"

        gchp_res: str
            Cubed-sphere resolution of GCHP data grid.
            Only needed for restart files.
            Default value: "c00".

        gchp_is_pre_14_0: bool
            Set this switch to True to obtain GCHP file pathnames used in
            versions before 14.0. Only needed for restart files.

    Returns:
        path: str
            Pathname for the specified collection and date.
    """
    gcpy_util.verify_variable_type(datadir, str)
    gcpy_util.verify_variable_type(col, str)
    gcpy_util.verify_variable_type(date, np.datetime64)

    # Set filename template, extension, separator, and date string from
    # the collection, date, and data directory arguments
    separator = "_"
    extension = "z.nc4"
    date_str = np.datetime_as_string(date, unit="m")
    if is_gchp:
        if "Restart" in col:
            extension = ".nc4"
            date_str = np.datetime_as_string(date, unit="s")
            if gchp_is_pre_14_0:
                file_tmpl = os.path.join(
                    datadir,
                    "gcchem_internal_checkpoint.restart."
                )
            else:
                file_tmpl = os.path.join(
                    datadir,
                    "GEOSChem.Restart."
                )
        else:
            file_tmpl = os.path.join(datadir, f"GEOSChem.{col}.")
    else:
        if "Emissions" in col:
            file_tmpl = os.path.join(datadir, "HEMCO_diagnostics.")
            extension = ".nc"
            separator = ""
        elif "Restart" in col:
            file_tmpl = os.path.join(datadir, "GEOSChem.Restart.")
        else:
            file_tmpl = os.path.join(datadir, f"GEOSChem.{col}.")
    if isinstance(date_str, np.str_):
        date_str = str(date_str)
    date_str = date_str.replace("T", separator)
    date_str = date_str.replace("-", "")
    date_str = date_str.replace(":", "")

    # Set file path. Include grid resolution if GCHP restart file.
    path = file_tmpl + date_str + extension
    if is_gchp and "Restart" in col and not gchp_is_pre_14_0:
        path = file_tmpl + date_str[:len(date_str)-2] + "z." + gchp_res + extension

    return path


def get_filepaths(
        datadir,
        collections,
        dates,
        is_gchp=False,
        gchp_res="c00",
        gchp_is_pre_14_0=False
):
    """
    Routine to return filepaths for a given GEOS-Chem "Classic"
    (aka "GCC") or GCHP diagnostic collection.

    Args:
        datadir: str
            Path name of the directory containing GCC or GCHP data files.
        collections: list of str
            Names of collections (e.g. Emissions, SpeciesConc, etc.)
            for which file paths will be returned.
        dates: array of numpy.datetime64
            Array of dates for which file paths are requested.

    Keyword Args (optional):
        is_gchp: bool
            Set this switch to True to obtain file pathnames to
            GCHP diagnostic data files. If False, assumes GEOS-Chem "Classic"

        gchp_res: str
            Cubed-sphere resolution of GCHP data grid.
            Only needed for restart files.
            Default value: "c00".

        gchp_is_pre_14_0: bool
            Set this switch to True to obtain GCHP file pathnames used in
            versions before 14.0. Only needed for diagnostic files.

    Returns:
        paths: 2D list of str
            A list of pathnames for each specified collection and date.
            First dimension is collection, and second is date.
    """

    # ==================================================================
    # Initialization
    # ==================================================================

    # If collections is passed as a scalar
    # make it a list so that we can iterate
    if not isinstance(collections, list):
        collections = [collections]

    # Create the return variable
    rows, cols = (len(collections), len(dates))
    paths = [[''] * cols] * rows

    # ==================================================================
    # Create the file list
    # ==================================================================
    for c_ind, collection in enumerate(collections):

        separator = "_"
        extension = "z.nc4"
        if is_gchp:
            # ---------------------------------------
            # Get the file path template for GCHP
            # ---------------------------------------
            if "Restart" in collection:
                extension = ".nc4"
                if gchp_is_pre_14_0:
                    file_tmpl = os.path.join(
                        datadir,
                        "gcchem_internal_checkpoint.restart."
                    )
                else:
                    file_tmpl = os.path.join(
                        datadir,
                        "GEOSChem.Restart."
                    )
            else:
                file_tmpl = os.path.join(
                    datadir,
                    f"GEOSChem.{collection}."
                )
        else:
            # ---------------------------------------
            # Get the file path template for GCC
            # ---------------------------------------
            if "Emissions" in collection:
                file_tmpl = os.path.join(
                    datadir,
                    "HEMCO_diagnostics."
                )
                separator = ""
                extension = ".nc"
            elif "Restart" in collection:
                file_tmpl = os.path.join(
                    datadir,
                    "GEOSChem.Restart."
                )

            else:
                file_tmpl = os.path.join(
                    datadir,
                    f"GEOSChem.{collection}."
                )

        # --------------------------------------------
        # Create a list of files for each date/time
        # --------------------------------------------
        for d_ind, date in enumerate(dates):
            if is_gchp and "Restart" in collection:
                date_time = str(np.datetime_as_string(date, unit="s"))
            else:
                date_time = str(np.datetime_as_string(date, unit="m"))
            date_time = date_time.replace("T", separator)
            date_time = date_time.replace("-", "")
            date_time = date_time.replace(":", "")

            # Set file path. Include grid resolution if GCHP restart file.
            paths[c_ind][d_ind] = file_tmpl + date_time + extension
            if is_gchp and "Restart" in collection and not gchp_is_pre_14_0:
                paths[c_ind][d_ind] = \
                    file_tmpl + \
                    date_time[:len(date_time)-2] + "z." + \
                    gchp_res + extension

    return paths


def get_gcc_filepath(
        outputdir,
        collection,
        day,
        time
):
    '''
    Routine for getting filepath of GEOS-Chem Classic output

    Args:
        outputdir: str
             Path of the OutputDir directory
        collection: str
             Name of output collection, e.g. Emissions or SpeciesConc
        day: str
             Number day of output, e.g. 31
        time: str
             Z time of output, e.g. 1200z

    Returns:
        filepath: str
             Path of requested file
    '''
    if collection == "Emissions":
        filepath = os.path.join(
            outputdir,
            f"HEMCO_diagnostics.{day}{time}.nc"
        )
    else:
        filepath = os.path.join(
            outputdir,
            f"GEOSChem.{collection}.{day}_{time}z.nc4"
        )
    return filepath


def get_gchp_filepath(
        outputdir,
        collection,
        day,
        time
):
    '''
    Routine for getting filepath of GCHP output

    Args:
        outputdir: str
             Path of the OutputDir directory
        collection: str
             Name of output collection, e.g. Emissions or SpeciesConc
        day: str
             Number day of output, e.g. 31
        time: str
             Z time of output, e.g. 1200z

    Returns:
        filepath: str
             Path of requested file
    '''

    filepath = os.path.join(
        outputdir,
        f"GCHP.{collection}.{day}_{time}z.nc4"
    )
    return filepath


def trim_cloud_benchmark_label(
        label
):
    """
    Removes the first part of the cloud benchmark label string
    (e.g. "gchp-c24-1Hr", "gcc-4x5-1Mon", etc) to avoid clutter.
    """
    gcpy_util.verify_variable_type(label, str)

    for var in [
        "gcc-4x5-1Hr",
        "gchp-c24-1Hr",
        "gcc-4x5-1Mon",
        "gchp-c24-1Mon",
    ]:
        if var in label:
            label.replace(var, "")

    return label


def diff_list_to_text(
        refstr,
        devstr,
        diff_list,
        fancy_format=False
):
    """
    Converts a list of species/emissions/inventories/diagnostics that
    show differences between GEOS-Chem versions ot a printable text
    string.

    Args:
    -----
    diff_list : list
        List to be converted into text.  "None" values will be dropped.
    fancy_format: bool
        Set to True if you wish output text to be bookended with '###'.

    Returns:
    diff_text : str
        String with concatenated list values.
    """
    gcpy_util.verify_variable_type(diff_list, list)

    # Use "Dev" and "Ref" for inserting into a header
    if fancy_format:
        refstr = "Ref"
        devstr = "Dev"

    # Strip out duplicates from diff_list
    # Prepare a message about species differences (or alternate msg)
    diff_list = gcpy_util.unique_values(diff_list, drop=[None])

    # Print the text
    n_diff = len(diff_list)
    if n_diff > 0:
        diff_text = f"{devstr} and {refstr} show {n_diff} differences"
    else:
        diff_text = f"{devstr} and {refstr} are identical"

    # If we are placing the text in a header,
    # then trim the length of diff_text to fit.
    if fancy_format:
        diff_text = gcpy_util.wrap_text(
            diff_text,
            width=83
        )
        diff_text = f"### {diff_text : <82}{'###'}"

    return diff_text.strip()


def diff_of_diffs_toprow_title(
        config,
        model
):
    """
    Creates the diff-of-diffs plot title for the top row of the
    six-plot output.  If the title string is too long (as empirically
    determined), then a newline will be inserted in order to prevent
    the title strings from overlapping.

    Args:
    -----
    config : dict
       Dictionary containing the benchmark options (as read from a
       YAML file such as 1mo_benchmark.yml, etc.)
    model: str
       The model to plot.  Accepted values are "gcc" or "gchp".

    Returns:
    --------
    title: str
        The plot title string for the diff-of-diff
    """
    gcpy_util.verify_variable_type(config, dict)
    gcpy_util.verify_variable_type(model, str)

    if not "gcc" in model and not "gchp" in model:
        msg = "The 'model' argument must be either 'gcc' or 'gchp'!"
        raise ValueError(msg)

    title = (
        config["data"]["dev"][model]["version"]
        + " - "
        + config["data"]["ref"][model]["version"]
    )

    if len(title) > 40:
        title = (
            config["data"]["dev"][model]["version"]
            + " -\n"
            + config["data"]["ref"][model]["version"]
        )

    return title


def get_species_categories(
        benchmark_type="FullChemBenchmark"
):
    """
    Returns the list of benchmark categories that each species
    belongs to.  This determines which PDF files will contain the
    plots for the various species.

    Args:
        benchmark_type: str
            Specifies the type of the benchmark:
            FullChemBenchmark (default)
            TransportTracersBenchmark
            CH4Benchmark

    Returns:
        spc_cat_dict: dict
            A nested dictionary of categories (and sub-categories)
            and the species belonging to each.

    NOTE: The benchmark categories are specified in YAML file
    benchmark_species.yml.
    """
    spc_cat_dict = gcpy_util.read_config_file(bcf.BENCHMARK_CAT_YAML)
    return spc_cat_dict[benchmark_type]


def archive_config_file(
        config_file_path,
        dst
):
    """
    Copies a configuration file in a destination folder.

    Args:
        config_file_path : str
            Path to the configuration file
        dst: str
            Name of the folder where the YAML file containing
            benchmark categories ("benchmark_species.yml")
            will be written.
    """
    gcpy_util.verify_variable_type(config_file_path, str)
    gcpy_util.verify_variable_type(dst, str)

    orig_basename = os.path.basename(config_file_path)
    copy = os.path.join(dst, orig_basename)
    if not os.path.exists(copy):
        print(f"\nArchiving {orig_basename} in {dst}")
        shutil.copyfile(config_file_path, copy)


def add_lumped_species_to_dataset(
        dset,
        lspc_dict=None,
        lspc_yaml="",
        verbose=False,
        overwrite=False,
        prefix="SpeciesConcVV_",
):
    """
    Function to calculate lumped species concentrations and add
    them to an xarray Dataset. Lumped species definitions may be passed
    as a dictionary or a path to a yaml file. If neither is passed then
    the lumped species yaml file stored in gcpy is used. This file is
    customized for use with benchmark simuation SpeciesConc diagnostic
    collection output.

    Args:
        dset: xarray Dataset
            An xarray Dataset object prior to adding lumped species.

    Keyword Args (optional):
        lspc_dict: dictionary
            Dictionary containing list of constituent species and their
            integer scale factors per lumped species.
            Default value: None
        lspc_yaml: str
            Name of the YAML file containing the list of constituent s
            species and their integer scale factors per lumped species.
            Default value: ""
        verbose: bool
            Whether to print informational output.
            Default value: False
        overwrite: bool
            Whether to overwrite an existing species dataarray in a dataset
            if it has the same name as a new lumped species. If False and
            overlapping names are found then the function will raise an error.
            Default value: False
        prefix: str
            Prefix to prepend to new lumped species names. This argument is
            also used to extract an existing dataarray in the dataset with
            the correct size and dimensions to use during initialization of
            new lumped species dataarrays.
            Default value: "SpeciesConcVV_"

    Returns:
        dset: xarray Dataset
            A new xarray Dataset object containing all of the original
            species plus new lumped species.
    """
    gcpy_util.verify_variable_type(dset, xr.Dataset)

    # Default is to add all benchmark lumped species.
    # Can overwrite by passing a dictionary
    # or a yaml file path containing one
    assert not (
        lspc_dict is not None and lspc_yaml != ""
    ), "Cannot pass both lspc_dict and lspc_yaml. Choose one only."
    if lspc_dict is None and lspc_yaml == "":
        lspc_dict = gcpy_util.read_config_file(bcf.LUMPED_SPC_YAML)
    elif lspc_dict is None and lspc_yaml != "":
        lspc_dict = gcpy_util.read_config_file(lspc_yaml)

    # Make sure attributes are transferred when copying dataset / dataarrays
    with xr.set_options(keep_attrs=True):

        # Get a dummy DataArray to use for initialization
        dummy_darr = None
        for var in dset.data_vars:
            if prefix in var or prefix.replace("VV", "") in var:
                dummy_darr = dset[var]
                dummy_type = dummy_darr.dtype
                dummy_shape = dummy_darr.shape
                break
        if dummy_darr is None:
            msg = "Invalid prefix: " + prefix
            raise ValueError(msg)

        # Create a list with a copy of the dummy DataArray object
        n_lumped_spc = len(lspc_dict)
        lumped_spc = [None] * n_lumped_spc
        for var, spcname in enumerate(lspc_dict):
            lumped_spc[var] = dummy_darr.copy(deep=False)
            lumped_spc[var].name = prefix + spcname
            lumped_spc[var].values = np.full(dummy_shape, 0.0, dtype=dummy_type)

        # Loop over lumped species list
        for var, lspc in enumerate(lumped_spc):

            # Search key for lspc_dict is lspc.name minus the prefix
            char = lspc.name.find("_")
            key = lspc.name[char+1:]

            # Check if overlap with existing species
            if lspc.name in dset.data_vars and overwrite:
                dset.drop(lspc.name)
            else:
                assert(lspc.name not in dset.data_vars), \
                    f"{lspc.name} already in dataset. To overwrite pass overwrite=True."

            # Verbose prints
            if verbose:
                print(f"Creating {lspc.name}")

            # Loop over and sum constituent species values
            num_spc = 0
            for _, spcname in enumerate(lspc_dict[key]):
                varname = prefix + spcname
                if varname not in dset.data_vars:
                    if verbose:
                        print(f"Warning: {varname} needed for {lspc_dict[key][spcname]} not in dataset")
                    continue
                if verbose:
                    print(f" -> adding {varname} with scale {lspc_dict[key][spcname]}")
                lspc.values += dset[varname].values * lspc_dict[key][spcname]
                num_spc += 1

            # Replace values with NaN if no species found in dataset
            if num_spc == 0:
                if verbose:
                    print("No constituent species found! Setting to NaN.")
                lspc.values = np.full(lspc.shape, np.nan)

        # Insert the DataSet into the list of DataArrays
        # so that we can only do the merge operation once
        lumped_spc.insert(0, dset)
        dset = xr.merge(lumped_spc)

    return dset
