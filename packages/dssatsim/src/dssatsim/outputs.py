"""DSSAT output parsing and explanation.

Replaces explain_dssat_outputs.py from old dssatsim code. Key differences from v2:
- explain_summary_out() accepts a string (from dssat.output_files["Summary"]) instead of a file path.
- output timeseries come from dssat.output_tables instead of dssat.output[code].
"""

import json
import numpy as np
import pandas as pd

from dssatsim.config import (
    OUTPUT_CODE_TYPES,
    ALL_DSSAT_CDE_FILES,
    CDE_SUFIX_SEP,
    SUMMARY_OUT_CATEGORIES_COLS,
    MISSING_NA_VALUE,
)


# ---------------------------------------------------------------------------
# DSSAT code lookups
# ---------------------------------------------------------------------------

def get_proper_description(code: str) -> str:
    df = pd.read_csv(ALL_DSSAT_CDE_FILES)
    res = df[df["@CDE"] == code]["DESCRIPTION"]
    if res.size == 1:
        return res.item()
    elif res.size > 1:
        return res.to_list()[0]
    return code


def get_dssat_code_characteristic(code: str, value, dssat_output_category: str) -> dict:
    code_no_suffix = code.split(CDE_SUFIX_SEP)[0]
    base_output = {"code": code, "value": value}
    df = pd.read_csv(ALL_DSSAT_CDE_FILES)

    if code_no_suffix not in df["@CDE"].values:
        return base_output

    df = df[df["@CDE"] == code_no_suffix]
    df = df.replace({np.nan: "None"})

    base_output.update({
        "label": df["LABEL"].iloc[0],
        "description": df["DESCRIPTION"].iloc[0],
        "unit": df["UNIT"].iloc[0],
    })
    return base_output


# ---------------------------------------------------------------------------
# Output timeseries helpers
# ---------------------------------------------------------------------------

def rename_year_doy_das_columns(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    return df.copy().rename(columns={
        "@YEAR": f"@year{CDE_SUFIX_SEP}{suffix}",
        "DOY": f"DOY{CDE_SUFIX_SEP}{suffix}",
        "DAS": f"DAS{CDE_SUFIX_SEP}{suffix}",
    })


def average_numerical_output(df: pd.DataFrame) -> dict:
    return df.select_dtypes(include="number").mean().round(3).to_dict()


def explain_output_tables(output_tables: dict) -> dict:
    """Build explanations dict from dssat.output_tables.

    Args:
        output_tables: dict from dssat.output_tables
            e.g. {"PlantGro": df, "SoilWat": df, ...}

    Returns:
        Dict mapping output codes to their averaged, annotated values.
    """
    explanations = {}
    for output_code in OUTPUT_CODE_TYPES:
        if output_code not in output_tables:
            continue
        df = output_tables[output_code]
        df = rename_year_doy_das_columns(df, output_code)
        averages = average_numerical_output(df)
        for code, value in averages.items():
            clean_code = code.replace(f"__{output_code}", "")
            explanations[clean_code] = get_dssat_code_characteristic(
                clean_code, value, output_code
            )
    return explanations


# ---------------------------------------------------------------------------
# Summary.OUT parsing
# ---------------------------------------------------------------------------

def _extract_columns_from_header_line(header_line: str) -> list:
    return [c.replace(".", "") for c in header_line.strip().split() if c != "@"]


def _convert_date_columns(row: pd.Series) -> pd.Series:
    from dssatsim.utils import yrdoy_to_date
    for col in SUMMARY_OUT_CATEGORIES_COLS["DATES"]:
        if col == "HYEAR":
            continue
        try:
            row[col] = yrdoy_to_date(row[col])
        except (ValueError, TypeError):
            row[col] = MISSING_NA_VALUE
    return row


def summary_str_to_table(summary_str: str) -> pd.DataFrame:
    """Parse Summary.OUT content string into a DataFrame.

    Summary.OUT is fixed-width. TNAM (treatment name) may contain spaces,
    so we locate it by character position from the @ header line, extract it
    directly, then parse the remaining columns with whitespace splitting.

    Args:
        summary_str: Full content of Summary.OUT as a string,
                     as returned by dssat.output_files["Summary"].

    Returns:
        DataFrame with one row per treatment run.
    """
    lines = summary_str.splitlines()

    header_lines = [l for l in lines if l.startswith("@") or l.startswith("!")]
    data_lines = [
        l for l in lines
        if l.strip() and not l.startswith("@") and not l.startswith("!")
        and not l.startswith("*")
    ]

    if not data_lines:
        return pd.DataFrame()

    col_names = _extract_columns_from_header_line(header_lines[-1])
    header_line = header_lines[-1]

    # Find the character positions of TNAM and the column after it (FNAM)
    tnam_start = header_line.index("TNAM")
    tnam_end = tnam_start
    while tnam_end < len(header_line) and header_line[tnam_end] != " ":
        tnam_end += 1
    fnam_start = tnam_end
    while fnam_start < len(header_line) and header_line[fnam_start] == " ":
        fnam_start += 1

    rows = []
    for line in data_lines:
        tnam_val = line[tnam_start:fnam_start].strip()
        line_without_tnam = line[:tnam_start] + line[fnam_start:]
        tokens = line_without_tnam.split()
        tokens.insert(8, tnam_val)
        rows.append(tokens[:len(col_names)])

    df = pd.DataFrame(rows, columns=col_names)

    string_cols = {"CR", "MODEL", "EXNAME", "TNAM", "FNAM", "WSTA", "SOIL_ID"}
    for col in df.columns:
        if col not in string_cols and col != "HYEAR":
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().all():
                df[col] = converted

    df = df.apply(_convert_date_columns, axis=1)

    return df


def _explain_xdates(df_summary: pd.DataFrame, columns_oi: list) -> tuple:
    explanations = {}
    df_oi = df_summary[columns_oi].copy()

    for col in [
        "Crop establishment start", "Crop establishment end", "Crop establishment duration",
        "Vegetative growth start", "Vegetative growth end", "Vegetative growth duration",
        "Yield formation start", "Yield formation end", "Yield formation duration",
        "Entire period start", "Entire period end", "Entire period duration",
    ]:
        if col not in df_oi.columns:
            df_oi[col] = MISSING_NA_VALUE

    pdat = df_oi["PDAT"].item()
    edat = df_oi["EDAT"].item()
    adat = df_oi["ADAT"].item()
    mdat = df_oi["MDAT"].item()

    if str(pdat) != MISSING_NA_VALUE and str(edat) != MISSING_NA_VALUE:
        df_oi["Crop establishment start"] = df_oi["PDAT"]
        df_oi["Crop establishment end"] = df_oi["EDAT"] - pd.Timedelta(days=1)
        df_oi["Crop establishment duration"] = (
            df_oi["Crop establishment end"] - df_oi["Crop establishment start"]
        )

    if str(edat) != MISSING_NA_VALUE and str(adat) != MISSING_NA_VALUE:
        df_oi["Vegetative growth start"] = df_oi["EDAT"]
        df_oi["Vegetative growth end"] = df_oi["ADAT"] - pd.Timedelta(days=1)
        df_oi["Vegetative growth duration"] = (
            df_oi["Vegetative growth end"] - df_oi["Vegetative growth start"]
        )

    if str(adat) != MISSING_NA_VALUE and str(mdat) != MISSING_NA_VALUE:
        df_oi["Yield formation start"] = df_oi["ADAT"]
        df_oi["Yield formation end"] = df_oi["MDAT"] - pd.Timedelta(days=1)
        df_oi["Yield formation duration"] = (
            df_oi["Yield formation end"] - df_oi["Yield formation start"]
        )

    if str(pdat) != MISSING_NA_VALUE and str(mdat) != MISSING_NA_VALUE:
        df_oi["Entire period start"] = df_oi["PDAT"]
        df_oi["Entire period end"] = df_oi["MDAT"]
        df_oi["Entire period duration"] = (
            df_oi["Entire period end"] - df_oi["Entire period start"]
        )

    new_cols = [get_proper_description(c) for c in df_oi.columns]
    df_oi.columns = [
        c.replace(" (YrDoy)", "").replace(" (YRDOY)", "") for c in new_cols
    ]
    explanations["Dates"] = df_oi.astype(str).to_dict(orient="records")[0]

    return explanations, df_oi


def explain_summary_out(
    summary_str: str,
    out_fname: str = None,
    exclude_columns: list = None,
) -> tuple:
    """Parse and explain Summary.OUT content.

    Args:
        summary_str: Full content of Summary.OUT as a string.
        out_fname: Optional path to write explanations as JSON.
        exclude_columns: Optional list of columns to drop before processing.

    Returns:
        Tuple of (explanations_dict, df_summary_final).
    """
    explanations = {}
    df_summary = summary_str_to_table(summary_str)

    if exclude_columns:
        df_summary = df_summary.drop(columns=exclude_columns)

    for category, columns in SUMMARY_OUT_CATEGORIES_COLS.items():
        if category == "DATES":
            continue
        df_sub = df_summary[[c for c in columns if c in df_summary.columns]].copy()
        df_sub.columns = [get_proper_description(c) for c in df_sub.columns]
        explanations[category.capitalize()] = df_sub.to_dict(orient="records")[0]

    explanations_xdates, df_xdates = _explain_xdates(
        df_summary, columns_oi=SUMMARY_OUT_CATEGORIES_COLS["DATES"]
    )
    explanations_final = {**explanations, **explanations_xdates}
    df_summary_final = pd.concat([df_summary, df_xdates], axis=1)

    if out_fname is not None:
        with open(out_fname, "w") as f:
            json.dump(explanations_final, f, indent=4)
        print(f"Summary.OUT saved as JSON to {out_fname}")

    return explanations_final, df_summary_final

