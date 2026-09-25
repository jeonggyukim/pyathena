"""Read AthenaK text-header/native-binary radial-profile files."""

from pathlib import Path
import os

import numpy as np
import xarray as xr

_SIGNATURE = "Athena radial profile output version="
_VERSION = "1.0"
_ORIGIN_VELOCITIES = ["velocity_x_origin", "velocity_y_origin", "velocity_z_origin"]


def _read_line(stream):
    line = stream.readline()
    if not line.endswith(b"\n"):
        raise ValueError("Truncated radial-profile text header")
    try:
        return line.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise ValueError("Radial-profile text header is not ASCII") from error


def _read_value(stream, key, dtype=int):
    name, separator, value = _read_line(stream).partition("=")
    if not separator or name != key:
        raise ValueError(f"Expected radial-profile header entry {key!r}")
    try:
        return dtype(value)
    except ValueError as error:
        raise ValueError(f"Invalid radial-profile header value for {key!r}: {value!r}") from error


def _read_array(stream, dtype, count, description):
    values = np.fromfile(stream, dtype=dtype, count=count)
    if values.size != count:
        raise ValueError(f"Truncated radial-profile file while reading {description}")
    return values


def read_radial_profile(path, center_ids=None, *, metadata_only=False):
    """Return an AthenaK version-1.0 radial-profile file as an xarray.Dataset.

    The first three stored variables are origin velocities with dimension
    ``center_id``. All later variables have dimensions ``(center_id, r)``.
    Stored names, ordering and values (including NaNs/infinities) are unchanged.
    Coordinates are uint64 ``center_id`` and float64 ``r``, ``x1``, ``x2``, ``x3``.

    ``center_ids=None`` retains file order. An iterable selects centers in the
    requested order, including repeated requests; missing IDs raise KeyError.
    Duplicate IDs in the file raise ValueError in every mode. Providers must
    supply unique IDs. With ``metadata_only=True``, return only coordinates and
    attributes, checking file size without reading variable payloads.

    Payloads use native byte order. Historical descriptor-based ATHRPRF files
    are unsupported. Grid geometry absent from the header is not reconstructed.
    """
    path = Path(path)
    with path.open("rb") as stream:
        signature = _read_line(stream)
        if not signature.startswith(_SIGNATURE):
            raise ValueError(f"{path} is not an Athena radial profile output file")
        version = signature[len(_SIGNATURE):]
        if version != _VERSION:
            raise ValueError(f"Unsupported radial-profile format version {version!r}")
        if _read_value(stream, "size of preheader") != 9:
            raise ValueError("Expected radial-profile preheader size 9")
        time = _read_value(stream, "time", float)
        cycle = _read_value(stream, "cycle")
        output_number = _read_value(stream, "output number")
        feature_flags = _read_value(stream, "feature flags")
        nsub = _read_value(stream, "nsub")
        nbins_subcell_corrected = _read_value(stream, "nbins_subcell_corrected")
        ncenter = _read_value(stream, "number of centers")
        nr = _read_value(stream, "number of radial bins")
        nvariable = _read_value(stream, "number of variables")
        if ncenter < 0 or nr < 1 or nvariable < 3:
            raise ValueError("Invalid radial-profile array counts")
        names_line = _read_line(stream)
        if not names_line.startswith("variables:"):
            raise ValueError("Expected radial-profile variable names")
        names = names_line[len("variables:"):].split()
        if len(names) != nvariable or len(set(names)) != nvariable:
            raise ValueError("Radial-profile variable names must match the count and be unique")
        if names[:3] != _ORIGIN_VELOCITIES:
            raise ValueError("The first three radial-profile variables must be origin velocities")
        if set(names) & {"center_id", "r", "x1", "x2", "x3"}:
            raise ValueError("Radial-profile variable names conflict with coordinates")

        payload_offset = stream.tell()
        expected_size = payload_offset + 8*(7*ncenter + nr + (nvariable-3)*ncenter*nr)
        if os.fstat(stream.fileno()).st_size != expected_size:
            raise ValueError(f"Incorrect radial-profile file size; expected {expected_size} bytes")
        all_ids = _read_array(stream, "=u8", ncenter, "center IDs")
        if np.unique(all_ids).size != ncenter:
            raise ValueError("Center IDs are not unique")
        positions = _read_array(stream, "=f8", 3*ncenter, "center positions").reshape(ncenter, 3)
        radii = _read_array(stream, "=f8", nr, "radii")

        if center_ids is None:
            selection = slice(None)
        else:
            requested = list(center_ids)
            index_by_id = {int(value): index for index, value in enumerate(all_ids)}
            missing = [value for value in requested if value not in index_by_id]
            if missing:
                raise KeyError(f"Center IDs not present in {path}: {missing}")
            selection = np.asarray([index_by_id[value] for value in requested], dtype=np.int64)

        data_vars = {}
        if not metadata_only:
            offset = stream.tell()
            for index, name in enumerate(names):
                radial = index >= 3
                row_size = nr if radial else 1
                if center_ids is None:
                    values = _read_array(stream, "=f8", ncenter*row_size, name)
                    if radial:
                        values = values.reshape(ncenter, nr)
                else:
                    values = np.empty((selection.size, row_size), dtype=np.float64)
                    for target, source in enumerate(selection):
                        stream.seek(offset + 8*int(source)*row_size)
                        values[target] = _read_array(stream, "=f8", row_size, name)
                    if not radial:
                        values = values[:, 0]
                dimensions = ("center_id", "r") if radial else ("center_id",)
                data_vars[name] = (dimensions, values)
                offset += 8*ncenter*row_size

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "center_id": all_ids[selection],
            "r": radii,
            "x1": ("center_id", positions[selection, 0]),
            "x2": ("center_id", positions[selection, 1]),
            "x3": ("center_id", positions[selection, 2]),
        },
        attrs={
            "format_version": version, "feature_flags": feature_flags,
            "num": output_number, "cycle": cycle, "time": time,
            "nsub": nsub, "nbins_subcell_corrected": nbins_subcell_corrected,
            "path": os.fspath(path),
        },
    )
