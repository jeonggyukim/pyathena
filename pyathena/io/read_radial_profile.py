"""Read Athena++ on-the-fly radial-profile binary files."""

from pathlib import Path
import os
import struct

import numpy as np
import xarray as xr

_MAGIC = b"ATHRPRF\0"
_VERSION = 1
_ENDIAN_MARKER = 0x01020304
_BASE_HEADER_BYTES = 148
_DESCRIPTOR_BYTES = 56
_FLOAT64 = 1
_CENTER_VARIABLE = 0
_RADIAL_VARIABLE = 1


def _read_exact(stream, size, description):
    data = stream.read(size)
    if len(data) != size:
        raise ValueError(
            f"Truncated radial-profile file while reading {description}: "
            f"expected {size} bytes, found {len(data)}"
        )
    return data


def _unpack(stream, fmt, description):
    layout = struct.Struct("<" + fmt)
    return layout.unpack(_read_exact(stream, layout.size, description))


def _read_array(stream, dtype, count, description):
    values = np.fromfile(stream, dtype=dtype, count=count)
    if values.size != count:
        raise ValueError(f"Truncated radial-profile file while reading {description}")
    return values


def read_radial_profile(path, center_ids=None, *, metadata_only=False):
    """Return a version-1 rprof file as an xarray.Dataset.

    Radial fields use dimensions (center_id, r). Center positions and
    center-only fields use dimension center_id. Selective reads preserve the
    order supplied in center_ids and raise KeyError for absent IDs.
    With metadata_only=True, read coordinates and validate the file layout
    without reading any radial or center-variable payloads.
    """
    path = Path(path)
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        if _read_exact(stream, 8, "magic") != _MAGIC:
            raise ValueError(f"{path} is not an ATHRPRF file")
        version, endian_marker = _unpack(stream, "II", "version and endian marker")
        if version != _VERSION:
            raise ValueError(
                f"Unsupported radial-profile format version {version}; expected {_VERSION}"
            )
        if endian_marker != _ENDIAN_MARKER:
            raise ValueError(
                f"Invalid endian marker 0x{endian_marker:08x}; expected little-endian"
            )

        feature_flags, header_size, output_number, cycle = _unpack(
            stream, "QQQQ", "fixed header integers"
        )
        (time,) = _unpack(stream, "d", "simulation time")
        nx1, nx2, nx3 = _unpack(stream, "III", "grid dimensions")
        bounds = _unpack(stream, "dddddd", "grid bounds")
        dx, rmax = _unpack(stream, "dd", "radial geometry")
        nsub, ncenter, nr, nvariable = _unpack(stream, "IIII", "array counts")

        expected_header = _BASE_HEADER_BYTES + _DESCRIPTOR_BYTES*nvariable
        if header_size != expected_header:
            raise ValueError(
                f"Invalid header size {header_size}; expected {expected_header}"
            )
        if header_size > file_size:
            raise ValueError("Header extends beyond end of file")
        if nsub != 4:
            raise ValueError(f"Unsupported subcell count {nsub}; expected 4")
        if nr == 0:
            raise ValueError("Radial-profile file has no radial bins")
        if not np.isfinite(dx) or dx <= 0:
            raise ValueError(f"Invalid cell width {dx}")
        if not np.isfinite(rmax) or rmax <= 0:
            raise ValueError(f"Invalid rmax {rmax}")

        descriptors = []
        names = set()
        for index in range(nvariable):
            encoded_name = _read_exact(stream, 32, f"descriptor {index} name")
            try:
                name = encoded_name.split(b"\0", 1)[0].decode("ascii")
            except UnicodeDecodeError as error:
                raise ValueError(f"Descriptor {index} name is not ASCII") from error
            if not name or name in names:
                raise ValueError(f"Empty or duplicate variable name {name!r}")
            names.add(name)
            classification, dtype_code, count, offset = _unpack(
                stream, "IIQQ", f"descriptor {name!r}"
            )
            if classification not in (_CENTER_VARIABLE, _RADIAL_VARIABLE):
                raise ValueError(
                    f"Variable {name!r} has invalid classification {classification}"
                )
            if dtype_code != _FLOAT64:
                raise ValueError(
                    f"Variable {name!r} has unsupported dtype code {dtype_code}"
                )
            expected_count = ncenter*(nr if classification == _RADIAL_VARIABLE else 1)
            if count != expected_count:
                raise ValueError(
                    f"Variable {name!r} has {count} elements; expected {expected_count}"
                )
            if offset < header_size or offset + 8*count > file_size:
                raise ValueError(f"Variable {name!r} points outside the file")
            descriptors.append((name, classification, count, offset))

        id_offset = header_size
        position_offset = id_offset + 8*ncenter
        radius_offset = position_offset + 24*ncenter
        first_variable_offset = radius_offset + 8*nr
        if first_variable_offset > file_size:
            raise ValueError("Center metadata or radial coordinates are truncated")
        previous_end = first_variable_offset
        for name, classification, count, offset in descriptors:
            if offset < previous_end:
                raise ValueError(f"Variable {name!r} overlaps preceding data")
            previous_end = offset + 8*count

        stream.seek(id_offset)
        all_ids = _read_array(stream, "<u8", ncenter, "center IDs")
        stream.seek(position_offset)
        positions = _read_array(
            stream, "<f8", 3*ncenter, "center positions"
        ).reshape(ncenter, 3)
        stream.seek(radius_offset)
        radii = _read_array(stream, "<f8", nr, "radii")

        if np.unique(all_ids).size != ncenter:
            raise ValueError("Center IDs are not unique")
        if not np.all(np.isfinite(positions)):
            raise ValueError("Center positions contain non-finite values")
        expected_radii = np.arange(nr, dtype=np.float64)*dx
        radius_tolerance = 8*np.spacing(max(dx, 1.0))
        if not np.allclose(
            radii, expected_radii, rtol=0.0, atol=radius_tolerance
        ):
            raise ValueError("Radius coordinates are inconsistent with dx")

        if center_ids is None:
            selection = np.arange(ncenter, dtype=np.int64)
        else:
            requested = np.asarray(list(center_ids), dtype=np.uint64)
            index_by_id = {int(value): index for index, value in enumerate(all_ids)}
            missing = [int(value) for value in requested if int(value) not in index_by_id]
            if missing:
                raise KeyError(f"Center IDs not present in {path}: {missing}")
            selection = np.asarray(
                [index_by_id[int(value)] for value in requested], dtype=np.int64
            )

        if not np.isfinite(time):
            raise ValueError("Nonfinite radial-profile output time")
        data_vars = {}
        if metadata_only:
            descriptors = []
        for name, classification, count, offset in descriptors:
            if classification == _RADIAL_VARIABLE:
                values = np.empty((selection.size, nr), dtype=np.float64)
                for target, source in enumerate(selection):
                    stream.seek(offset + 8*int(source)*nr)
                    values[target] = _read_array(
                        stream, "<f8", nr, f"variable {name!r}"
                    )
                data_vars[name] = (("center_id", "r"), values)
            else:
                values = np.empty(selection.size, dtype=np.float64)
                for target, source in enumerate(selection):
                    stream.seek(offset + 8*int(source))
                    values[target] = _unpack(
                        stream, "d", f"variable {name!r}"
                    )[0]
                data_vars[name] = ("center_id", values)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Variable {name!r} contains non-finite values")

    if "vshell" in data_vars and np.any(data_vars["vshell"][1] <= 0):
        raise ValueError("vshell must be positive in every center and radial bin")
    if "rho" in data_vars and np.any(data_vars["rho"][1] <= 0):
        raise ValueError("rho must be positive in every center and radial bin")

    x1min, x1max, x2min, x2max, x3min, x3max = bounds
    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "center_id": all_ids[selection], "r": radii,
            "x1": ("center_id", positions[selection, 0]),
            "x2": ("center_id", positions[selection, 1]),
            "x3": ("center_id", positions[selection, 2]),
        },
        attrs={
            "format_version": version, "feature_flags": feature_flags,
            "num": output_number, "cycle": cycle, "time": time,
            "nx1": nx1, "nx2": nx2, "nx3": nx3,
            "x1min": x1min, "x1max": x1max,
            "x2min": x2min, "x2max": x2max,
            "x3min": x3min, "x3max": x3max,
            "dx": dx, "rmax": rmax, "nsub": nsub, "path": os.fspath(path),
        },
    )
