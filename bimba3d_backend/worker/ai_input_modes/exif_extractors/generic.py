"""
Generic EXIF extractor - works for all cameras using standard EXIF tags

This is the existing _read_exif logic extracted into a reusable class.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Tuple

from PIL import ExifTags, Image

from .base import ExifExtractor


class GenericExtractor(ExifExtractor):
    """
    Generic EXIF extraction using PIL and standard EXIF tags.

    Works for most cameras including DJI, Sony, Canon, Nikon, etc.
    Handles:
    - Standard EXIF IFD tags
    - GPS IFD
    - XMP metadata (fallback)
    """

    def extract(self, image_path: Path) -> Tuple[Dict[str, Any], int, int]:
        """Extract EXIF data using PIL"""
        with Image.open(image_path) as img:
            width, height = img.size
            raw = img.getexif() or {}
            exif: Dict[str, Any] = {}

            # Read main EXIF tags
            for key, value in raw.items():
                name = ExifTags.TAGS.get(key, key)
                # Clean string values to remove null bytes
                if isinstance(value, (str, bytes)):
                    exif[str(name)] = self._clean_string(value)
                else:
                    exif[str(name)] = value

            # Read ExifIFD subdirectory (contains FocalLength, ExposureTime, ISO, etc.)
            if hasattr(raw, "get_ifd"):
                try:
                    exif_ifd = raw.get_ifd(0x8769)  # ExifIFD pointer tag
                    if isinstance(exif_ifd, dict):
                        for k, v in exif_ifd.items():
                            exif_name = ExifTags.TAGS.get(k, k)
                            # Clean string values
                            if isinstance(v, (str, bytes)):
                                v = self._clean_string(v)
                            # Don't overwrite if already present
                            if exif_name not in exif or exif[exif_name] is None:
                                exif[exif_name] = v
                except Exception:
                    pass

            # Ensure GPSInfo is a parsed dict
            if not isinstance(exif.get("GPSInfo"), dict) and hasattr(raw, "get_ifd"):
                try:
                    gps_ifd = raw.get_ifd(0x8825)  # GPS IFD pointer tag
                except Exception:
                    gps_ifd = None

                if isinstance(gps_ifd, dict):
                    gps: Dict[Any, Any] = {}
                    for k, v in gps_ifd.items():
                        gps_name = ExifTags.GPSTAGS.get(k, k)
                        gps[gps_name] = v
                        gps[k] = v
                    exif["GPSInfo"] = gps

            # Extract XMP metadata (DJI and many others use this)
            xmp_attrs = self._extract_xmp_attrs(img.info.get("xmp"))

            # Fallback: If PIL didn't expose XMP, try reading directly from file
            if not xmp_attrs:
                xmp_text = self._extract_xmp_from_file(image_path)
                if xmp_text:
                    xmp_attrs = self._extract_xmp_attrs(xmp_text)

            # Apply XMP fallbacks (fills missing fields)
            self._apply_xmp_fallbacks(exif, xmp_attrs)

            return exif, width, height

    @staticmethod
    def _clean_string(value: Any) -> str:
        """Clean EXIF string values by removing null bytes and extra whitespace."""
        if isinstance(value, bytes):
            text = value.decode("utf-8", errors="ignore")
        elif isinstance(value, str):
            text = value
        else:
            return str(value)

        # Remove null bytes and strip whitespace
        text = text.replace("\x00", "").strip()
        return text

    @staticmethod
    def _extract_xmp_from_file(file_path: Path) -> str | None:
        """Extract XMP packet directly from JPEG file bytes (fallback for PIL)."""
        try:
            with open(file_path, "rb") as f:
                data = f.read()

            xmp_start = data.find(b"<x:xmpmeta")
            xmp_end = data.find(b"</x:xmpmeta>")

            if xmp_start != -1 and xmp_end != -1:
                xmp_bytes = data[xmp_start : xmp_end + len(b"</x:xmpmeta>")]
                return xmp_bytes.decode("utf-8", errors="ignore")
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_xmp_attrs(xmp_blob: Any) -> Dict[str, str]:
        """Extract key-value pairs from XMP metadata."""
        if isinstance(xmp_blob, (bytes, bytearray)):
            text = xmp_blob.decode("utf-8", errors="ignore")
        elif isinstance(xmp_blob, str):
            text = xmp_blob
        else:
            return {}

        attrs: Dict[str, str] = {}
        for key, value in re.findall(r'([A-Za-z0-9_.:-]+)\s*=\s*"([^"]*)"', text):
            attrs[key] = value
        return attrs

    @staticmethod
    def _xmp_pick(attrs: Dict[str, str], keys: list[str]) -> str | None:
        """Pick first available value from XMP attributes."""
        for key in keys:
            val = attrs.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return None

    @staticmethod
    def _as_float(value: Any) -> float | None:
        """
        Convert EXIF value to float, handling all standard EXIF formats.

        EXIF Standard Units (NO conversion needed):
        - FocalLength: always mm
        - ExposureTime: always seconds
        - Altitude: always meters
        - Angles: always degrees
        """
        try:
            if isinstance(value, tuple) and len(value) == 2:
                num, den = value
                if float(den) == 0.0:
                    return None
                return float(num) / float(den)
            if isinstance(value, str):
                # Handle fractions like "1/400"
                if "/" in value:
                    num_s, den_s = value.split("/", 1)
                    den = float(den_s.strip())
                    if den == 0.0:
                        return None
                    return float(num_s.strip()) / den
                # Handle strings with units and/or signs
                cleaned = value.strip().lower()
                if cleaned.startswith('+'):
                    cleaned = cleaned[1:]
                for unit in [' mm', 'mm', ' cm', 'cm', ' m', ' s', 's', ' ms', 'ms', ' deg', 'deg', '°']:
                    cleaned = cleaned.replace(unit, '')
                return float(cleaned.strip())
            return float(value)
        except Exception:
            return None

    def _apply_xmp_fallbacks(self, exif: Dict[str, Any], xmp_attrs: Dict[str, str]) -> None:
        """Apply XMP metadata as fallback for missing EXIF fields."""
        if not xmp_attrs:
            return

        # Lens model
        if not exif.get("LensModel"):
            lens = self._xmp_pick(xmp_attrs, ["aux:Lens", "exifEX:LensModel", "drone-dji:Lens"])
            if lens:
                exif["LensModel"] = lens

        # Focal length
        if exif.get("FocalLength") is None:
            focal = self._xmp_pick(xmp_attrs, ["exif:FocalLength", "tiff:FocalLength"])
            focal_val = self._as_float(focal)
            if focal_val is None:
                # DJI CalibratedFocalLength (pixels, but sometimes in mm range)
                calibrated = self._as_float(
                    self._xmp_pick(xmp_attrs, ["drone-dji:CalibratedFocalLength"])
                )
                if calibrated is not None and 1.0 <= calibrated <= 100.0:
                    focal_val = calibrated
            if focal_val is None:
                # Parse from lens text
                lens_text = str(exif.get("LensModel") or "")
                m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*mm", lens_text, flags=re.IGNORECASE)
                if m:
                    focal_val = self._as_float(m.group(1))
            if focal_val is not None:
                exif["FocalLength"] = focal_val

        # F-number
        if exif.get("FNumber") is None:
            f_number = self._as_float(
                self._xmp_pick(xmp_attrs, ["exif:FNumber", "aux:LensInfoAperture"])
            )
            if f_number is None:
                lens_text = str(exif.get("LensModel") or "")
                m = re.search(r"f\s*/?\s*([0-9]+(?:\.[0-9]+)?)", lens_text, flags=re.IGNORECASE)
                if m:
                    f_number = self._as_float(m.group(1))
            if f_number is not None:
                exif["FNumber"] = f_number

        # Exposure time
        if exif.get("ExposureTime") is None:
            exp = self._as_float(self._xmp_pick(xmp_attrs, ["exif:ExposureTime"]))
            if exp is not None:
                exif["ExposureTime"] = exp

        # ISO
        if exif.get("ISOSpeedRatings") is None and exif.get("PhotographicSensitivity") is None:
            iso = self._as_float(
                self._xmp_pick(
                    xmp_attrs,
                    ["exif:ISOSpeedRatings", "exif:PhotographicSensitivity", "drone-dji:ISO"],
                )
            )
            if iso is not None:
                exif["ISOSpeedRatings"] = iso

        # Date/time
        if exif.get("DateTimeOriginal") is None:
            dt = self._xmp_pick(
                xmp_attrs, ["exif:DateTimeOriginal", "xmp:CreateDate", "drone-dji:CreateDate"]
            )
            if dt:
                exif["DateTimeOriginal"] = dt

        # Pitch/camera angle
        if exif.get("Pitch") is None and exif.get("CameraElevationAngle") is None:
            pitch = self._as_float(
                self._xmp_pick(xmp_attrs, ["drone-dji:GimbalPitchDegree", "drone-dji:FlightPitchDegree"])
            )
            if pitch is not None:
                exif["Pitch"] = pitch

        # Relative altitude (DJI-specific but commonly used)
        if exif.get("RelativeAltitude") is None:
            rel_alt = self._as_float(self._xmp_pick(xmp_attrs, ["drone-dji:RelativeAltitude"]))
            if rel_alt is not None:
                exif["RelativeAltitude"] = rel_alt

        # Absolute altitude
        if exif.get("AbsoluteAltitude") is None:
            abs_alt = self._as_float(self._xmp_pick(xmp_attrs, ["drone-dji:AbsoluteAltitude"]))
            if abs_alt is not None:
                exif["AbsoluteAltitude"] = abs_alt

        # GPS fallback from XMP
        if not isinstance(exif.get("GPSInfo"), dict):
            lat = self._as_float(
                self._xmp_pick(xmp_attrs, ["drone-dji:GpsLatitude", "exif:GPSLatitude"])
            )
            lon = self._as_float(
                self._xmp_pick(xmp_attrs, ["drone-dji:GpsLongitude", "exif:GPSLongitude"])
            )
            alt = self._as_float(
                self._xmp_pick(
                    xmp_attrs,
                    ["drone-dji:AbsoluteAltitude", "drone-dji:RelativeAltitude", "exif:GPSAltitude"],
                )
            )
            gps_payload: Dict[str, float] = {}
            if lat is not None:
                gps_payload["lat"] = lat
            if lon is not None:
                gps_payload["lon"] = lon
            if alt is not None:
                gps_payload["alt"] = alt
            if gps_payload:
                exif["GPSInfo"] = gps_payload
