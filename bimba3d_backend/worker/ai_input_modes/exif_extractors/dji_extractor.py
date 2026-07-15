"""
DJI-specific EXIF extractor

Optimized for DJI drone cameras using PIL + XMP parsing.
Handles all DJI models with enhanced field extraction.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Any, Tuple

from PIL import Image, ExifTags

from .base import BaseExtractor

logger = logging.getLogger(__name__)


class DJIExtractor(BaseExtractor):
    """
    DJI-specific EXIF extraction using PIL + XMP parsing.

    Supports all DJI drone models:
    - M3E, M3T, M4E (Mavic 3/4 series)
    - FC2403, FC7503 (Mini/Air series)
    - EP800 (Enterprise)
    - And all other DJI drones
    """

    # Known DJI camera sensor sizes (width, height) in mm
    SENSOR_DATABASE = {
        "M3E": (17.3, 13.0),      # Mavic 3 Enterprise - 4/3" sensor
        "M3T": (9.6, 7.2),        # Mavic 3 Thermal - 1/2" sensor (wide camera)
        "M4E": (17.3, 13.0),      # Mavic 4 Enterprise - 4/3" sensor
        "FC2403": (9.6, 7.2),     # Mavic 2 Enterprise / Mini - 1/2" sensor
        "FC7503": (9.6, 7.2),     # Air series - 1/2" sensor
        "EP800": (13.2, 8.8),     # Enterprise drone - 1" sensor
        "FC6310": (17.3, 13.0),   # Mavic 3 (older naming)
        "FC3170": (13.2, 8.8),    # Mavic 2 Pro
        "FC6510": (9.6, 7.2),     # Air 3
        "FC330": (13.2, 8.8),     # Phantom 3
        "FC350": (13.2, 8.8),     # Phantom 4
    }

    def extract(self, image_path: Path) -> Tuple[Dict[str, Any], int, int]:
        """
        Extract EXIF from DJI drone image.

        Process:
        1. Read standard EXIF tags via PIL
        2. Read ExifIFD for focal length, exposure, ISO
        3. Parse XMP metadata for DJI-specific fields
        4. Apply pitch fallback (GimbalPitchDegree if Pitch missing)
        5. Add sensor dimensions from database

        Args:
            image_path: Path to DJI image

        Returns:
            (exif_dict, width, height)
        """
        with Image.open(image_path) as img:
            width, height = img.size
            raw = img.getexif() or {}
            exif: Dict[str, Any] = {}

            # Read main EXIF tags
            for key, value in raw.items():
                name = ExifTags.TAGS.get(key, key)
                # Clean string values
                if isinstance(value, (str, bytes)):
                    exif[str(name)] = self._clean_string(value)
                else:
                    exif[str(name)] = value

            # Read ExifIFD subdirectory (FocalLength, ExposureTime, ISO, etc.)
            if hasattr(raw, "get_ifd"):
                try:
                    exif_ifd = raw.get_ifd(0x8769)  # ExifIFD pointer
                    if isinstance(exif_ifd, dict):
                        for k, v in exif_ifd.items():
                            exif_name = ExifTags.TAGS.get(k, k)
                            if isinstance(v, (str, bytes)):
                                v = self._clean_string(v)
                            if exif_name not in exif or exif[exif_name] is None:
                                exif[exif_name] = v
                except Exception as e:
                    logger.debug(f"Could not read ExifIFD: {e}")

            # Read GPS IFD
            if not isinstance(exif.get("GPSInfo"), dict) and hasattr(raw, "get_ifd"):
                try:
                    gps_ifd = raw.get_ifd(0x8825)  # GPS IFD pointer
                    if isinstance(gps_ifd, dict):
                        gps: Dict[Any, Any] = {}
                        for k, v in gps_ifd.items():
                            gps_name = ExifTags.GPSTAGS.get(k, k)
                            gps[gps_name] = v
                            gps[k] = v
                        exif["GPSInfo"] = gps
                except Exception as e:
                    logger.debug(f"Could not read GPS IFD: {e}")

            # Extract XMP metadata (DJI stores critical data here)
            xmp_attrs = self._extract_xmp_attrs(img.info.get("xmp"))

            # Fallback: Read XMP directly from file if PIL didn't expose it
            if not xmp_attrs:
                xmp_text = self._extract_xmp_from_file(image_path)
                if xmp_text:
                    xmp_attrs = self._extract_xmp_attrs(xmp_text)

            # Apply XMP fallbacks for missing EXIF fields
            self._apply_xmp_fallbacks(exif, xmp_attrs)

            # DJI-specific enhancements
            self._apply_dji_enhancements(exif, xmp_attrs)

            # Normalize GPS to decimal degrees for easier consumption
            self._normalize_gps(exif)

            # Add sensor dimensions if available
            model = str(exif.get("Model", "")).strip()
            if model in self.SENSOR_DATABASE and "SensorWidth" not in exif:
                sensor_width, sensor_height = self.SENSOR_DATABASE[model]
                exif["SensorWidth"] = sensor_width
                exif["SensorHeight"] = sensor_height

            return exif, width, height

    @staticmethod
    def _clean_string(value: Any) -> str:
        """Remove null bytes and whitespace from EXIF strings"""
        if isinstance(value, bytes):
            text = value.decode("utf-8", errors="ignore")
        elif isinstance(value, str):
            text = value
        else:
            return str(value)
        return text.replace("\x00", "").strip()

    @staticmethod
    def _extract_xmp_from_file(file_path: Path) -> str | None:
        """Extract XMP packet directly from JPEG file"""
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
        """Parse XMP metadata into key-value pairs"""
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
        """Pick first available value from XMP attributes"""
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

        Supports:
        - Tuples: (1, 400) → 0.0025
        - Fractions: "1/400" → 0.0025
        - Strings with units: "8.8 mm", "-90 deg"
        - Strings with signs: "+15.495"
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
        """Fill missing EXIF fields from XMP metadata"""
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
                calibrated = self._as_float(
                    self._xmp_pick(xmp_attrs, ["drone-dji:CalibratedFocalLength"])
                )
                if calibrated is not None and 1.0 <= calibrated <= 100.0:
                    focal_val = calibrated
            if focal_val is None:
                lens_text = str(exif.get("LensModel") or "")
                m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*mm", lens_text, flags=re.IGNORECASE)
                if m:
                    focal_val = self._as_float(m.group(1))
            if focal_val is not None:
                exif["FocalLength"] = focal_val

        # Other standard EXIF fields from XMP
        if exif.get("FNumber") is None:
            f_number = self._as_float(
                self._xmp_pick(xmp_attrs, ["exif:FNumber", "aux:LensInfoAperture"])
            )
            if f_number is not None:
                exif["FNumber"] = f_number

        if exif.get("ExposureTime") is None:
            exp = self._as_float(self._xmp_pick(xmp_attrs, ["exif:ExposureTime"]))
            if exp is not None:
                exif["ExposureTime"] = exp

        if exif.get("ISOSpeedRatings") is None and exif.get("PhotographicSensitivity") is None:
            iso = self._as_float(
                self._xmp_pick(
                    xmp_attrs,
                    ["exif:ISOSpeedRatings", "exif:PhotographicSensitivity", "drone-dji:ISO"],
                )
            )
            if iso is not None:
                exif["ISOSpeedRatings"] = iso

        if exif.get("DateTimeOriginal") is None:
            dt = self._xmp_pick(
                xmp_attrs, ["exif:DateTimeOriginal", "xmp:CreateDate", "drone-dji:CreateDate"]
            )
            if dt:
                exif["DateTimeOriginal"] = dt

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
            if lat is not None or lon is not None or alt is not None:
                gps_payload: Dict[str, float] = {}
                if lat is not None:
                    gps_payload["lat"] = lat
                if lon is not None:
                    gps_payload["lon"] = lon
                if alt is not None:
                    gps_payload["alt"] = alt
                exif["GPSInfo"] = gps_payload

    def _apply_dji_enhancements(self, exif: Dict[str, Any], xmp_attrs: Dict[str, str]) -> None:
        """
        Apply DJI-specific field extraction and fallbacks.

        Critical for AI training:
        - RelativeAltitude (for GSD calculation)
        - Pitch/GimbalPitchDegree (for camera angle classification)
        """
        # Relative altitude (critical for GSD)
        if exif.get("RelativeAltitude") is None:
            rel_alt = self._as_float(self._xmp_pick(xmp_attrs, ["drone-dji:RelativeAltitude"]))
            if rel_alt is not None:
                exif["RelativeAltitude"] = rel_alt

        # Absolute altitude
        if exif.get("AbsoluteAltitude") is None:
            abs_alt = self._as_float(self._xmp_pick(xmp_attrs, ["drone-dji:AbsoluteAltitude"]))
            if abs_alt is not None:
                exif["AbsoluteAltitude"] = abs_alt

        # PITCH FALLBACK - Critical fix for FC7503 and similar models
        # Try multiple DJI pitch field names in order of preference
        if exif.get("Pitch") is None and exif.get("CameraElevationAngle") is None:
            pitch = self._as_float(
                self._xmp_pick(
                    xmp_attrs,
                    [
                        "drone-dji:GimbalPitchDegree",    # Most common
                        "drone-dji:FlightPitchDegree",    # Alternative
                        "drone-dji:CameraPitch",          # Some models
                        "drone-dji:GimbalPitch",          # Variation
                    ]
                )
            )
            if pitch is not None:
                exif["Pitch"] = pitch
                logger.debug(f"Applied pitch fallback from XMP: {pitch}")

    def _normalize_gps(self, exif: Dict[str, Any]) -> None:
        """
        Normalize GPS data to simplified lat/lon/alt format.

        Converts DMS (degrees/minutes/seconds) to decimal degrees.
        Adds simplified 'lat', 'lon', 'alt' keys to GPSInfo dict for easier access.
        """
        gps = exif.get("GPSInfo")
        if not isinstance(gps, dict):
            return

        # Check if already normalized
        if "lat" in gps and "lon" in gps:
            return

        # Extract latitude
        lat_ref = gps.get("GPSLatitudeRef") or gps.get(1)
        lat_dms = gps.get("GPSLatitude") or gps.get(2)

        if lat_dms and isinstance(lat_dms, (tuple, list)) and len(lat_dms) == 3:
            deg = self._as_float(lat_dms[0]) or 0.0
            min = self._as_float(lat_dms[1]) or 0.0
            sec = self._as_float(lat_dms[2]) or 0.0
            lat = deg + (min / 60.0) + (sec / 3600.0)
            if str(lat_ref).upper() == "S":
                lat = -lat
            gps["lat"] = lat

        # Extract longitude
        lon_ref = gps.get("GPSLongitudeRef") or gps.get(3)
        lon_dms = gps.get("GPSLongitude") or gps.get(4)

        if lon_dms and isinstance(lon_dms, (tuple, list)) and len(lon_dms) == 3:
            deg = self._as_float(lon_dms[0]) or 0.0
            min = self._as_float(lon_dms[1]) or 0.0
            sec = self._as_float(lon_dms[2]) or 0.0
            lon = deg + (min / 60.0) + (sec / 3600.0)
            if str(lon_ref).upper() == "W":
                lon = -lon
            gps["lon"] = lon

        # Extract altitude
        alt_raw = gps.get("GPSAltitude") or gps.get(6)
        if alt_raw is not None:
            alt = self._as_float(alt_raw)
            if alt is not None:
                gps["alt"] = alt
