"""
DJI-specific EXIF extractor

Handles DJI drone cameras with enhanced XMP metadata parsing.

Supported models (verified from actual images):
- M3E (Mavic 3 Enterprise) - 4/3" sensor, 12.29mm focal
- M3T (Mavic 3 Thermal) - 1/2" sensor, 4.4mm focal
- M4E (Mavic 4 Enterprise) - 4/3" sensor, 12.29mm focal
- FC2403 (Mavic 2 Enterprise / Mini) - 1/2" sensor, 4.5mm focal
- FC7503 (Air series) - 1/2" sensor, 4.49mm focal
- EP800 (Enterprise drone) - 1" sensor, 8.8mm focal
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .base import ExifExtractor


class DJIExtractor(ExifExtractor):
    """
    DJI-specific EXIF enhancements.

    All DJI cameras already have good EXIF extraction through the generic
    extractor + XMP fallbacks. This class provides:
    1. Validation of extracted values
    2. Model-specific sensor size mapping
    3. Future enhancements as needed
    """

    # Known DJI camera sensor sizes (width, height) in mm
    # Verified from actual images in training pipeline
    SENSOR_DATABASE = {
        "M3E": (17.3, 13.0),       # Mavic 3 Enterprise - 4/3" sensor
        "M3T": (9.6, 7.2),         # Mavic 3 Thermal - 1/2" sensor (wide camera)
        "M4E": (17.3, 13.0),       # Mavic 4 Enterprise - 4/3" sensor
        "FC2403": (9.6, 7.2),      # Mavic 2 Enterprise / Mini - 1/2" sensor
        "FC7503": (9.6, 7.2),      # Air series - 1/2" sensor
        "EP800": (13.2, 8.8),      # Enterprise drone - 1" sensor
        "FC6310": (17.3, 13.0),    # Mavic 3 (older naming) - 4/3" sensor
        "FC3170": (13.2, 8.8),     # Mavic 2 Pro - 1" sensor
        "FC6510": (9.6, 7.2),      # Air 3 - 1/2" sensor
        "FC330": (13.2, 8.8),      # Phantom 3 - 1" sensor
        "FC350": (13.2, 8.8),      # Phantom 4 - 1" sensor
    }

    def extract(self, image_path: Path) -> tuple[Dict[str, Any], int, int]:
        """
        DJI cameras work well with generic extraction.
        This is here for API consistency.
        """
        from .generic import GenericExtractor

        generic = GenericExtractor()
        return generic.extract(image_path)

    def enhance(self, exif: Dict[str, Any], image_path: Path) -> Dict[str, Any]:
        """
        Enhance DJI EXIF data with model-specific information.

        Currently adds:
        - SensorWidth/SensorHeight based on camera model
        - Validation of critical fields

        Args:
            exif: Existing EXIF dictionary from generic extraction
            image_path: Path to image file

        Returns:
            Enhanced EXIF dictionary with DJI-specific fields
        """
        model = str(exif.get("Model", "")).strip()

        # Add sensor dimensions if known
        if model in self.SENSOR_DATABASE and "SensorWidth" not in exif:
            sensor_width, sensor_height = self.SENSOR_DATABASE[model]
            exif["SensorWidth"] = sensor_width
            exif["SensorHeight"] = sensor_height
            exif["SensorWidthSource"] = "DJI_DATABASE"

        # Validate critical DJI fields are present
        self._validate_dji_fields(exif, model)

        return exif

    @staticmethod
    def _validate_dji_fields(exif: Dict[str, Any], model: str) -> None:
        """
        Validate that critical DJI fields are present.

        Logs warnings if expected fields are missing.
        """
        import logging

        logger = logging.getLogger(__name__)

        # All DJI drones should have these fields
        expected_fields = {
            "FocalLength": "focal length",
            "ExposureTime": "exposure time",
            "ISOSpeedRatings": "ISO",
            "GPSInfo": "GPS data",
            "RelativeAltitude": "relative altitude",
            "Pitch": "camera pitch",
        }

        missing = []
        for field, description in expected_fields.items():
            if field == "GPSInfo":
                if not isinstance(exif.get(field), dict):
                    missing.append(description)
            elif field == "Pitch":
                # Pitch can be in multiple fields
                if (
                    exif.get("Pitch") is None
                    and exif.get("CameraElevationAngle") is None
                    and exif.get("GimbalPitchDegree") is None
                ):
                    missing.append(description)
            else:
                if exif.get(field) is None:
                    missing.append(description)

        if missing:
            logger.warning(
                f"DJI {model}: Missing expected fields: {', '.join(missing)}. "
                f"This may indicate incomplete XMP metadata or unsupported camera variant."
            )

    @classmethod
    def get_sensor_dimensions(cls, model: str) -> tuple[float, float] | None:
        """
        Get sensor dimensions for a DJI camera model.

        Args:
            model: Camera model string (e.g., "M3E", "FC7503")

        Returns:
            Tuple of (width, height) in mm, or None if unknown
        """
        return cls.SENSOR_DATABASE.get(model)
