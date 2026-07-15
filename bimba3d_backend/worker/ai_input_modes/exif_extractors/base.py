"""
Base class for EXIF extractors
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Tuple


class BaseExtractor(ABC):
    """
    Base class for all EXIF extractors.

    Subclasses implement extraction logic for specific camera manufacturers.
    """

    @abstractmethod
    def extract(self, image_path: Path) -> Tuple[Dict[str, Any], int, int]:
        """
        Extract EXIF data from image.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (exif_dict, width, height)
            - exif_dict: Dictionary with all EXIF fields
            - width: Image width in pixels
            - height: Image height in pixels
        """
        pass

    def get_name(self) -> str:
        """Get extractor name for logging"""
        return self.__class__.__name__
