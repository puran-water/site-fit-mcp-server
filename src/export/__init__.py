"""Export utilities for solutions."""

from .geojson import (
    solution_to_geojson,
    placements_to_geojson,
    road_network_to_geojson,
)
from .svg import (
    export_solution_to_svg,
    export_comparison_svg,
)
from .quantities import (
    compute_quantities,
    QuantityTakeoff,
)
from .pack import (
    export_pack,
    ExportPackResult,
)

# DXF export is optional (requires ezdxf)
try:
    from .dxf import solution_to_dxf, save_solution_to_dxf
    _HAS_DXF = True
except ImportError:
    _HAS_DXF = False

    def solution_to_dxf(*args, **kwargs):
        raise ImportError(
            "ezdxf is required for DXF export. "
            "Install with: pip install site-fit-mcp-server[dxf]"
        )

    def save_solution_to_dxf(*args, **kwargs):
        raise ImportError(
            "ezdxf is required for DXF export. "
            "Install with: pip install site-fit-mcp-server[dxf]"
        )

# PDF export is optional (requires weasyprint)
try:
    from .pdf_report import generate_pdf_report, PDFReportConfig
    _HAS_PDF = True
except ImportError:
    _HAS_PDF = False

    def generate_pdf_report(*args, **kwargs):
        raise ImportError(
            "weasyprint is required for PDF export. "
            "Install with: pip install site-fit-mcp-server[pdf]"
        )

    PDFReportConfig = None  # type: ignore

__all__ = [
    "solution_to_geojson",
    "placements_to_geojson",
    "road_network_to_geojson",
    "export_solution_to_svg",
    "export_comparison_svg",
    "solution_to_dxf",
    "save_solution_to_dxf",
    "compute_quantities",
    "QuantityTakeoff",
    "export_pack",
    "ExportPackResult",
    "generate_pdf_report",
    "PDFReportConfig",
]
