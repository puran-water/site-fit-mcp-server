"""Export utilities for solutions."""

from .geojson import (
    placements_to_geojson,
    road_network_to_geojson,
    solution_to_geojson,
)
from .pack import (
    ExportPackResult,
    export_pack,
)
from .quantities import (
    QuantityTakeoff,
    compute_quantities,
)
from .svg import (
    export_comparison_svg,
    export_solution_to_svg,
)

# DXF export is optional (requires ezdxf)
try:
    from .dxf import save_solution_to_dxf, solution_to_dxf
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
    from .pdf_report import PDFReportConfig, generate_pdf_report
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
