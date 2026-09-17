"""
Audit telemetry export to JSON, CSV, and formatted Markdown summaries.
"""

import os
import json
import csv
from typing import List, Dict, Any

from visioninspect.metrology.measurement import SampleInspectionResult


class ResultExporter:
    """Exports inspection telemetry across standard machine-readable and human-readable formats."""

    @staticmethod
    def export_sample_json(result: SampleInspectionResult, filepath: str) -> None:
        """Saves detailed single-sample telemetry to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

    @staticmethod
    def export_batch_csv(results: List[SampleInspectionResult], filepath: str) -> None:
        """Exports batch inspection table to CSV for spreadsheet/database ingestion."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        fieldnames = [
            "sample_name",
            "overall_status",
            "total_defects",
            "critical_count",
            "major_count",
            "minor_count",
            "pass_count",
            "total_defect_area_mm2",
            "processing_time_ms",
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "sample_name": r.sample_name,
                    "overall_status": r.overall_status,
                    "total_defects": r.total_defects,
                    "critical_count": r.severity_breakdown.get("CRITICAL", 0),
                    "major_count": r.severity_breakdown.get("MAJOR", 0),
                    "minor_count": r.severity_breakdown.get("MINOR", 0),
                    "pass_count": r.severity_breakdown.get("PASS", 0),
                    "total_defect_area_mm2": round(r.total_defect_area_mm2, 4),
                    "processing_time_ms": round(r.processing_time_ms, 2),
                })

    @staticmethod
    def export_batch_markdown_summary(results: List[SampleInspectionResult], filepath: str) -> None:
        """Generates an executive Markdown audit log."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        total_samples = len(results)
        passed_samples = sum(1 for r in results if r.overall_status == "PASS")
        failed_samples = total_samples - passed_samples
        yield_rate = (passed_samples / total_samples * 100.0) if total_samples > 0 else 0.0
        avg_latency = (sum(r.processing_time_ms for r in results) / total_samples) if total_samples > 0 else 0.0

        total_defects = sum(r.total_defects for r in results)
        total_crit = sum(r.severity_breakdown.get("CRITICAL", 0) for r in results)
        total_maj = sum(r.severity_breakdown.get("MAJOR", 0) for r in results)
        total_min = sum(r.severity_breakdown.get("MINOR", 0) for r in results)

        lines = [
            "# VisionInspect-AI: Batch Quality Audit Summary",
            "",
            "## 1. Executive Telemetry Overview",
            f"- **Total Units Inspected**: {total_samples}",
            f"- **Yield Rate**: `{yield_rate:.2f}%` ({passed_samples} Passed, {failed_samples} Failed)",
            f"- **Average Processing Latency**: `{avg_latency:.2f} ms`",
            f"- **Cumulative Defects Isolated**: {total_defects}",
            f"  - Critical: `{total_crit}`",
            f"  - Major: `{total_maj}`",
            f"  - Minor: `{total_min}`",
            "",
            "## 2. Sample-by-Sample Inspection Log",
            "",
            "| Sample Name | Status | Defects | Critical | Major | Minor | Defect Area (mm²) | Latency (ms) |",
            "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]

        for r in results:
            status_badge = "**PASS**" if r.overall_status == "PASS" else "**FAIL**"
            lines.append(
                f"| `{r.sample_name}` | {status_badge} | {r.total_defects} | "
                f"{r.severity_breakdown.get('CRITICAL', 0)} | "
                f"{r.severity_breakdown.get('MAJOR', 0)} | "
                f"{r.severity_breakdown.get('MINOR', 0)} | "
                f"{r.total_defect_area_mm2:.3f} | {r.processing_time_ms:.1f} |"
            )

        lines.append("")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
