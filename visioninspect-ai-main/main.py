"""
VisionInspect-AI: Master Command-Line Interface (CLI).
Provides commands for single-sample inspection, batch pipeline processing,
ground-truth benchmark evaluation, and latency/FPS benchmarking.
"""

import os
import sys
import glob
import time
import argparse
import json
from typing import List
import cv2
import numpy as np

from visioninspect.config import InspectionConfig
from visioninspect.pipeline import InspectionPipeline
from visioninspect.reporting.exporter import ResultExporter
from visioninspect.evaluation import BenchmarkEvaluator, EvaluationMetrics


def cmd_inspect(args: argparse.Namespace) -> int:
    """Inspects a single image and generates visual artifacts, telemetry, and diagnostic cards."""
    if not os.path.exists(args.image):
        print(f"[ERROR] Target image not found: {args.image}", file=sys.stderr)
        return 1

    config = InspectionConfig()
    if args.calibration:
        config.metrology.pixel_to_mm_ratio = args.calibration

    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
            config = InspectionConfig.from_dict(cfg_dict)

    print(f"\n" + "=" * 65)
    print(f"  VisionInspect-AI | Single Image Inspection")
    print(f"  Target: {args.image}")
    print(f"  Scale Calibration: {config.metrology.pixel_to_mm_ratio} mm/pixel")
    print("=" * 65)

    pipeline = InspectionPipeline(config)
    sample_name = os.path.splitext(os.path.basename(args.image))[0]

    result, images = pipeline.inspect(args.image, sample_name=sample_name)

    # Prepare output directory
    out_dir = args.output_dir or os.path.join("output", "inspections", sample_name)
    os.makedirs(out_dir, exist_ok=True)

    # Save visual outputs
    annotated_path = os.path.join(out_dir, f"{sample_name}_annotated.png")
    quad_path = os.path.join(out_dir, f"{sample_name}_diagnostic_panel.png")
    mask_path = os.path.join(out_dir, f"{sample_name}_defect_mask.png")
    heatmap_path = os.path.join(out_dir, f"{sample_name}_anomaly_heatmap.png")
    json_path = os.path.join(out_dir, f"{sample_name}_telemetry.json")

    cv2.imwrite(annotated_path, images["annotated"])
    cv2.imwrite(quad_path, images["quad_panel"])
    cv2.imwrite(mask_path, images["binary_mask"])
    cv2.imwrite(heatmap_path, images["heatmap"])
    ResultExporter.export_sample_json(result, json_path)

    # Print summary to terminal
    status_symbol = "[PASS]" if result.overall_status == "PASS" else "[FAIL]"
    print(f"\nInspection Verdict: {status_symbol} ({result.overall_status})")
    print(f"Execution Latency: {result.processing_time_ms:.2f} ms")
    print(f"Total Defects Detected: {result.total_defects}")
    print(f"Total Defect Area: {result.total_defect_area_mm2:.4f} mm²")
    print(f"Defect Severity Breakdown:")
    for sev, count in result.severity_breakdown.items():
        print(f"  - {sev}: {count}")

    if result.defects:
        print("\nDefect Detail Breakdown:")
        for d in result.defects:
            print(
                f"  * #{d.defect_id} [{d.defect_class.value}] Severity: {d.severity.value} | "
                f"Length: {d.length_mm:.2f}mm, Width: {d.width_mm:.2f}mm, Area: {d.area_mm2:.4f}mm² | "
                f"Confidence: {d.confidence*100:.1f}%"
            )

    print(f"\nArtifacts Saved Successfully:")
    print(f"  - Annotated Image:   {annotated_path}")
    print(f"  - Diagnostic Panel:  {quad_path}")
    print(f"  - Binary Mask:       {mask_path}")
    print(f"  - Anomaly Heatmap:   {heatmap_path}")
    print(f"  - Telemetry JSON:    {json_path}")
    print("=" * 65 + "\n")
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    """Executes high-throughput batch inspection across a directory of images."""
    if not os.path.isdir(args.input_dir):
        print(f"[ERROR] Input directory not found: {args.input_dir}", file=sys.stderr)
        return 1

    extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
    # Exclude mask subdirectory if inside input_dir
    image_paths = [p for p in sorted(image_paths) if "masks" not in p]

    if not image_paths:
        print(f"[WARNING] No image files found in {args.input_dir}")
        return 0

    config = InspectionConfig()
    if args.calibration:
        config.metrology.pixel_to_mm_ratio = args.calibration

    out_dir = args.output_dir or os.path.join("output", "batch_results")
    vis_dir = os.path.join(out_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    print(f"\n" + "=" * 65)
    print(f"  VisionInspect-AI | Batch Inspection Pipeline")
    print(f"  Input Directory: {args.input_dir} ({len(image_paths)} images)")
    print(f"  Output Directory: {out_dir}")
    print("=" * 65)

    pipeline = InspectionPipeline(config)
    results = []

    start_batch = time.perf_counter()
    for idx, path in enumerate(image_paths, 1):
        sample_name = os.path.splitext(os.path.basename(path))[0]
        res, imgs = pipeline.inspect(path, sample_name=sample_name)
        results.append(res)

        # Save annotated image and diagnostic panel
        annotated_file = os.path.join(vis_dir, f"{sample_name}_annotated.png")
        panel_file = os.path.join(vis_dir, f"{sample_name}_panel.png")
        cv2.imwrite(annotated_file, imgs["annotated"])
        cv2.imwrite(panel_file, imgs["quad_panel"])

        status_flag = "PASS" if res.overall_status == "PASS" else "FAIL"
        print(
            f"[{idx}/{len(image_paths)}] {sample_name:25s} -> {status_flag:4s} | "
            f"Defects: {res.total_defects:2d} | Latency: {res.processing_time_ms:6.1f}ms"
        )

    total_batch_time = (time.perf_counter() - start_batch) * 1000.0

    # Export audit reports
    csv_path = os.path.join(out_dir, "batch_inspection_summary.csv")
    md_path = os.path.join(out_dir, "batch_inspection_report.md")
    ResultExporter.export_batch_csv(results, csv_path)
    ResultExporter.export_batch_markdown_summary(results, md_path)

    # Print summary
    passed = sum(1 for r in results if r.overall_status == "PASS")
    yield_rate = (passed / len(results)) * 100.0 if results else 0.0

    print("\n" + "-" * 65)
    print(f"Batch Processing Completed in {total_batch_time:.1f} ms")
    print(f"Total Inspected: {len(results)} | Passed: {passed} | Failed: {len(results) - passed}")
    print(f"Factory Yield Rate: {yield_rate:.1f}%")
    print(f"Batch CSV Log:     {csv_path}")
    print(f"Audit Markdown:    {md_path}")
    print("=" * 65 + "\n")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluates predicted defect masks against ground truth masks and prints academic metrics."""
    if not os.path.isdir(args.images_dir):
        print(f"[ERROR] Images directory not found: {args.images_dir}", file=sys.stderr)
        return 1
    if not os.path.isdir(args.masks_dir):
        print(f"[ERROR] Ground truth masks directory not found: {args.masks_dir}", file=sys.stderr)
        return 1

    out_dir = args.output_dir or os.path.join("output", "evaluation_results")
    os.makedirs(out_dir, exist_ok=True)

    config = InspectionConfig()
    pipeline = InspectionPipeline(config)

    extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(args.images_dir, ext)))
    image_paths = [p for p in sorted(image_paths) if "masks" not in p]

    print(f"\n" + "=" * 65)
    print(f"  VisionInspect-AI | Ground Truth Benchmark Evaluation")
    print(f"  Evaluating {len(image_paths)} samples against {args.masks_dir}")
    print("=" * 65)

    eval_results: List[EvaluationMetrics] = []

    for path in image_paths:
        base_name = os.path.basename(path)
        sample_name = os.path.splitext(base_name)[0]
        mask_path = os.path.join(args.masks_dir, base_name)

        if not os.path.exists(mask_path):
            continue

        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            continue

        res, imgs = pipeline.inspect(path, sample_name=sample_name)
        pred_mask = imgs["binary_mask"]

        # If size differs, resize pred_mask to match gt_mask
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        metrics = BenchmarkEvaluator.evaluate_sample(pred_mask, gt_mask, sample_name=sample_name)
        eval_results.append(metrics)

        print(
            f"Sample: {sample_name:24s} | IoU: {metrics.pixel_iou:6.3f} | "
            f"Dice/F1: {metrics.dice_coefficient:6.3f} | Prec: {metrics.precision:6.3f} | Rec: {metrics.recall:6.3f}"
        )

    aggregate = BenchmarkEvaluator.aggregate_metrics(eval_results)

    # Save to JSON
    out_json = os.path.join(out_dir, "benchmark_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "aggregate_scores": aggregate,
            "individual_samples": [m.to_dict() for m in eval_results]
        }, f, indent=2)

    print("\n" + "-" * 65)
    print(f"Benchmark Aggregate Scores across {aggregate.get('num_samples', 0)} samples:")
    print(f"  - Mean Pixel IoU:          {aggregate.get('mean_pixel_iou', 0.0)*100:.2f}%")
    print(f"  - Mean Dice Coefficient:   {aggregate.get('mean_dice_f1', 0.0)*100:.2f}%")
    print(f"  - Mean Precision:          {aggregate.get('mean_precision', 0.0)*100:.2f}%")
    print(f"  - Mean Recall:             {aggregate.get('mean_recall', 0.0)*100:.2f}%")
    print(f"  - Mean Pixel Accuracy:     {aggregate.get('mean_accuracy', 0.0)*100:.2f}%")
    print(f"Evaluation Report Saved:     {out_json}")
    print("=" * 65 + "\n")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Measures pipeline inference latency, throughput (FPS), and memory footprint."""
    print(f"\n" + "=" * 65)
    print(f"  VisionInspect-AI | Performance & Latency Benchmark")
    print(f"  Iterations: {args.runs} runs | Resolution: {args.size}x{args.size}")
    print("=" * 65)

    # Synthesize realistic test surface
    rng = np.random.default_rng(1234)
    test_img = rng.integers(0, 256, (args.size, args.size, 3), dtype=np.uint8)

    config = InspectionConfig()
    pipeline = InspectionPipeline(config)

    # Warmup runs
    for _ in range(3):
        pipeline.inspect(test_img, sample_name="warmup")

    latencies = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        pipeline.inspect(test_img, sample_name=f"bench_{i}")
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))
    min_lat = float(np.min(latencies))
    max_lat = float(np.max(latencies))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    print(f"\nBenchmark Results:")
    print(f"  - Mean Latency:  {mean_lat:.2f} ms (+/- {std_lat:.2f} ms)")
    print(f"  - Min / Max:     {min_lat:.2f} ms / {max_lat:.2f} ms")
    print(f"  - Throughput:    {fps:.1f} Frames Per Second (FPS)")
    print(f"  - Real-time:     {'YES (>=30 FPS)' if fps >= 30 else 'Near-real-time high-resolution inspection'}")
    print("=" * 65 + "\n")
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="visioninspect",
        description="VisionInspect-AI: Industrial Surface Quality Assessment & Defect Segmentation CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. Inspect
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a single surface image")
    inspect_parser.add_argument("--image", required=True, help="Path to target image file")
    inspect_parser.add_argument("--output-dir", help="Directory to save inspection outputs")
    inspect_parser.add_argument("--calibration", type=float, help="Pixel to millimeter scale ratio")
    inspect_parser.add_argument("--config", help="Path to JSON configuration file")

    # 2. Batch
    batch_parser = subparsers.add_parser("batch", help="Batch inspect an entire directory of images")
    batch_parser.add_argument("--input-dir", required=True, help="Directory containing target images")
    batch_parser.add_argument("--output-dir", help="Directory to save batch audit outputs")
    batch_parser.add_argument("--calibration", type=float, help="Pixel to millimeter scale ratio")

    # 3. Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate segmentation masks against ground truth")
    eval_parser.add_argument("--images-dir", required=True, help="Directory containing test images")
    eval_parser.add_argument("--masks-dir", required=True, help="Directory containing ground truth masks")
    eval_parser.add_argument("--output-dir", help="Directory to save benchmark scores")

    # 4. Benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Measure latency and throughput (FPS)")
    bench_parser.add_argument("--runs", type=int, default=30, help="Number of benchmark iterations")
    bench_parser.add_argument("--size", type=int, default=512, help="Image resolution edge length")

    args = parser.parse_args()

    if args.command == "inspect":
        sys.exit(cmd_inspect(args))
    elif args.command == "batch":
        sys.exit(cmd_batch(args))
    elif args.command == "evaluate":
        sys.exit(cmd_evaluate(args))
    elif args.command == "benchmark":
        sys.exit(cmd_benchmark(args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
