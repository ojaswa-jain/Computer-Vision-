"""
Integration tests for CLI execution and end-to-end inspection pipeline.
"""

import os
import subprocess
import pytest
import numpy as np
import cv2


@pytest.fixture(scope="session")
def test_workspace(tmp_path_factory):
    """Creates a temporary workspace with synthetic images and ground truth masks."""
    base_dir = tmp_path_factory.mktemp("vision_workspace")
    images_dir = base_dir / "samples"
    masks_dir = images_dir / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    # Image with defect
    img = np.full((256, 256, 3), 170, dtype=np.uint8)
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.line(img, (50, 50), (200, 200), (20, 20, 20), 5)
    cv2.line(mask, (50, 50), (200, 200), 255, 5)

    img_path = str(images_dir / "test_scratch.png")
    mask_path = str(masks_dir / "test_scratch.png")
    cv2.imwrite(img_path, img)
    cv2.imwrite(mask_path, mask)

    return {
        "base": str(base_dir),
        "images": str(images_dir),
        "masks": str(masks_dir),
        "sample": img_path
    }


def test_cli_inspect(test_workspace):
    out_dir = os.path.join(test_workspace["base"], "out_inspect")
    cmd = [
        "python3", "main.py", "inspect",
        "--image", test_workspace["sample"],
        "--output-dir", out_dir
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Inspect command failed: {res.stderr}"
    assert "Inspection Verdict:" in res.stdout
    assert os.path.exists(os.path.join(out_dir, "test_scratch_annotated.png"))
    assert os.path.exists(os.path.join(out_dir, "test_scratch_telemetry.json"))


def test_cli_batch(test_workspace):
    out_dir = os.path.join(test_workspace["base"], "out_batch")
    cmd = [
        "python3", "main.py", "batch",
        "--input-dir", test_workspace["images"],
        "--output-dir", out_dir
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Batch command failed: {res.stderr}"
    assert "Batch Processing Completed" in res.stdout
    assert os.path.exists(os.path.join(out_dir, "batch_inspection_summary.csv"))
    assert os.path.exists(os.path.join(out_dir, "batch_inspection_report.md"))


def test_cli_evaluate(test_workspace):
    out_dir = os.path.join(test_workspace["base"], "out_eval")
    cmd = [
        "python3", "main.py", "evaluate",
        "--images-dir", test_workspace["images"],
        "--masks-dir", test_workspace["masks"],
        "--output-dir", out_dir
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Evaluate command failed: {res.stderr}"
    assert "Mean Pixel IoU:" in res.stdout
    assert os.path.exists(os.path.join(out_dir, "benchmark_metrics.json"))


def test_cli_benchmark():
    cmd = ["python3", "main.py", "benchmark", "--runs", "5", "--size", "128"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Benchmark command failed: {res.stderr}"
    assert "Benchmark Results:" in res.stdout
    assert "Throughput:" in res.stdout
