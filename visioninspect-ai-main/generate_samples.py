"""
Synthetic Industrial Sample Generator for VisionInspect-AI.
Generates realistic industrial surface textures (steel, PCB, ceramic, fabric)
with controlled defect anomalies and ground truth annotation masks for benchmark validation.
"""

import os
import cv2
import numpy as np


def generate_brushed_metal_texture(h: int = 512, w: int = 512, seed: int = 42) -> np.ndarray:
    """Generates brushed stainless steel surface with directional grain and subtle lighting gradient."""
    rng = np.random.default_rng(seed)
    # High-frequency horizontal noise
    base = rng.normal(160, 15, (h, w)).astype(np.float32)
    # Directional horizontal blur to simulate brushed grain
    kernel = np.zeros((1, 31), dtype=np.float32)
    kernel[0, :] = 1.0 / 31.0
    brushed = cv2.filter2D(base, -1, kernel)

    # Add gentle factory lighting gradient
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)
    vignette = 1.0 - 0.15 * (xx**2 + yy**2)
    metal = np.clip(brushed * vignette, 0, 255).astype(np.uint8)
    return cv2.cvtColor(metal, cv2.COLOR_GRAY2BGR)


def generate_ceramic_texture(h: int = 512, w: int = 512, seed: int = 101) -> np.ndarray:
    """Generates smooth glazed ceramic tile with subtle mottling."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(210, 8, (h // 4, w // 4)).astype(np.float32)
    smooth = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    smooth = cv2.GaussianBlur(smooth, (9, 9), 0)
    ceramic = np.clip(smooth, 0, 255).astype(np.uint8)
    # Add subtle warm ivory tint
    b = (ceramic * 0.95).astype(np.uint8)
    g = (ceramic * 0.98).astype(np.uint8)
    r = ceramic
    return cv2.merge([b, g, r])


def generate_pcb_texture(h: int = 512, w: int = 512, seed: int = 202) -> np.ndarray:
    """Generates green solder mask PCB with copper circuit traces."""
    # Dark green base
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (20, 95, 30)  # FR-4 green

    # Add copper tracks (gold-orange color: BGR 25, 170, 220)
    trace_color = (30, 175, 225)
    for y in range(40, h, 60):
        cv2.line(img, (20, y), (w - 20, y), trace_color, 8, cv2.LINE_AA)
        cv2.circle(img, (80, y), 12, trace_color, -1, cv2.LINE_AA)
        cv2.circle(img, (w - 80, y), 12, trace_color, -1, cv2.LINE_AA)
        # Via holes
        cv2.circle(img, (80, y), 5, (10, 30, 15), -1, cv2.LINE_AA)
        cv2.circle(img, (w - 80, y), 5, (10, 30, 15), -1, cv2.LINE_AA)

    # Vertical interconnect tracks
    for x in (180, 340):
        cv2.line(img, (x, 40), (x, h - 40), trace_color, 6, cv2.LINE_AA)

    return img


def generate_fabric_texture(h: int = 512, w: int = 512, seed: int = 303) -> np.ndarray:
    """Generates woven textile / fabric canvas texture."""
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    weave = (np.sin(x * 0.4) * np.cos(y * 0.4) * 20.0 + 170.0).astype(np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 5, (h, w))
    fabric_gray = np.clip(weave + noise, 0, 255).astype(np.uint8)
    # Slate denim blue tint
    b = np.clip(fabric_gray * 1.15, 0, 255).astype(np.uint8)
    g = np.clip(fabric_gray * 0.95, 0, 255).astype(np.uint8)
    r = np.clip(fabric_gray * 0.85, 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])


def create_all_samples(output_dir: str = "samples") -> None:
    """Creates complete set of realistic test samples with ground truth masks."""
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    h, w = 512, 512

    # 1. Steel Plate with Stress Crack (CRITICAL)
    steel = generate_brushed_metal_texture(h, w, seed=42)
    mask_crack = np.zeros((h, w), dtype=np.uint8)
    # Draw winding, branching crack
    crack_pts = [
        (120, 200), (150, 215), (180, 210), (220, 240),
        (260, 235), (310, 270), (350, 280), (390, 310)
    ]
    for i in range(len(crack_pts) - 1):
        cv2.line(mask_crack, crack_pts[i], crack_pts[i + 1], 255, 4, cv2.LINE_AA)
        cv2.line(steel, crack_pts[i], crack_pts[i + 1], (30, 30, 35), 4, cv2.LINE_AA)
    # Branch
    branch_pts = [(260, 235), (285, 210), (320, 195)]
    for i in range(len(branch_pts) - 1):
        cv2.line(mask_crack, branch_pts[i], branch_pts[i + 1], 255, 3, cv2.LINE_AA)
        cv2.line(steel, branch_pts[i], branch_pts[i + 1], (30, 30, 35), 3, cv2.LINE_AA)

    cv2.imwrite(os.path.join(output_dir, "steel_plate_crack.png"), steel)
    cv2.imwrite(os.path.join(masks_dir, "steel_plate_crack.png"), mask_crack)

    # 2. PCB with Abrasive Scratch across trace (MAJOR)
    pcb = generate_pcb_texture(h, w, seed=202)
    mask_scratch = np.zeros((h, w), dtype=np.uint8)
    scratch_start = (90, 80)
    scratch_end = (410, 360)
    cv2.line(mask_scratch, scratch_start, scratch_end, 255, 5, cv2.LINE_AA)
    cv2.line(pcb, scratch_start, scratch_end, (190, 200, 210), 5, cv2.LINE_AA)

    cv2.imwrite(os.path.join(output_dir, "pcb_trace_scratch.png"), pcb)
    cv2.imwrite(os.path.join(masks_dir, "pcb_trace_scratch.png"), mask_scratch)

    # 3. Ceramic Tile with Pitting / Voids (MINOR)
    ceramic = generate_ceramic_texture(h, w, seed=101)
    mask_pit = np.zeros((h, w), dtype=np.uint8)
    pits = [(180, 220, 10), (320, 340, 8), (220, 360, 7)]
    for px, py, r in pits:
        cv2.circle(mask_pit, (px, py), r, 255, -1, cv2.LINE_AA)
        # Draw dark pit with slight shadow rim
        cv2.circle(ceramic, (px, py), r, (40, 45, 50), -1, cv2.LINE_AA)
        cv2.circle(ceramic, (px, py), r + 2, (120, 125, 130), 1, cv2.LINE_AA)

    cv2.imwrite(os.path.join(output_dir, "ceramic_tile_pit.png"), ceramic)
    cv2.imwrite(os.path.join(masks_dir, "ceramic_tile_pit.png"), mask_pit)

    # 4. Fabric with Surface Blemish / Oil Discoloration (MINOR)
    fabric = generate_fabric_texture(h, w, seed=303)
    mask_blemish = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask_blemish, (256, 256), (60, 40), 25, 0, 360, 255, -1, cv2.LINE_AA)
    # Apply warm brownish/darker tint to the blemish area
    blemish_overlay = fabric.copy()
    cv2.ellipse(blemish_overlay, (256, 256), (60, 40), 25, 0, 360, (20, 60, 90), -1, cv2.LINE_AA)
    fabric = cv2.addWeighted(fabric, 0.55, blemish_overlay, 0.45, 0)

    cv2.imwrite(os.path.join(output_dir, "fabric_weave_blemish.png"), fabric)
    cv2.imwrite(os.path.join(masks_dir, "fabric_weave_blemish.png"), mask_blemish)

    # 5. Metal with High-Reflectance Particulate Inclusion (MAJOR)
    metal_part = generate_brushed_metal_texture(h, w, seed=505)
    mask_part = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_part, (240, 190), 12, 255, -1, cv2.LINE_AA)
    cv2.circle(metal_part, (240, 190), 12, (255, 255, 255), -1, cv2.LINE_AA)

    cv2.imwrite(os.path.join(output_dir, "metal_chip_inclusion.png"), metal_part)
    cv2.imwrite(os.path.join(masks_dir, "metal_chip_inclusion.png"), mask_part)

    # 6. Pristine Clean Reference Surface (PASS)
    clean = generate_brushed_metal_texture(h, w, seed=999)
    mask_clean = np.zeros((h, w), dtype=np.uint8)

    cv2.imwrite(os.path.join(output_dir, "clean_reference_pass.png"), clean)
    cv2.imwrite(os.path.join(masks_dir, "clean_reference_pass.png"), mask_clean)

    print(f"[OK] Generated 6 benchmark samples and ground-truth masks in '{output_dir}/'")


if __name__ == "__main__":
    create_all_samples()
