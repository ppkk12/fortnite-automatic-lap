"""Compare upstream template matching with the recognition PR's detectors.

Run from the repository root with:
    python tools/benchmark_recognition.py --iterations 30

Fixtures are deterministic composites of the repository's button/reward assets;
results are synthetic benchmarks, not a claim of live-game accuracy.
"""

import argparse
import json
import os
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config
from config import Settings
from reward_vision import find_pirate_token, find_reward_banner, load_reward_banner_templates
from vision import find_plate, find_template, load_template, load_template_auto


BUTTON_CASES = (
    ("play", "play_btn.png", "play_btn_en.png", (40, 530)),
    ("return_lobby", "return_lobby_btn.png", "return_lobby_btn_en.png", (720, 230)),
    ("claim", "claim_btn.png", "claim_btn_en.png", (800, 620)),
    ("claim_single", "claim_single_btn.png", "claim_single_btn_en.png", (800, 620)),
)
NEGATIVE_BUTTONS = ("continue_btn.png", "skip_btn.png", "menu_btn.png", "hud_btn.png")


def _asset(name):
    path = os.path.join(config.assets_dir(), name)
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def _frame_with(image=None, pos=(0, 0), seed=0, size=(720, 1280)):
    h, w = size
    y = np.linspace(20, 48, h, dtype=np.uint8)[:, None, None]
    frame = np.broadcast_to(y, (h, w, 3)).copy()
    noise = np.random.default_rng(seed).integers(0, 5, size=frame.shape, dtype=np.uint8)
    frame = cv2.add(frame, noise)
    if image is not None:
        ih, iw = image.shape[:2]
        x, top = pos
        frame[top:top + ih, x:x + iw] = image
    return frame


def _button_runs():
    legacy = []
    current = []
    negatives = []
    for key, ja_name, en_name, pos in BUTTON_CASES:
        for lang, source_name in (("ja", ja_name), ("en", en_name)):
            button = _asset(source_name)
            frame = _frame_with(button, pos, seed=len(legacy) + 11)
            old_tpl = load_template(os.path.join(config.assets_dir(), source_name))
            new_tpl = load_template_auto(config.template_path(key, lang), config.PLATES[key])
            legacy.append((frame, old_tpl, True))
            current.append((frame, new_tpl, True))
            for index, distractor_name in enumerate(NEGATIVE_BUTTONS):
                distractor = _asset(distractor_name)
                negatives.append((
                    key,
                    lang,
                    distractor_name,
                    _frame_with(distractor, pos, seed=index + 30 + len(negatives)),
                    old_tpl,
                    new_tpl,
                ))
    return legacy, current, negatives


def _cross_language_button_runs():
    cases = []
    for key, ja_name, en_name, pos in BUTTON_CASES:
        if key == "claim":
            continue
        for source_lang, source_name, wrong_lang, wrong_name in (
            ("ja", ja_name, "en", en_name),
            ("en", en_name, "ja", ja_name),
        ):
            frame = _frame_with(_asset(source_name), pos, seed=170 + len(cases))
            old_template = load_template(os.path.join(config.assets_dir(), wrong_name))
            new_template = load_template_auto(config.template_path(key, source_lang), config.PLATES[key])
            cases.append((frame, old_template, new_template))
    return cases


def _button_detect_old(frame, template):
    return find_template(frame, template, Settings().threshold, Settings().scales)[0] is not None


def _button_detect_new(frame, template):
    if template is None:
        return False
    if hasattr(template, "roi"):
        return find_plate(frame, template, Settings().scales)[0] is not None
    return find_template(frame, template, Settings().threshold, Settings().scales)[0] is not None


def _legacy_reward_visible(frame, templates):
    bounds = ((0.16, 0.18, 0.40, 0.52), (0.60, 0.18, 0.85, 0.52))
    h, w = frame.shape[:2]
    edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 70, 150)
    matches = []
    for template, box in zip(templates, bounds):
        x0, y0, x1, y1 = (int(q * n) for q, n in zip(box, (w, h, w, h)))
        region = edges[y0:y1, x0:x1]
        scaled = cv2.resize(template, (round(template.shape[1] * w / 1280), round(template.shape[0] * h / 720)), interpolation=cv2.INTER_NEAREST)
        if scaled.shape[0] >= region.shape[0] or scaled.shape[1] >= region.shape[1]:
            return False
        result = cv2.matchTemplate(region, scaled, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(result)
        matches.append((score, y0 + loc[1] + scaled.shape[0] / 2))
    return all(x[0] >= 0.48 for x in matches) and abs(matches[0][1] - matches[1][1]) <= h * 0.10


def _reward_frame(templates, width, height, variation, y_ratio, include_right=True, seed=0):
    frame = _frame_with(seed=seed, size=(height, width))
    for index, template in enumerate(templates):
        if index == 1 and not include_right:
            continue
        sx, sy = width / 1280 * variation, height / 720 * variation
        patch = cv2.resize(template, (round(template.shape[1] * sx), round(template.shape[0] * sy)), interpolation=cv2.INTER_NEAREST)
        cx = round(width * (0.28 if index == 0 else 0.72))
        cy = round(height * y_ratio)
        x, y = cx - patch.shape[1] // 2, cy - patch.shape[0] // 2
        frame[y:y + patch.shape[0], x:x + patch.shape[1]] = patch[:, :, None]
    return frame


def _timing_ms(function, cases, iterations):
    for case in cases:
        function(*case)
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        for case in cases:
            for _ in range(iterations):
                function(*case)
        samples.append((time.perf_counter() - start) * 1000 / (len(cases) * iterations))
    return round(statistics.median(samples), 3)


def _rate(detector, cases):
    positives = [(frame, template, _) for frame, template, _ in cases]
    true_positives = sum(detector(frame, template) for frame, template, _ in positives)
    return true_positives, len(positives)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=4)
    args = parser.parse_args()
    cv2.setNumThreads(1)

    old_buttons, new_buttons, negative_frames = _button_runs()
    old_tp, total = _rate(_button_detect_old, old_buttons)
    new_tp, _ = _rate(_button_detect_new, new_buttons)
    cross_language_cases = _cross_language_button_runs()
    cross_old_tp = sum(_button_detect_old(frame, old_tpl) for frame, old_tpl, _ in cross_language_cases)
    cross_new_tp = sum(_button_detect_new(frame, new_tpl) for frame, _, new_tpl in cross_language_cases)
    old_neg = [(frame, old_tpl) for _, _, _, frame, old_tpl, _ in negative_frames]
    new_neg = [(frame, new_tpl) for _, _, _, frame, _, new_tpl in negative_frames]
    old_fp = sum(_button_detect_old(frame, tpl) for frame, tpl in old_neg)
    new_fp = sum(_button_detect_new(frame, tpl) for frame, tpl in new_neg)
    new_fp_labels = [
        f"{key}/{lang} vs {distractor}"
        for key, lang, distractor, frame, _, template in negative_frames
        if _button_detect_new(frame, template)
    ]
    old_ms = _timing_ms(_button_detect_old, [(f, t) for f, t, _ in old_buttons], args.iterations)
    new_ms = _timing_ms(_button_detect_new, [(f, t) for f, t, _ in new_buttons], args.iterations)

    icon_path = config.template_path("pirate_token", "ja")
    icon_full = cv2.imread(icon_path, cv2.IMREAD_COLOR)
    icon_plate = load_template_auto(icon_path, config.PLATES["pirate_token"])
    icon_cases = []
    icon_source = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
    for index, scale in enumerate((0.35, 0.45, 0.6, 0.75, 0.85)):
        size = round(256 * scale)
        rgba = cv2.resize(icon_source, (size, size), interpolation=cv2.INTER_AREA)
        frame = _frame_with(seed=index + 70)
        x, y = 640 - size // 2, 150 - size // 2
        region = frame[y:y + size, x:x + size]
        mask = rgba[:, :, 3] > 0
        region[mask] = rgba[:, :, :3][mask]
        icon_cases.append((frame, icon_full, icon_plate, True))
    icon_negatives = [_frame_with(seed=100 + index) for index in range(3)]
    reward_decoy = _reward_frame(load_reward_banner_templates(config.assets_dir(), config.bundled_assets_dir()), 1280, 720, 1.0, 0.35)
    icon_negatives.append(reward_decoy)
    off_region = _frame_with(seed=105)
    off_size = 192
    off_icon = cv2.resize(icon_source, (off_size, off_size), interpolation=cv2.INTER_AREA)
    off_x, off_y = 640 - off_size // 2, 420
    off_roi = off_region[off_y:off_y + off_size, off_x:off_x + off_size]
    off_mask = off_icon[:, :, 3] > 0
    off_roi[off_mask] = off_icon[:, :, :3][off_mask]
    icon_negatives.append(off_region)
    icon_old = sum(find_template(f, t, Settings().threshold, Settings().scales)[0] is not None for f, t, _, _ in icon_cases)
    icon_new = sum(find_pirate_token(f, p)[0] is not None for f, _, p, _ in icon_cases)
    icon_old_fp = sum(find_template(f, icon_full, Settings().threshold, Settings().scales)[0] is not None for f in icon_negatives)
    icon_new_fp = sum(find_pirate_token(f, icon_plate)[0] is not None for f in icon_negatives)
    icon_old_ms = _timing_ms(lambda f, t, p: find_template(f, t, Settings().threshold, Settings().scales), [(f, t, p) for f, t, p, _ in icon_cases], args.iterations)
    icon_new_ms = _timing_ms(lambda f, t, p: find_pirate_token(f, p), [(f, t, p) for f, t, p, _ in icon_cases], args.iterations)

    reward_templates = load_reward_banner_templates(config.assets_dir(), config.bundled_assets_dir())
    reward_cases = []
    for i, (w, h, scale) in enumerate(((1280, 720, 0.85), (1280, 720, 1.0), (1280, 720, 1.15), (1920, 1080, 0.925), (1920, 1080, 1.075))):
        reward_cases.append((_reward_frame(reward_templates, w, h, scale, 0.34 + i * 0.01, seed=i), reward_templates))
    reward_negatives = [
        (_reward_frame(reward_templates, 1280, 720, 1.0, 0.35, include_right=False, seed=20), reward_templates),
        (_reward_frame(reward_templates, 1280, 720, 1.0, 0.25, include_right=False, seed=21), reward_templates),
        (_frame_with(seed=22), reward_templates),
    ]
    old_reward_tp = sum(_legacy_reward_visible(f, t) for f, t in reward_cases)
    new_reward_tp = sum(find_reward_banner(f, t).found for f, t in reward_cases)
    old_reward_fp = sum(_legacy_reward_visible(f, t) for f, t in reward_negatives)
    new_reward_fp = sum(find_reward_banner(f, t).found for f, t in reward_negatives)
    old_reward_ms = _timing_ms(_legacy_reward_visible, reward_cases, args.iterations)
    new_reward_ms = _timing_ms(lambda f, t: find_reward_banner(f, t).found, reward_cases, args.iterations)

    print(json.dumps({
        "fixture_note": "synthetic composites from shipped assets; not live-game footage",
        "reward_baseline_note": "previous single-scale helper from the personal fork; upstream main has no reward-banner detector",
        "iterations_per_case_per_round": args.iterations,
        "button_detection": {
            "upstream_exact_template": {"true_positives": f"{old_tp}/{total}", "false_positives": f"{old_fp}/{len(old_neg)}", "median_ms_per_frame": old_ms},
            "pr_masked_plate": {"true_positives": f"{new_tp}/{total}", "false_positives": f"{new_fp}/{len(new_neg)}", "false_positive_cases": new_fp_labels, "median_ms_per_frame": new_ms},
            "speedup": round(old_ms / new_ms, 2) if new_ms else None,
        },
        "cross_language_buttons": {
            "upstream_language_specific_template": f"{cross_old_tp}/{len(cross_language_cases)}",
            "pr_language_neutral_plate": f"{cross_new_tp}/{len(cross_language_cases)}",
        },
        "pirate_icon": {
            "generic_template_baseline": {"true_positives": f"{icon_old}/{len(icon_cases)}", "false_positives": f"{icon_old_fp}/{len(icon_negatives)}", "median_ms_per_frame": icon_old_ms},
            "pr_alpha_mask_multi_scale": {"true_positives": f"{icon_new}/{len(icon_cases)}", "false_positives": f"{icon_new_fp}/{len(icon_negatives)}", "median_ms_per_frame": icon_new_ms},
            "speedup": round(icon_old_ms / icon_new_ms, 2) if icon_new_ms else None,
        },
        "reward_banner": {
            "previous_single_scale_match": {"true_positives": f"{old_reward_tp}/{len(reward_cases)}", "false_positives": f"{old_reward_fp}/{len(reward_negatives)}", "median_ms_per_frame": old_reward_ms},
            "pr_paired_multi_scale_match": {"true_positives": f"{new_reward_tp}/{len(reward_cases)}", "false_positives": f"{new_reward_fp}/{len(reward_negatives)}", "median_ms_per_frame": new_reward_ms},
            "speedup": round(old_reward_ms / new_reward_ms, 2) if new_reward_ms else None,
        },
        "added_assets_bytes": sum(os.path.getsize(os.path.join(config.assets_dir(), name)) for name in ("pirate_token_btn.webp", "reward_banner_left.png", "reward_banner_right.png")),
    }, indent=2))


if __name__ == "__main__":
    main()
