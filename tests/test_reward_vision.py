import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config
from vision import Plate, find_plate, load_template_auto
from reward_vision import find_pirate_token, find_reward_banner, load_reward_banner_templates


def _reward_templates():
    return load_reward_banner_templates(config.assets_dir(), config.bundled_assets_dir())


def _reward_frame(width, height, variation=1.0, y_ratio=0.35, include_right=True):
    frame = np.full((height, width, 3), 35, dtype=np.uint8)
    for index, template in enumerate(_reward_templates()):
        if index == 1 and not include_right:
            continue
        scale_x = width / 1280 * variation
        scale_y = height / 720 * variation
        patch = cv2.resize(
            template,
            (round(template.shape[1] * scale_x), round(template.shape[0] * scale_y)),
            interpolation=cv2.INTER_NEAREST,
        )
        center_x = round(width * (0.28 if index == 0 else 0.72))
        center_y = round(height * y_ratio)
        x, y = center_x - patch.shape[1] // 2, center_y - patch.shape[0] // 2
        frame[y:y + patch.shape[0], x:x + patch.shape[1]] = patch[:, :, None]
    return frame


def test_reward_banner_matches_both_corners_across_resolution_and_scale():
    templates = _reward_templates()
    for width, height, variation in ((1280, 720, 0.925), (1920, 1080, 1.075)):
        result = find_reward_banner(_reward_frame(width, height, variation), templates)
        assert result.found, result
        assert result.left_score >= 0.55 and result.right_score >= 0.55


def test_reward_banner_rejects_a_single_corner_and_misaligned_pair():
    templates = _reward_templates()
    assert not find_reward_banner(_reward_frame(1280, 720, include_right=False), templates).found
    misaligned = _reward_frame(1280, 720, y_ratio=0.28, include_right=False)
    right = templates[1]
    h, w = right.shape
    x = round(1280 * 0.72) - w // 2
    y = round(720 * 0.47) - h // 2
    misaligned[y:y + h, x:x + w] = right[:, :, None]
    assert not find_reward_banner(misaligned, templates).found


def test_pirate_token_template_is_masked_and_matches_its_asset():
    template = load_template_auto(
        config.template_path("pirate_token", "ja"), config.PLATES["pirate_token"]
    )
    assert isinstance(template, Plate)

    source = cv2.imread(config.template_path("pirate_token", "ja"), cv2.IMREAD_UNCHANGED)
    scaled = cv2.resize(source, (90, 90), interpolation=cv2.INTER_AREA)
    frame = np.full((720, 1280, 3), 30, dtype=np.uint8)
    y, x = 90, 595
    alpha = scaled[:, :, 3]
    roi = frame[y:y + 90, x:x + 90]
    roi[alpha > 0] = scaled[:, :, :3][alpha > 0]

    match, score = find_pirate_token(frame, template)
    assert match is not None and score >= template.threshold, score


def test_reward_detector_rejects_empty_and_too_small_frames():
    templates = _reward_templates()
    assert not find_reward_banner(np.zeros((90, 120, 3), np.uint8), templates).found
    assert not find_reward_banner(np.zeros((720, 1280, 3), np.uint8), (None, templates[1])).found
