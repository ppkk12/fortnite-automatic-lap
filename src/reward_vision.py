"""Language-independent recognition for Fortnite reward indicators."""

from dataclasses import dataclass
import os
from typing import Optional, Sequence

import cv2
import numpy as np

from vision import Plate, find_plate, find_template

REWARD_REFERENCE_SIZE = (1280, 720)
REWARD_BANNER_FILES = ("reward_banner_left.png", "reward_banner_right.png")
REWARD_CORNER_SEARCH = ((0.16, 0.18, 0.40, 0.52), (0.60, 0.18, 0.85, 0.52))
REWARD_SCALE_VARIATIONS = (0.85, 0.925, 1.0, 1.075, 1.15)
REWARD_SEARCH_SCALE = 0.35
PIRATE_TOKEN_SCALES = tuple(round(0.35 + 0.05 * index, 2) for index in range(11))


@dataclass(frozen=True)
class RewardBannerMatch:
    found: bool
    left_score: float = 0.0
    right_score: float = 0.0


def find_pirate_token(frame: np.ndarray, template, threshold: float = 0.78):
    """Match the language-neutral Pirate Shop Access icon at toast-sized scales."""
    if template is None:
        return None, 0.0
    if isinstance(template, Plate):
        return find_plate(frame, template, PIRATE_TOKEN_SCALES)
    return find_template(frame, template, threshold, PIRATE_TOKEN_SCALES)


def load_reward_banner_templates(assets_dir: str, bundled_assets_dir: str):
    """Load user overrides first, then the packaged language-neutral assets."""
    roots = tuple(dict.fromkeys((assets_dir, bundled_assets_dir)))
    templates = []
    for name in REWARD_BANNER_FILES:
        loaded = None
        for root in roots:
            path = os.path.join(root, name)
            if os.path.isfile(path):
                loaded = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if loaded is not None:
                    break
        templates.append(loaded)
    return tuple(templates)


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] >= 3:
        return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    return np.empty((0, 0), dtype=np.uint8)


def _top_locations(scores: np.ndarray, limit: int = 20):
    flat = scores.ravel()
    if flat.size == 0:
        return ()
    count = min(limit, flat.size)
    indices = np.argpartition(flat, -count)[-count:]
    width = scores.shape[1]
    return tuple(
        (int(index % width), int(index // width), float(flat[index]))
        for index in indices
    )


def find_reward_banner(
    frame: np.ndarray,
    templates: Sequence[Optional[np.ndarray]],
    threshold: float = 0.55,
) -> RewardBannerMatch:
    """Find a reward toast from its paired, language-neutral lightning corners.

    Multiple nearby scales handle display scaling. Requiring both corners in the
    expected left/right regions and on the same row reduces matches from unrelated
    HUD edges or a single similar shape.
    """
    if len(templates) != 2 or any(template is None for template in templates):
        return RewardBannerMatch(False)
    height, width = frame.shape[:2]
    if height < 120 or width < 160:
        return RewardBannerMatch(False)

    gray = _gray(frame)
    if gray.size == 0:
        return RewardBannerMatch(False)
    ref_w, ref_h = REWARD_REFERENCE_SIZE
    x_scale, y_scale = width / ref_w, height / ref_h
    best = RewardBannerMatch(False)
    best_pair_score = -1.0

    for variation in REWARD_SCALE_VARIATIONS:
        locations = []
        for template, bounds in zip(templates, REWARD_CORNER_SEARCH):
            source = _gray(template)
            if source.size == 0:
                return RewardBannerMatch(False)
            x0, y0, x1, y1 = (
                int(bounds[0] * width), int(bounds[1] * height),
                int(bounds[2] * width), int(bounds[3] * height),
            )
            region_gray = gray[y0:y1, x0:x1]
            if region_gray.size == 0:
                locations.append(())
                continue
            region_width = max(1, round(region_gray.shape[1] * REWARD_SEARCH_SCALE))
            region_height = max(1, round(region_gray.shape[0] * REWARD_SEARCH_SCALE))
            region = cv2.resize(
                region_gray,
                (region_width, region_height),
                interpolation=cv2.INTER_AREA,
            )
            tw = max(8, round(source.shape[1] * x_scale * variation * REWARD_SEARCH_SCALE))
            th = max(8, round(source.shape[0] * y_scale * variation * REWARD_SEARCH_SCALE))
            if tw >= region.shape[1] or th >= region.shape[0]:
                locations.append(())
                continue
            scaled = cv2.resize(source, (tw, th), interpolation=cv2.INTER_AREA)
            scores = cv2.matchTemplate(region, scaled, cv2.TM_CCOEFF_NORMED)
            scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
            locations.append(tuple(
                (
                    x0 + (x + tw / 2) / REWARD_SEARCH_SCALE,
                    y0 + (y + th / 2) / REWARD_SEARCH_SCALE,
                    score,
                )
                for x, y, score in _top_locations(scores)
            ))

        for left_x, left_y, left_score in locations[0]:
            for right_x, right_y, right_score in locations[1]:
                if min(left_score, right_score) < threshold:
                    continue
                if abs(left_y - right_y) > height * 0.08:
                    continue
                gap = right_x - left_x
                if not width * 0.16 <= gap <= width * 0.70:
                    continue
                pair_score = min(left_score, right_score)
                if pair_score > best_pair_score:
                    best_pair_score = pair_score
                    best = RewardBannerMatch(True, left_score, right_score)

    return best
