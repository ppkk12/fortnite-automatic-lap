import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config
from config import Settings
from controller import Controller
from state_machine import State, StateMachine
from vision import Plate, find_plate, load_template_auto

H, W = 600, 960
ROI = (0.0, 0.5, 0.5, 1.0)

def make_plate_parts(seed=1, size=(60, 180)):
    rng = np.random.default_rng(seed)
    rim = rng.integers(0, 255, size=(size[0], size[1], 3), dtype=np.uint8)
    mask = np.full(size, 255, np.uint8)
    mask[int(size[0] * 0.25):int(size[0] * 0.75), int(size[1] * 0.15):int(size[1] * 0.85)] = 0
    return rim, mask

def button_with_label(rim, mask, label_seed):
    out = rim.copy()
    label = np.random.default_rng(label_seed).integers(0, 255, size=rim.shape, dtype=np.uint8)
    out[mask == 0] = label[mask == 0]
    return out

def frame_with(img, pos):
    frame = np.full((H, W, 3), 30, dtype=np.uint8)
    if img is not None:
        h, w = img.shape[:2]
        frame[pos[1]:pos[1] + h, pos[0]:pos[0] + w] = img
    return frame

def make_plate(threshold=0.9):
    rim, mask = make_plate_parts()
    return Plate(rim, mask, ROI, threshold), rim, mask

def test_plate_found_whatever_the_label_says():
    plate, rim, mask = make_plate()
    for label_seed in (11, 22, 33):
        frame = frame_with(button_with_label(rim, mask, label_seed), (100, 450))
        m, score = find_plate(frame, plate, Settings().scales)
        assert m is not None and score > 0.95, (label_seed, score)
        assert abs(m[0] - 190) <= 3 and abs(m[1] - 480) <= 3

def test_plate_absent_is_not_found():
    plate, _, _ = make_plate()
    m, _ = find_plate(frame_with(None, (0, 0)), plate, Settings().scales)
    assert m is None

def test_plate_outside_its_search_region_is_ignored():
    plate, rim, mask = make_plate()
    frame = frame_with(button_with_label(rim, mask, 5), (600, 100))
    m, _ = find_plate(frame, plate, Settings().scales)
    assert m is None

def test_load_template_auto_reads_alpha_as_plate(tmp_path):
    rim, mask = make_plate_parts()
    path = tmp_path / "plate.png"
    cv2.imencode(".png", np.dstack([rim, mask]))[1].tofile(str(path))
    loaded = load_template_auto(str(path), {"roi": ROI, "threshold": 0.9})
    assert isinstance(loaded, Plate) and loaded.mask.shape == mask.shape and loaded.threshold == 0.9
    assert (loaded.mask == mask).all()

    plain = load_template_auto(str(path))
    assert plain.ndim == 3 and plain.shape[2] == 3
    assert load_template_auto(str(tmp_path / "missing.png")) is None

def test_state_machine_clicks_a_plate_in_any_language(monkeypatch):
    plate, rim, mask = make_plate()
    s = Settings(click_cooldown=1.0, delay_min=0, delay_max=0)
    clicks = []
    monkeypatch.setattr(time, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(Controller, "_click", staticmethod(lambda x, y: clicks.append((x, y))))
    ctrl = Controller(s)
    sm = StateMachine(s, {"play": plate}, ctrl)
    sm.start()
    sm.step(frame_with(button_with_label(rim, mask, 77), (100, 450)))
    assert sm.state is State.IN_MATCH and len(clicks) == 1
    assert abs(clicks[0][0] - 190) <= s.jitter_px + 3 and abs(clicks[0][1] - 480) <= s.jitter_px + 3

LABEL_PLATES = [k for k in config.PLATES if k not in config.LANG_INDEPENDENT]

def test_plate_languages_use_shared_plate_files():
    for lang in config.PLATE_LANGS:
        for key in LABEL_PLATES:
            assert config.template_filename(key, lang) == config.plate_filename(key)
            assert os.path.isfile(config.template_path(key, lang)), (lang, key)

        assert config.template_filename("continue", lang) == "continue_btn.png"
        assert config.uses_template("claim_single", lang)
    assert config.uses_template("claim_single", "en")

def test_every_language_has_its_core_templates():
    for lang in config.LANGS:
        for key in ("play", "continue", "skip", "return_lobby"):
            assert os.path.isfile(config.template_path(key, lang)), (lang, key)


def test_language_neutral_japanese_button_plates_load_as_masked_templates():
    for key in ("play", "return_lobby", "claim_single", "menu_return"):
        for lang in config.LANGS:
            plate = load_template_auto(config.template_path(key, lang), config.PLATES[key])
            assert isinstance(plate, Plate), (lang, key)


def test_japanese_and_english_action_buttons_match_the_same_plate():
    cases = (
        ("play", "play_btn.png", "play_btn_en.png", (40, 530)),
        ("return_lobby", "return_lobby_btn.png", "return_lobby_btn_en.png", (720, 230)),
        ("claim_single", "claim_single_btn.png", "claim_single_btn_en.png", (800, 620)),
    )
    for key, japanese, english, (x, y) in cases:
        plate = load_template_auto(config.template_path(key, "ja"), config.PLATES[key])
        for filename in (japanese, english):
            button = cv2.imdecode(np.fromfile(config.assets_dir() + "/" + filename, np.uint8), cv2.IMREAD_COLOR)
            frame = np.full((720, 1280, 3), 35, np.uint8)
            h, w = button.shape[:2]
            frame[y:y + h, x:x + w] = button
            match, score = find_plate(frame, plate, Settings().scales)
            assert match is not None and score >= plate.threshold, (key, filename, score)


def test_japanese_and_english_claim_buttons_use_their_matching_assets():
    for lang, filename in (("ja", "claim_btn.png"), ("en", "claim_btn_en.png")):
        template = load_template_auto(config.template_path("claim", lang), config.PLATES["claim"])
        assert isinstance(template, Plate), lang
        button = cv2.imdecode(np.fromfile(config.assets_dir() + "/" + filename, np.uint8), cv2.IMREAD_COLOR)
        frame = np.full((720, 1280, 3), 35, np.uint8)
        h, w = button.shape[:2]
        frame[620:620 + h, 800:800 + w] = button
        match, score = find_plate(frame, template, Settings().scales)
        assert match is not None and score >= template.threshold, (lang, score)


def test_menu_return_plate_ignores_japanese_menu_labels():
    plate = load_template_auto(config.template_path("menu_return", "ja"), config.PLATES["menu_return"])
    source = cv2.imdecode(np.fromfile(config.assets_dir() + "/menu_return_btn.png", np.uint8), cv2.IMREAD_COLOR)
    frame = np.full((720, 1280, 3), 35, np.uint8)
    x, y = 650, 80
    h, w = source.shape[:2]
    frame[y:y + h, x:x + w] = source
    baseline, baseline_score = find_plate(frame, plate, Settings().scales)
    frame[y + 117:y + 158, x + 88:x + 270] = 255
    frame[y + 187:y + 231, x + 55:x + 295] = 0
    changed_text, changed_score = find_plate(frame, plate, Settings().scales)
    assert baseline is not None and changed_text is not None
    assert abs(baseline_score - changed_score) < 0.03


def test_shipped_plates_are_real_plates():
    for key in LABEL_PLATES:
        p = load_template_auto(config.template_path(key, "es"), config.PLATES[key])
        assert isinstance(p, Plate), key
        assert 0.2 < (p.mask == 0).mean() < 0.6
