import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config

def freeze(monkeypatch, exe_dir, bundle_dir):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle_dir), raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_dir / "pk-auto.exe"))

def test_uses_bundled_template_when_no_external_folder(tmp_path, monkeypatch):
    exe_dir, bundle = tmp_path / "app", tmp_path / "bundle"
    (bundle / "assets_default").mkdir(parents=True)
    exe_dir.mkdir()
    (bundle / "assets_default" / "play_btn_plate.png").write_bytes(b"x")
    freeze(monkeypatch, exe_dir, bundle)
    assert config.template_path("play") == str(bundle / "assets_default" / "play_btn_plate.png")

def test_external_template_overrides_bundled(tmp_path, monkeypatch):
    exe_dir, bundle = tmp_path / "app", tmp_path / "bundle"
    (bundle / "assets_default").mkdir(parents=True)
    (exe_dir / "assets").mkdir(parents=True)
    (bundle / "assets_default" / "play_btn_plate.png").write_bytes(b"bundled")
    (exe_dir / "assets" / "play_btn_plate.png").write_bytes(b"mine")
    freeze(monkeypatch, exe_dir, bundle)
    assert config.template_path("play") == str(exe_dir / "assets" / "play_btn_plate.png")

def test_cropper_always_saves_next_to_exe(tmp_path, monkeypatch):
    exe_dir, bundle = tmp_path / "app", tmp_path / "bundle"
    (bundle / "assets_default").mkdir(parents=True)
    exe_dir.mkdir()
    (bundle / "assets_default" / "play_btn_plate.png").write_bytes(b"x")
    freeze(monkeypatch, exe_dir, bundle)
    assert config.external_template_path("play") == str(exe_dir / "assets" / "play_btn_plate.png")

def test_template_filename_per_language():
    assert config.template_filename("play", "ja") == "play_btn_plate.png"
    assert config.template_filename("play", "en") == "play_btn_plate.png"
    assert config.template_filename("return_lobby", "en") == "return_lobby_btn_plate.png"
    assert config.template_filename("claim_single", "ru") == "claim_single_btn_plate.png"
    assert config.template_filename("claim", "ja") == "claim_btn.png"
    assert config.template_filename("claim", "en") == "claim_btn_en.png"
    assert config.template_filename("claim", "es") == "claim_btn_plate.png"

def test_language_specific_external_override(tmp_path, monkeypatch):
    exe_dir, bundle = tmp_path / "app", tmp_path / "bundle"
    (bundle / "assets_default").mkdir(parents=True)
    (exe_dir / "assets").mkdir(parents=True)
    (bundle / "assets_default" / "popup_btn_en.png").write_bytes(b"bundled-en")
    (exe_dir / "assets" / "popup_btn.png").write_bytes(b"mine-ja")
    freeze(monkeypatch, exe_dir, bundle)

    assert config.template_path("popup", "ja") == str(exe_dir / "assets" / "popup_btn.png")
    assert config.template_path("popup", "en") == str(bundle / "assets_default" / "popup_btn_en.png")
    assert config.external_template_path("popup", "en") == str(exe_dir / "assets" / "popup_btn_en.png")


def test_language_neutral_buttons_share_plate_templates():
    for key in ("play", "return_lobby", "claim_single", "menu_return"):
        filenames = {config.template_filename(key, lang) for lang in config.LANGS}
        assert len(filenames) == 1, (key, filenames)
        for lang in config.LANGS:
            assert os.path.isfile(config.template_path(key, lang)), (lang, key)


def test_claim_button_has_a_template_for_every_language():
    for lang in config.LANGS:
        assert os.path.isfile(config.template_path("claim", lang)), lang


def test_every_control_button_used_by_the_state_machines_has_a_template():
    keys = ("play", "continue", "skip", "return_lobby", "claim", "claim_single", "menu_btn", "menu_return", "hud")
    for lang in config.LANGS:
        for key in keys:
            assert config.uses_template(key, lang)
            assert os.path.isfile(config.template_path(key, lang)), (lang, key)

def test_core_templates_exist_for_every_language():
    for lang in config.LANGS:
        for key in ("play", "continue", "skip", "return_lobby"):
            assert os.path.isfile(config.template_path(key, lang)), (lang, key)

def test_claim_single_template_exists_for_every_language():
    for lang in config.LANGS:
        assert os.path.isfile(config.template_path("claim_single", lang)), lang
        assert os.path.isfile(config.template_path("claim", lang)), lang


def test_pirate_token_asset_is_shared_across_languages():
    for lang in config.LANGS:
        assert config.template_filename("pirate_token", lang) == "pirate_token_btn.webp"
        assert os.path.isfile(config.template_path("pirate_token", lang)), lang
