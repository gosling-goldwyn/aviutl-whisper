"""アプリ起動設定の単体テスト。"""

import pytest

from aviutl_whisper import app, settings


def test_configure_remote_debugging_uses_pywebview_setting(monkeypatch):
    monkeypatch.setenv(app.REMOTE_DEBUGGING_PORT_ENV, "9333")
    monkeypatch.setitem(app.webview.settings, "REMOTE_DEBUGGING_PORT", None)

    app._configure_remote_debugging()

    assert app.webview.settings["REMOTE_DEBUGGING_PORT"] == 9333


@pytest.mark.parametrize("value", ["invalid", "0", "65536"])
def test_configure_remote_debugging_rejects_invalid_port(monkeypatch, value):
    monkeypatch.setenv(app.REMOTE_DEBUGGING_PORT_ENV, value)

    with pytest.raises(ValueError, match=app.REMOTE_DEBUGGING_PORT_ENV):
        app._configure_remote_debugging()


def test_settings_path_can_be_isolated_without_replacing_localappdata(
    monkeypatch, tmp_path
):
    isolated_path = tmp_path / "settings.json"
    monkeypatch.setenv("AVIUTL_WHISPER_SETTINGS_PATH", str(isolated_path))

    assert settings._get_settings_path() == isolated_path
