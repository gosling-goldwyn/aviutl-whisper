# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for aviutl-whisper"""

import os
import sys
from pathlib import Path

block_cipher = None

# プロジェクトルート
ROOT = Path(SPECPATH)
SRC = ROOT / "src" / "aviutl_whisper"

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=[
        # webアセットを同梱
        (str(SRC / "web"), "aviutl_whisper/web"),
    ],
    hiddenimports=[
        "aviutl_whisper",
        "aviutl_whisper.app",
        "aviutl_whisper.api",
        "aviutl_whisper.audio",
        "aviutl_whisper.models",
        "aviutl_whisper.transcriber",
        "aviutl_whisper.diarizer",
        "aviutl_whisper.exporter",
        "faster_whisper",
        "ctranslate2",
        "speechbrain",
        "speechbrain.inference",
        "speechbrain.inference.speaker",
        "sklearn.cluster",
        "sklearn.utils._cython_blas",
        "pydub",
        "webview",
        "torchaudio",
        "soundfile",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="aviutl-whisper",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUIアプリなのでコンソール非表示
    icon=None,  # アイコンが用意できたら設定
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="aviutl-whisper",
)
