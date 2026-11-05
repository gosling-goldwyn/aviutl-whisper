"""Run the real app with deterministic native file-dialog answers for MP4 E2E.

Only native dialogs are replaced. All JS, codecs, HTTP reads and Python writes
are production implementations; this module is never shipped with the app.
"""
import json
import os
from pathlib import Path

import webview

from aviutl_whisper.app import main

original_create = webview.create_window
control = Path(os.environ['AVIUTL_TEST_VIDEO_CONTROL'])


def create_window(*args, **kwargs):
    window = original_create(*args, **kwargs)

    def dialog(dialog_type, **options):
        data = json.loads(control.read_text(encoding='utf-8'))
        return [data['save' if dialog_type == webview.FileDialog.SAVE else 'open']]

    window.create_file_dialog = dialog
    return window


webview.create_window = create_window
main()
