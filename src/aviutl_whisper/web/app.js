// aviutl-whisper フロントエンド

const $ = (sel) => document.querySelector(sel);
const show = (el) => el.classList.remove("hidden");
const hide = (el) => el.classList.add("hidden");

let selectedFile = null;
let isProcessing = false;
let exoDefaults = null;
let lastSpeakers = [];
let currentMapping = {};
let backgroundImage = "";
let savedTtsVoiceIds = {};
let availableTtsVoices = [];
let availableTtsModels = [];
let ttsVoicePreviewAudio = null;
let ttsSelectedIndex = 0;
let ttsBusy = false;
let ttsSaveTimer = null;
let selectedTtsStatusTimer = null;
let selectedTtsStatusRequestId = 0;
let selectedTtsStatusItem = null;

// --- セグメントテキスト編集セッション管理 ---
let _segTextEditStarted = false;
let _segTextDebounceTimer = null;

// --- プロジェクト状態 ---
let isDirty = false;

function markDirty() {
    isDirty = true;
    try { pywebview.api.mark_dirty(); } catch (e) { /* API未準備時は無視 */ }
}

function clearDirty() {
    isDirty = false;
}

function enableSaveMenuItems() {
    $("#menu-save-project").disabled = false;
    $("#menu-save-project-as").disabled = false;
}

function setTtsAvailability(enabled) {
    $("#btn-open-tts").disabled = !enabled;
    $("#menu-open-tts").disabled = !enabled;
}

// --- Undo/Redo ---
const MAX_UNDO = 50;
let undoStack = [];
let redoStack = [];
let pendingExoSnapshot = null;

const DEFAULT_SPEAKER_COLORS = [
    "ffffff", "00ffff", "00ff00", "ff00ff",
    "ffff00", "ff8000", "8080ff", "80ff80",
];

const LAYOUT_STORAGE_KEYS = {
    leftCollapsed: "aviutlWhisper.leftSidebarCollapsed",
    rightWidth: "aviutlWhisper.rightPaneWidth",
    segmentListHeight: "aviutlWhisper.segmentListHeight",
};

// --- 初期化 ---
document.addEventListener("DOMContentLoaded", () => {
    initEventListeners();
    initPaneLayout();
    updateHfTokenVisibility();
    renderSpeakerColors();
    renderSpeakerTachie();
});

// pywebview API が準備完了してからAPI呼び出し
window.__aviutlWhisperReady = false;
window.addEventListener("pywebviewready", async () => {
    await loadDeviceInfo();
    await loadExoDefaults();
    await loadFonts();
    await loadSavedSettings();
    window.__aviutlWhisperReady = true;
});

function initEventListeners() {
    $("#btn-select-file").addEventListener("click", selectFile);
    $("#btn-open-settings").addEventListener("click", openTranscriptionModal);
    $("#btn-close-modal").addEventListener("click", closeTranscriptionModal);
    $("#btn-modal-ok").addEventListener("click", closeTranscriptionModal);
    $("#btn-start").addEventListener("click", startTranscription);
    $("#btn-cancel").addEventListener("click", cancelTranscription);
    $("#btn-save").addEventListener("click", saveResult);
    $("#btn-open-tts").addEventListener("click", openTtsModal);
    $("#btn-close-tts").addEventListener("click", closeTtsModal);
    $("#btn-select-tts-dir").addEventListener("click", selectTtsOutputDir);
    $("#btn-save-tts-txt").addEventListener("click", () => saveTtsScript("txt"));
    $("#btn-save-tts-csv").addEventListener("click", () => saveTtsScript("csv"));
    $("#btn-save-tts-wav").addEventListener("click", saveCombinedTtsWav);
    $("#btn-generate-tts-all").addEventListener("click", generateAllTts);
    $("#btn-generate-tts-selected").addEventListener("click", () => generateOneTts(ttsSelectedIndex));
    $("#btn-play-tts-selected").addEventListener("click", () => playTts(ttsSelectedIndex));
    $("#btn-refresh-tts").addEventListener("click", refreshTtsStatus);
    $("#btn-cancel-tts").addEventListener("click", () => pywebview.api.cancel_tts());
    $("#btn-seg-tts-play").addEventListener("click", () => playTts(previewIndex));
    $("#btn-seg-tts-regenerate").addEventListener("click", () => generateOneTts(previewIndex));
    $("#btn-fetch-tts-voices").addEventListener("click", fetchTtsVoices);
    $("#btn-fetch-tts-models").addEventListener("click", fetchTtsModels);
    $("#btn-bg-image").addEventListener("click", selectBackgroundImage);
    $("#btn-bg-image-clear").addEventListener("click", clearBackgroundImage);
    $("#btn-prev-seg").addEventListener("click", () => navigatePreview(-1));
    $("#btn-next-seg").addEventListener("click", () => navigatePreview(1));
    $("#btn-seg-apply").addEventListener("click", applySegmentEdit);
    $("#btn-seg-play").addEventListener("click", playSegmentAudio);

    // テキストボックスの変更を自動でプレビューに反映
    const segTextEl = $("#seg-edit-text");
    segTextEl.addEventListener("focus", () => {
        if (!_segTextEditStarted) {
            _segTextEditStarted = true;
            pushUndo();
        }
    });
    segTextEl.addEventListener("input", () => {
        clearTimeout(_segTextDebounceTimer);
        _segTextDebounceTimer = setTimeout(() => applySegmentEdit(true), 300);
    });
    segTextEl.addEventListener("blur", () => {
        _segTextEditStarted = false;
    });
    $("#btn-seg-add").addEventListener("click", addSegment);
    $("#btn-seg-merge-prev").addEventListener("click", mergePrevSegment);
    $("#btn-seg-merge-next").addEventListener("click", mergeNextSegment);
    $("#btn-seg-delete").addEventListener("click", deleteSegment);
    $("#btn-load-project").addEventListener("click", loadProjectWithCheck);
    $("#btn-save-project").addEventListener("click", saveProject);
    $("#diarization-method").addEventListener("change", updateHfTokenVisibility);
    $("#num-speakers").addEventListener("change", () => {
        renderSpeakerColors();
        renderSpeakerTachie();
    });

    // モーダル背景クリックで閉じる
    $("#transcription-modal").addEventListener("click", (e) => {
        if (e.target === $("#transcription-modal")) closeTranscriptionModal();
    });
    $("#tts-modal").addEventListener("click", (e) => {
        if (e.target === $("#tts-modal")) closeTtsModal();
    });

    // メニューバー
    const fileEntry = $("#menu-file-entry");
    fileEntry.querySelector(".menu-entry-btn").addEventListener("click", (e) => {
        e.stopPropagation();
        fileEntry.classList.toggle("open");
    });
    document.addEventListener("click", () => {
        fileEntry.classList.remove("open");
    });
    $("#menu-open-project").addEventListener("click", () => { fileEntry.classList.remove("open"); loadProjectWithCheck(); });
    $("#menu-save-project").addEventListener("click", () => { fileEntry.classList.remove("open"); saveProject(); });
    $("#menu-save-project-as").addEventListener("click", () => { fileEntry.classList.remove("open"); saveProjectAs(); });

    const editEntry = $("#menu-edit-entry");
    editEntry.querySelector(".menu-entry-btn").addEventListener("click", (e) => {
        e.stopPropagation();
        editEntry.classList.toggle("open");
    });
    document.addEventListener("click", () => {
        editEntry.classList.remove("open");
    });
    $("#menu-undo").addEventListener("click", () => { editEntry.classList.remove("open"); undo(); });
    $("#menu-redo").addEventListener("click", () => { editEntry.classList.remove("open"); redo(); });

    const toolsEntry = $("#menu-tools-entry");
    toolsEntry.querySelector(".menu-entry-btn").addEventListener("click", (e) => {
        e.stopPropagation();
        toolsEntry.classList.toggle("open");
    });
    document.addEventListener("click", () => {
        toolsEntry.classList.remove("open");
    });
    $("#menu-open-tts").addEventListener("click", () => {
        toolsEntry.classList.remove("open");
        openTtsModal();
    });

    // キーボードナビゲーション
    document.addEventListener("keydown", (e) => {
        // テキスト文字入力中のみネイティブ Ctrl+Z を優先（テキスト誤字修正用）
        // number input / SELECT / BUTTON などではアプリ側 Undo を発火させる
        const isTextInput = (
            (e.target.tagName === "TEXTAREA") ||
            (e.target.tagName === "INPUT" && (e.target.type === "text" || e.target.type === "password" || e.target.type === "search"))
        );

        if (e.ctrlKey && e.key === "z" && !e.shiftKey) {
            if (!isTextInput) {
                e.preventDefault();
                undo();
                return;
            }
        }
        if (e.ctrlKey && (e.key === "y" || (e.key === "z" && e.shiftKey))) {
            if (!isTextInput) {
                e.preventDefault();
                redo();
                return;
            }
        }
        if (e.ctrlKey && !e.shiftKey && e.key === "o") {
            e.preventDefault();
            loadProjectWithCheck();
            return;
        }
        if (e.ctrlKey && e.shiftKey && (e.key === "s" || e.key === "S")) {
            e.preventDefault();
            saveProjectAs();
            return;
        }
        if (e.ctrlKey && !e.shiftKey && (e.key === "s" || e.key === "S")) {
            e.preventDefault();
            saveProject();
            return;
        }
        if (e.ctrlKey && e.shiftKey && (e.key === "e" || e.key === "E")) {
            e.preventDefault();
            if (previewSegments.length > 0) openTtsModal();
            return;
        }
        if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;
        if (previewSegments.length === 0) return;
        if (e.key === "ArrowLeft") { navigatePreview(-1); e.preventDefault(); }
        if (e.key === "ArrowRight") { navigatePreview(1); e.preventDefault(); }
    });
}

function initPaneLayout() {
    const layout = $("#app-layout");
    const toggleBtn = $("#btn-toggle-settings-pane");
    const rightSplitter = $("#right-splitter");
    const centerSplitter = $("#center-splitter");
    const rightPane = $("#segment-editor-pane");
    const previewPane = $("#preview-pane");
    if (!layout || !toggleBtn || !rightSplitter || !centerSplitter || !rightPane || !previewPane) return;

    const savedCollapsed = readStorage(LAYOUT_STORAGE_KEYS.leftCollapsed);
    setSettingsPaneCollapsed(savedCollapsed === "true");

    const savedRightWidth = parseInt(readStorage(LAYOUT_STORAGE_KEYS.rightWidth), 10);
    if (!Number.isNaN(savedRightWidth)) {
        setRightPaneWidth(savedRightWidth);
    }

    const savedSegmentListHeight = parseInt(readStorage(LAYOUT_STORAGE_KEYS.segmentListHeight), 10);
    if (!Number.isNaN(savedSegmentListHeight)) {
        setSegmentListHeight(savedSegmentListHeight);
    }

    toggleBtn.addEventListener("click", () => {
        const collapsed = !layout.classList.contains("settings-collapsed");
        setSettingsPaneCollapsed(collapsed);
        writeStorage(LAYOUT_STORAGE_KEYS.leftCollapsed, String(collapsed));
    });

    rightSplitter.addEventListener("pointerdown", (e) => {
        if (window.matchMedia("(max-width: 768px)").matches) return;
        if (e.pointerType === "mouse") return;
        e.preventDefault();
        rightSplitter.setPointerCapture(e.pointerId);
        rightSplitter.classList.add("dragging");
        document.body.classList.add("pane-dragging");

        const startX = e.clientX;
        const startWidth = rightPane.getBoundingClientRect().width;

        const onMove = (moveEvent) => {
            const nextWidth = startWidth - (moveEvent.clientX - startX);
            setRightPaneWidth(nextWidth, true);
        };
        const onUp = () => {
            rightSplitter.classList.remove("dragging");
            document.body.classList.remove("pane-dragging");
            writeStorage(LAYOUT_STORAGE_KEYS.rightWidth, getCssPixelValue("--right-pane-w"));
            document.removeEventListener("pointermove", onMove);
            document.removeEventListener("pointerup", onUp);
            document.removeEventListener("pointercancel", onUp);
        };

        document.addEventListener("pointermove", onMove);
        document.addEventListener("pointerup", onUp);
        document.addEventListener("pointercancel", onUp);
    });

    rightSplitter.addEventListener("mousedown", (e) => {
        if (window.matchMedia("(max-width: 768px)").matches) return;
        if (e.button !== 0) return;
        e.preventDefault();
        rightSplitter.classList.add("dragging");
        document.body.classList.add("pane-dragging");

        const startX = e.clientX;
        const startWidth = rightPane.getBoundingClientRect().width;

        const onMove = (moveEvent) => {
            const nextWidth = startWidth - (moveEvent.clientX - startX);
            setRightPaneWidth(nextWidth, true);
        };
        const onUp = () => {
            rightSplitter.classList.remove("dragging");
            document.body.classList.remove("pane-dragging");
            writeStorage(LAYOUT_STORAGE_KEYS.rightWidth, getCssPixelValue("--right-pane-w"));
            document.removeEventListener("mousemove", onMove);
            document.removeEventListener("mouseup", onUp);
        };

        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup", onUp);
    });

    centerSplitter.addEventListener("pointerdown", (e) => {
        if (window.matchMedia("(max-width: 768px)").matches) return;
        if (e.pointerType === "mouse") return;
        e.preventDefault();
        centerSplitter.setPointerCapture(e.pointerId);
        centerSplitter.classList.add("dragging");
        document.body.classList.add("pane-dragging-y");

        const startY = e.clientY;
        const startHeight = parseFloat(getComputedStyle(layout).getPropertyValue("--segment-list-h")) || 320;

        const onMove = (moveEvent) => {
            const nextHeight = startHeight - (moveEvent.clientY - startY);
            setSegmentListHeight(nextHeight, true);
        };
        const onUp = () => {
            centerSplitter.classList.remove("dragging");
            document.body.classList.remove("pane-dragging-y");
            writeStorage(LAYOUT_STORAGE_KEYS.segmentListHeight, getCssPixelValue("--segment-list-h"));
            document.removeEventListener("pointermove", onMove);
            document.removeEventListener("pointerup", onUp);
            document.removeEventListener("pointercancel", onUp);
        };

        document.addEventListener("pointermove", onMove);
        document.addEventListener("pointerup", onUp);
        document.addEventListener("pointercancel", onUp);
    });

    centerSplitter.addEventListener("mousedown", (e) => {
        if (window.matchMedia("(max-width: 768px)").matches) return;
        if (e.button !== 0) return;
        e.preventDefault();
        centerSplitter.classList.add("dragging");
        document.body.classList.add("pane-dragging-y");

        const startY = e.clientY;
        const startHeight = parseFloat(getComputedStyle(layout).getPropertyValue("--segment-list-h")) || 320;

        const onMove = (moveEvent) => {
            const nextHeight = startHeight - (moveEvent.clientY - startY);
            setSegmentListHeight(nextHeight, true);
        };
        const onUp = () => {
            centerSplitter.classList.remove("dragging");
            document.body.classList.remove("pane-dragging-y");
            writeStorage(LAYOUT_STORAGE_KEYS.segmentListHeight, getCssPixelValue("--segment-list-h"));
            document.removeEventListener("mousemove", onMove);
            document.removeEventListener("mouseup", onUp);
        };

        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup", onUp);
    });

    window.addEventListener("resize", () => {
        setRightPaneWidth(parseFloat(getComputedStyle(layout).getPropertyValue("--right-pane-w")) || 340);
        setSegmentListHeight(parseFloat(getComputedStyle(layout).getPropertyValue("--segment-list-h")) || 320);
    });
}

function setSettingsPaneCollapsed(collapsed) {
    const layout = $("#app-layout");
    const toggleBtn = $("#btn-toggle-settings-pane");
    if (!layout || !toggleBtn) return;
    layout.classList.toggle("settings-collapsed", collapsed);
    toggleBtn.setAttribute("aria-expanded", String(!collapsed));
    toggleBtn.title = collapsed ? "サイドバーを展開" : "サイドバーを折り畳む";
    toggleBtn.setAttribute("aria-label", toggleBtn.title);
}

function setRightPaneWidth(width, persist = false) {
    const layout = $("#app-layout");
    if (!layout) return;
    const maxWidth = Math.max(280, Math.floor(window.innerWidth * 0.45));
    const nextWidth = clamp(Math.round(width), 280, maxWidth);
    layout.style.setProperty("--right-pane-w", `${nextWidth}px`);
    if (persist) writeStorage(LAYOUT_STORAGE_KEYS.rightWidth, String(nextWidth));
}

function setSegmentListHeight(height, persist = false) {
    const layout = $("#app-layout");
    const previewPane = $("#preview-pane");
    if (!layout || !previewPane) return;
    const maxHeight = Math.max(180, Math.floor(previewPane.getBoundingClientRect().height * 0.6));
    const nextHeight = clamp(Math.round(height), 180, maxHeight);
    layout.style.setProperty("--segment-list-h", `${nextHeight}px`);
    if (persist) writeStorage(LAYOUT_STORAGE_KEYS.segmentListHeight, String(nextHeight));
}

function getCssPixelValue(name) {
    const layout = $("#app-layout");
    if (!layout) return "";
    const value = parseFloat(getComputedStyle(layout).getPropertyValue(name));
    return Number.isNaN(value) ? "" : String(Math.round(value));
}

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

function readStorage(key) {
    try {
        return window.localStorage.getItem(key);
    } catch (e) {
        return null;
    }
}

function writeStorage(key, value) {
    try {
        window.localStorage.setItem(key, value);
    } catch (e) {
        // localStorage が利用できない環境ではレイアウト保存だけ省略する
    }
}

// --- モーダル ---
function openTranscriptionModal() {
    show($("#transcription-modal"));
}

function closeTranscriptionModal() {
    hide($("#transcription-modal"));
    scheduleAutoSave();
}

// --- HFトークン欄の表示/非表示 ---
function updateHfTokenVisibility() {
    const tokenItem = $("#hf-token-item");
    if ($("#diarization-method").value === "pyannote") {
        tokenItem.style.display = "";
    } else {
        tokenItem.style.display = "none";
    }
}

// --- フォント一覧読み込み ---
async function loadFonts() {
    try {
        const fonts = await pywebview.api.get_system_fonts();
        if (fonts && fonts.length > 0) {
            const select = $("#exo-font");
            select.innerHTML = "";
            for (const font of fonts) {
                const opt = document.createElement("option");
                opt.value = font;
                opt.textContent = font;
                if (font === (exoDefaults?.font || "MS UI Gothic")) {
                    opt.selected = true;
                }
                select.appendChild(opt);
            }
        }
    } catch (e) {
        console.error("フォント一覧取得エラー:", e);
    }
}

// --- exoデフォルト値の読み込み ---
async function loadExoDefaults() {
    try {
        exoDefaults = await pywebview.api.get_exo_defaults();
    } catch (e) {
        console.error("exoデフォルト取得エラー:", e);
    }
}

// --- 話者ごとの色設定UI生成 ---
function renderSpeakerColors() {
    const container = $("#speaker-colors-list");
    const numValue = $("#num-speakers").value;
    const numSpeakers = numValue === "auto" ? 2 : parseInt(numValue);
    const colors = exoDefaults?.speaker_colors || DEFAULT_SPEAKER_COLORS;
    const edgeColor = exoDefaults?.default_edge_color || "000000";

    container.innerHTML = "";
    for (let i = 0; i < numSpeakers; i++) {
        const color = colors[i % colors.length];
        const row = document.createElement("div");
        row.className = "speaker-color-row";
        row.innerHTML = `
            <span class="speaker-label">話者 ${i + 1}</span>
            <div class="color-group">
                <label>文字色</label>
                <input type="color" class="speaker-text-color" data-index="${i}" value="#${color}">
                <input type="text" class="hex-input speaker-text-hex" data-index="${i}" value="${color}" maxlength="6">
            </div>
            <div class="color-group">
                <label>縁色</label>
                <input type="color" class="speaker-edge-color" data-index="${i}" value="#${edgeColor}">
                <input type="text" class="hex-input speaker-edge-hex" data-index="${i}" value="${edgeColor}" maxlength="6">
            </div>
        `;
        container.appendChild(row);
    }

    // カラーピッカーとhex入力を同期
    container.querySelectorAll("input[type='color']").forEach(picker => {
        picker.addEventListener("input", (e) => {
            const idx = e.target.dataset.index;
            const isEdge = e.target.classList.contains("speaker-edge-color");
            const hexClass = isEdge ? ".speaker-edge-hex" : ".speaker-text-hex";
            const hexInput = container.querySelector(`${hexClass}[data-index="${idx}"]`);
            if (hexInput) hexInput.value = e.target.value.replace("#", "");
        });
    });
    container.querySelectorAll(".hex-input").forEach(input => {
        input.addEventListener("input", (e) => {
            const idx = e.target.dataset.index;
            const isEdge = e.target.classList.contains("speaker-edge-hex");
            const colorClass = isEdge ? ".speaker-edge-color" : ".speaker-text-color";
            const picker = container.querySelector(`${colorClass}[data-index="${idx}"]`);
            const hex = e.target.value.replace("#", "");
            if (hex.length === 6 && /^[0-9a-fA-F]{6}$/.test(hex) && picker) {
                picker.value = "#" + hex;
            }
        });
    });
}

// --- 話者ごとの立ち絵設定UI生成 ---
let tachieData = [];

function renderSpeakerTachie() {
    const container = $("#speaker-tachie-list");
    const numValue = $("#num-speakers").value;
    const numSpeakers = numValue === "auto" ? 2 : parseInt(numValue);

    while (tachieData.length < numSpeakers) {
        tachieData.push({ file: "", x: 0, y: 0, scale: 100 });
    }

    container.innerHTML = "";
    for (let i = 0; i < numSpeakers; i++) {
        const data = tachieData[i];
        const fileName = data.file ? data.file.split(/[\\/]/).pop() : "未選択";
        const row = document.createElement("div");
        row.className = "speaker-tachie-row";
        row.dataset.index = i;
        row.innerHTML = `
            <div class="tachie-header">
                <span class="speaker-label">話者 ${i + 1}</span>
                <div class="tachie-file-group">
                    <button class="btn btn-tachie-select" data-index="${i}">画像選択</button>
                    <span class="tachie-file-name" data-index="${i}">${fileName}</span>
                    <button class="btn-tachie-clear" data-index="${i}" title="クリア">✕</button>
                </div>
            </div>
            <div class="tachie-params">
                <div class="tachie-param">
                    <label>X位置</label>
                    <input type="number" class="tachie-x" data-index="${i}" value="${data.x}" step="0.1">
                </div>
                <div class="tachie-param">
                    <label>Y位置</label>
                    <input type="number" class="tachie-y" data-index="${i}" value="${data.y}" step="0.1">
                </div>
                <div class="tachie-param">
                    <label>拡大率 (%)</label>
                    <input type="number" class="tachie-scale" data-index="${i}" value="${data.scale}" step="1" min="1" max="1000">
                </div>
            </div>
        `;
        container.appendChild(row);
    }

    container.querySelectorAll(".btn-tachie-select").forEach(btn => {
        btn.addEventListener("click", (e) => selectTachieImage(parseInt(e.target.dataset.index)));
    });

    container.querySelectorAll(".btn-tachie-clear").forEach(btn => {
        btn.addEventListener("click", (e) => {
            const idx = parseInt(e.target.dataset.index);
            pushUndo();
            tachieData[idx].file = "";
            const label = container.querySelector(`.tachie-file-name[data-index="${idx}"]`);
            if (label) label.textContent = "未選択";
            scheduleAutoSave();
        });
    });

    container.querySelectorAll(".tachie-x, .tachie-y, .tachie-scale").forEach(input => {
        input.addEventListener("change", (e) => {
            const idx = parseInt(e.target.dataset.index);
            if (e.target.classList.contains("tachie-x")) tachieData[idx].x = parseFloat(e.target.value) || 0;
            if (e.target.classList.contains("tachie-y")) tachieData[idx].y = parseFloat(e.target.value) || 0;
            if (e.target.classList.contains("tachie-scale")) tachieData[idx].scale = parseFloat(e.target.value) || 100;
            scheduleAutoSave();
        });
    });
}

async function selectTachieImage(speakerIndex) {
    try {
        const path = await pywebview.api.select_image_file();
        if (path) {
            pushUndo();
            tachieData[speakerIndex].file = path;
            const label = document.querySelector(`.tachie-file-name[data-index="${speakerIndex}"]`);
            if (label) label.textContent = path.split(/[\\/]/).pop();
            scheduleAutoSave();
            schedulePreviewRedraw();
        }
    } catch (e) {
        console.error("画像選択エラー:", e);
    }
}

// --- exo設定を収集 ---
function collectExoSettings() {
    const speakerColors = [];
    const speakerEdgeColors = [];
    document.querySelectorAll(".speaker-text-hex").forEach(el => {
        speakerColors.push(el.value.replace("#", ""));
    });
    document.querySelectorAll(".speaker-edge-hex").forEach(el => {
        speakerEdgeColors.push(el.value.replace("#", ""));
    });

    return {
        font: $("#exo-font").value,
        font_size: parseInt($("#exo-font-size").value) || 34,
        spacing_x: parseInt($("#exo-spacing-x").value) || 0,
        spacing_y: parseInt($("#exo-spacing-y").value) || 0,
        display_speed: parseFloat($("#exo-display-speed").value) || 0,
        align: parseInt($("#exo-align").value),
        bold: $("#exo-bold").checked,
        italic: $("#exo-italic").checked,
        soft_edge: $("#exo-soft-edge").checked,
        pos_x: parseFloat($("#exo-pos-x").value) || 0,
        pos_y: parseFloat($("#exo-pos-y").value) || 0,
        max_chars_per_line: parseInt($("#exo-max-chars").value) || 0,
        speaker_colors: speakerColors,
        speaker_edge_colors: speakerEdgeColors,
        speaker_images: tachieData.map(d => ({
            file: d.file || "",
            x: d.x || 0,
            y: d.y || 0,
            scale: d.scale || 100,
        })),
        background_image: backgroundImage,
    };
}

// --- 背景画像選択 ---
async function selectBackgroundImage() {
    try {
        const result = await pywebview.api.select_image_file();
        if (result) {
            pushUndo();
            backgroundImage = result;
            const name = result.split(/[\\/]/).pop();
            $("#bg-image-name").textContent = name;
            markDirty();
            autoSave();
            schedulePreviewRedraw();
        }
    } catch (e) {
        console.error("背景画像選択エラー:", e);
    }
}

function clearBackgroundImage() {
    pushUndo();
    backgroundImage = "";
    $("#bg-image-name").textContent = "未選択";
    markDirty();
    autoSave();
    schedulePreviewRedraw();
}

// --- ファイル選択 ---
async function selectFile() {
    try {
        const result = await pywebview.api.select_file();
        if (result) {
            selectedFile = result.path;
            $("#file-name").textContent = result.name;
            const info = `形式: ${result.extension} | サイズ: ${formatBytes(result.size)}`;
            $("#file-info").textContent = info;
            show($("#file-info"));
            $("#btn-start").disabled = false;
        }
    } catch (e) {
        console.error("ファイル選択エラー:", e);
    }
}

// --- 文字起こし実行 ---
async function startTranscription() {
    if (!selectedFile || isProcessing) return;

    isProcessing = true;
    $("#btn-start").disabled = true;
    show($("#btn-cancel"));
    show($("#progress-area"));

    setProgress(0, "処理を開始しています...");

    const pollId = setInterval(async () => {
        try {
            const status = await pywebview.api.get_progress();
            if (status) {
                if (status.progress < 0) {
                    setProgressIndeterminate(status.message);
                } else {
                    setProgress(status.progress * 100, status.message);
                }
            }
        } catch (e) { /* ignore */ }
    }, 500);

    try {
        const settings = {
            model_size: $("#model-size").value,
            language: $("#language").value || null,
            num_speakers: (() => {
                const v = $("#num-speakers").value;
                return v === "auto" ? null : parseInt(v);
            })(),
            output_format: "exo",
            diarization_method: $("#diarization-method").value,
            hf_token: $("#hf-token").value || "",
            exo_settings: collectExoSettings(),
        };

        const result = await pywebview.api.transcribe(selectedFile, settings);

        clearInterval(pollId);

        if (result.success) {
            showResult(result);
        } else {
            alert("エラー: " + result.error);
        }
    } catch (e) {
        clearInterval(pollId);
        alert("処理中にエラーが発生しました: " + e);
    } finally {
        isProcessing = false;
        $("#btn-start").disabled = false;
        hide($("#btn-cancel"));
        hide($("#progress-area"));
    }
}

async function cancelTranscription() {
    try {
        await pywebview.api.cancel();
    } catch (e) { /* ignore */ }
}

// --- 結果表示 ---
function showResult(result) {
    const stats = $("#result-stats");
    stats.innerHTML = `
        <span>🎯 ${result.num_segments}セグメント</span>
        <span>🗣️ ${result.num_speakers}人</span>
        <span>🌐 ${result.language}</span>
    `;

    setProgress(100, "完了！");
    $("#btn-save").disabled = false;
    setTtsAvailability(true);
    $("#btn-save-project").disabled = false;

    // 新しい文字起こし結果なので Undo 履歴をリセット
    clearUndoHistory();
    markDirty();
    enableSaveMenuItems();

    // 話者マッピングUI
    lastSpeakers = result.speakers || [];
    if (lastSpeakers.length > 1) {
        renderSpeakerMapping(lastSpeakers);
        show($("#speaker-mapping-section"));
    } else {
        hide($("#speaker-mapping-section"));
    }

    // プレビュー + セグメントテーブル
    initExoPreview();
}

async function saveResult() {
    try {
        const exoSettings = collectExoSettings();
        const mapping = Object.keys(currentMapping).length > 0 ? currentMapping : null;
        const result = await pywebview.api.save_result("exo", exoSettings, mapping);
        if (result && result.success) {
            alert("保存しました: " + result.path);
        }
    } catch (e) {
        alert("保存エラー: " + e);
    }
}

// --- プロジェクト保存・読み込み ---
function _collectProjectData() {
    return {
        source_file: selectedFile || "",
        exo_settings: collectExoSettings(),
        preview_index: previewIndex,
    };
}

async function saveProject() {
    try {
        const result = await pywebview.api.save_project(_collectProjectData());
        if (result && result.success) {
            clearDirty();
            alert("プロジェクトを保存しました: " + result.path);
        } else if (result && result.error && result.error !== "キャンセルされました") {
            alert("保存エラー: " + result.error);
        }
    } catch (e) {
        alert("プロジェクト保存エラー: " + e);
    }
}

async function saveProjectAs() {
    try {
        const result = await pywebview.api.save_project_as(_collectProjectData());
        if (result && result.success) {
            clearDirty();
            alert("プロジェクトを保存しました: " + result.path);
        } else if (result && result.error && result.error !== "キャンセルされました") {
            alert("保存エラー: " + result.error);
        }
    } catch (e) {
        alert("プロジェクト保存エラー: " + e);
    }
}

function showSaveConfirmDialog(message, onYes, onNo, onCancel) {
    $("#save-confirm-message").textContent = message;
    show($("#save-confirm-modal"));

    function cleanup() { hide($("#save-confirm-modal")); }

    $("#btn-confirm-yes").onclick = () => { cleanup(); onYes(); };
    $("#btn-confirm-no").onclick = () => { cleanup(); onNo(); };
    $("#btn-confirm-cancel").onclick = () => { cleanup(); if (onCancel) onCancel(); };
}

async function loadProjectWithCheck() {
    if (!isDirty) {
        await loadProject();
        return;
    }
    showSaveConfirmDialog(
        "変更が保存されていません。プロジェクトを開く前に保存しますか？",
        async () => {
            // はい: 保存してから開く
            const result = await pywebview.api.save_project(_collectProjectData());
            if (result && result.success) {
                clearDirty();
                await loadProject();
            } else if (result && result.error && result.error !== "キャンセルされました") {
                alert("保存エラー: " + result.error);
            }
            // 保存キャンセル時は開かない
        },
        async () => {
            // いいえ: 保存せずに開く
            await loadProject();
        },
        null  // キャンセル: 何もしない
    );
}

async function loadProject() {
    try {
        const result = await pywebview.api.load_project();
        if (!result || !result.success) {
            if (result && result.error && result.error !== "キャンセルされました") {
                alert("読み込みエラー: " + result.error);
            }
            return;
        }

        // プロジェクト読み込み時は Undo 履歴をリセット
        clearUndoHistory();
        clearDirty();
        enableSaveMenuItems();

        // ファイル情報を復元
        selectedFile = result.source_file || null;
        if (selectedFile) {
            const name = selectedFile.split(/[\/]/).pop();
            $("#file-name").textContent = name;
        } else {
            $("#file-name").textContent = "未選択";
        }

        // exo設定をUIに反映
        const exo = result.exo_settings;
        if (exo) {
            applyExoSettingsToUI(exo);
        }

        // 結果表示
        const stats = $("#result-stats");
        stats.innerHTML = `
            <span>🎯 ${result.num_segments}セグメント</span>
            <span>🗣️ ${result.num_speakers}人</span>
            <span>🌐 ${result.language || "?"}</span>
        `;

        // ボタン有効化
        $("#btn-save").disabled = false;
        setTtsAvailability(true);
        $("#btn-start").disabled = !selectedFile;
        $("#btn-save-project").disabled = false;

        // 話者マッピングUI
        lastSpeakers = result.speakers || [];
        if (lastSpeakers.length > 1) {
            renderSpeakerMapping(lastSpeakers);
            show($("#speaker-mapping-section"));
        } else {
            hide($("#speaker-mapping-section"));
        }

        // プレビュー + セグメントテーブル
        previewIndex = result.preview_index || 0;
        await initExoPreview();
    } catch (e) {
        alert("プロジェクト読み込みエラー: " + e);
    }
}

// Python の closing イベントから evaluate_js 経由で呼び出されるグローバル関数
window._showCloseConfirm = function() {
    showSaveConfirmDialog(
        "変更が保存されていません。終了する前に保存しますか？",
        async () => {
            // はい: 保存してから終了
            const result = await pywebview.api.save_project(_collectProjectData());
            if (result && result.success) {
                clearDirty();
                await pywebview.api.force_close();
            } else if (!result || !result.error || result.error === "キャンセルされました") {
                // ファイル選択キャンセル時は終了しない
            } else {
                alert("保存エラー: " + result.error);
            }
        },
        async () => {
            // いいえ: 保存せずに終了
            await pywebview.api.force_close();
        },
        null  // キャンセル: 何もしない (ウィンドウは既に閉じずに待機中)
    );
};
function applyExoSettingsToUI(exo) {
    if (exo.font) $("#exo-font").value = exo.font;
    if (exo.font_size != null) $("#exo-font-size").value = exo.font_size;
    if (exo.spacing_x != null) $("#exo-spacing-x").value = exo.spacing_x;
    if (exo.spacing_y != null) $("#exo-spacing-y").value = exo.spacing_y;
    if (exo.display_speed != null) $("#exo-display-speed").value = exo.display_speed;
    if (exo.align != null) $("#exo-align").value = exo.align;
    if (exo.pos_x != null) $("#exo-pos-x").value = exo.pos_x;
    if (exo.pos_y != null) $("#exo-pos-y").value = exo.pos_y;
    if (exo.max_chars_per_line != null) $("#exo-max-chars").value = exo.max_chars_per_line;
    $("#exo-bold").checked = !!exo.bold;
    $("#exo-italic").checked = !!exo.italic;
    $("#exo-soft-edge").checked = exo.soft_edge !== false;

    if (exo.speaker_colors) {
        exoDefaults = exoDefaults || {};
        exoDefaults.speaker_colors = exo.speaker_colors;
    }
    if (exo.speaker_edge_colors && exo.speaker_edge_colors.length > 0) {
        exoDefaults = exoDefaults || {};
        exoDefaults.speaker_edge_colors = exo.speaker_edge_colors;
    }
    renderSpeakerColors();

    if (exo.speaker_images?.length > 0) {
        tachieData = exo.speaker_images.map(img => ({
            file: img.file || "",
            x: img.x || 0,
            y: img.y || 0,
            scale: img.scale || 100,
        }));
    }
    renderSpeakerTachie();

    if (exo.background_image) {
        backgroundImage = exo.background_image;
        const name = backgroundImage.split(/[\\/]/).pop();
        $("#bg-image-name").textContent = name;
    } else {
        backgroundImage = "";
        $("#bg-image-name").textContent = "未選択";
    }

    if (exo.speaker_edge_colors?.length > 0) {
        document.querySelectorAll(".speaker-edge-hex").forEach((el, i) => {
            if (i < exo.speaker_edge_colors.length) {
                el.value = exo.speaker_edge_colors[i];
                const picker = document.querySelector(`.speaker-edge-color[data-index="${i}"]`);
                if (picker) picker.value = "#" + exo.speaker_edge_colors[i];
            }
        });
    }
}

// --- 話者マッピング ---
function renderSpeakerMapping(speakers) {
    const container = $("#speaker-mapping-list");
    const colors = exoDefaults?.speaker_colors || DEFAULT_SPEAKER_COLORS;
    currentMapping = {};
    speakers.forEach((spk, i) => { currentMapping[spk.name] = i; });

    container.innerHTML = "";
    speakers.forEach((spk, i) => {
        const color = colors[i % colors.length];
        const row = document.createElement("div");
        row.className = "speaker-mapping-row";
        row.innerHTML = `
            <button class="btn-play-sample" data-speaker="${spk.name}" title="サンプル再生">▶</button>
            <span class="color-preview-dot" style="background:#${color}" data-index="${i}"></span>
            <span class="mapping-label">${spk.name}</span>
            <span class="sample-text" title="${spk.sample_text}">${spk.sample_text}</span>
            <select class="mapping-select" data-speaker="${spk.name}">
                ${speakers.map((_, j) => `<option value="${j}" ${j === i ? "selected" : ""}>設定 ${j + 1}</option>`).join("")}
            </select>
        `;
        container.appendChild(row);
    });

    container.querySelectorAll(".btn-play-sample").forEach(btn => {
        btn.addEventListener("click", () => playSpeakerSample(btn.dataset.speaker));
    });

    container.querySelectorAll(".mapping-select").forEach(sel => {
        sel.addEventListener("change", () => applyMapping());
    });

    const swapBtn = $("#btn-swap-speakers");
    if (speakers.length === 2) {
        show(swapBtn);
        swapBtn.onclick = swapSpeakers;
    } else {
        hide(swapBtn);
    }
}

async function playSpeakerSample(speakerName) {
    try {
        const btn = document.querySelector(`.btn-play-sample[data-speaker="${speakerName}"]`);
        if (btn) { btn.textContent = "⏳"; btn.disabled = true; }
        await pywebview.api.play_speaker_sample(speakerName);
        if (btn) { btn.textContent = "▶"; btn.disabled = false; }
    } catch (e) {
        console.error("再生エラー:", e);
        const btn = document.querySelector(`.btn-play-sample[data-speaker="${speakerName}"]`);
        if (btn) { btn.textContent = "▶"; btn.disabled = false; }
    }
}

function swapSpeakers() {
    const selects = document.querySelectorAll(".mapping-select");
    if (selects.length === 2) {
        const tmp = selects[0].value;
        selects[0].value = selects[1].value;
        selects[1].value = tmp;
        applyMapping();
    }
}

async function applyMapping() {
    pushUndo();
    const selects = document.querySelectorAll(".mapping-select");
    currentMapping = {};
    selects.forEach(sel => {
        currentMapping[sel.dataset.speaker] = parseInt(sel.value);
    });

    const colors = exoDefaults?.speaker_colors || DEFAULT_SPEAKER_COLORS;
    selects.forEach(sel => {
        const slot = parseInt(sel.value);
        const row = sel.closest(".speaker-mapping-row");
        const dot = row.querySelector(".color-preview-dot");
        if (dot) dot.style.background = "#" + colors[slot % colors.length];
    });

    try {
        const exoSettings = collectExoSettings();
        const result = await pywebview.api.remap_speakers(currentMapping, "exo", exoSettings);
        if (result && result.success) {
            markDirty();
        }
        initExoPreview();
    } catch (e) {
        console.error("マッピング変更エラー:", e);
    }
}

// --- 進捗表示 ---
function setProgress(percent, message) {
    const fill = $("#progress-fill");
    fill.classList.remove("indeterminate");
    fill.style.width = percent + "%";
    $("#progress-text").textContent = message;
}

function setProgressIndeterminate(message) {
    const fill = $("#progress-fill");
    fill.classList.add("indeterminate");
    fill.style.width = "";
    $("#progress-text").textContent = message;
}

// --- デバイス情報 ---
async function loadDeviceInfo() {
    try {
        const info = await pywebview.api.get_device_info();
        if (info) {
            $("#device-info").textContent = `${info.device} | ${info.detail}`;
        }
    } catch (e) { /* ignore */ }
}

// --- 設定の保存/読み込み ---
async function loadSavedSettings() {
    try {
        const saved = await pywebview.api.load_settings();
        if (!saved) return;

        if (saved.model_size) $("#model-size").value = saved.model_size;
        if (saved.language) $("#language").value = saved.language;
        if (saved.num_speakers) $("#num-speakers").value = saved.num_speakers;
        if (saved.diarization_method) $("#diarization-method").value = saved.diarization_method;
        if (saved.hf_token_decrypted) $("#hf-token").value = saved.hf_token_decrypted;
        if (saved.elevenlabs_api_key_decrypted) {
            $("#tts-api-key").value = saved.elevenlabs_api_key_decrypted;
        }
        renderTtsModelOptions(
            [],
            saved.elevenlabs_model_id || "eleven_multilingual_v2"
        );
        $("#tts-output-format").value = saved.elevenlabs_output_format || "mp3_44100_128";
        $("#tts-output-dir").value = saved.elevenlabs_output_dir || "";
        savedTtsVoiceIds = saved.elevenlabs_speaker_voice_ids || {};
        const voiceSettings = saved.elevenlabs_voice_settings || {};
        $("#tts-stability").value = voiceSettings.stability ?? 0.5;
        $("#tts-similarity-boost").value = voiceSettings.similarity_boost ?? 0.75;
        $("#tts-style").value = voiceSettings.style ?? 0;
        $("#tts-speed").value = voiceSettings.speed ?? 1;
        $("#tts-use-speaker-boost").checked =
            voiceSettings.use_speaker_boost !== false;
        const chunkSettings = saved.elevenlabs_chunk_settings || {};
        $("#tts-max-chars-per-chunk").value =
            chunkSettings.max_chars_per_chunk ?? 1200;
        $("#tts-max-segments-per-chunk").value =
            chunkSettings.max_segments_per_chunk ?? 8;
        $("#tts-split-on-speaker-change").checked =
            chunkSettings.split_on_speaker_change === true;
        syncTtsGenerationMode();

        const exo = saved.exo;
        if (exo) {
            if (exo.font) $("#exo-font").value = exo.font;
            if (exo.font_size != null) $("#exo-font-size").value = exo.font_size;
            if (exo.spacing_x != null) $("#exo-spacing-x").value = exo.spacing_x;
            if (exo.spacing_y != null) $("#exo-spacing-y").value = exo.spacing_y;
            if (exo.display_speed != null) $("#exo-display-speed").value = exo.display_speed;
            if (exo.align != null) $("#exo-align").value = exo.align;
            if (exo.pos_x != null) $("#exo-pos-x").value = exo.pos_x;
            if (exo.pos_y != null) $("#exo-pos-y").value = exo.pos_y;
            if (exo.max_chars_per_line != null) $("#exo-max-chars").value = exo.max_chars_per_line;
            $("#exo-bold").checked = !!exo.bold;
            $("#exo-italic").checked = !!exo.italic;
            $("#exo-soft-edge").checked = exo.soft_edge !== false;

            if (exo.speaker_colors) {
                exoDefaults = exoDefaults || {};
                exoDefaults.speaker_colors = exo.speaker_colors;
            }
            if (exo.speaker_edge_colors && exo.speaker_edge_colors.length > 0) {
                exoDefaults = exoDefaults || {};
                exoDefaults.speaker_edge_colors = exo.speaker_edge_colors;
            }
        }

        updateHfTokenVisibility();
        renderSpeakerColors();

        if (exo?.speaker_images?.length > 0) {
            tachieData = exo.speaker_images.map(img => ({
                file: img.file || "",
                x: img.x || 0,
                y: img.y || 0,
                scale: img.scale || 100,
            }));
        }
        renderSpeakerTachie();

        if (exo?.background_image) {
            backgroundImage = exo.background_image;
            const name = backgroundImage.split(/[\\/]/).pop();
            $("#bg-image-name").textContent = name;
        }

        if (exo?.speaker_edge_colors?.length > 0) {
            document.querySelectorAll(".speaker-edge-hex").forEach((el, i) => {
                if (i < exo.speaker_edge_colors.length) {
                    el.value = exo.speaker_edge_colors[i];
                    const picker = document.querySelector(`.speaker-edge-color[data-index="${i}"]`);
                    if (picker) picker.value = "#" + exo.speaker_edge_colors[i];
                }
            });
        }

        setupAutoSave();
    } catch (e) {
        console.error("設定読み込みエラー:", e);
        setupAutoSave();
    }
}

function collectAllSettings() {
    return {
        model_size: $("#model-size").value,
        language: $("#language").value,
        num_speakers: $("#num-speakers").value,
        output_format: "exo",
        diarization_method: $("#diarization-method").value,
        hf_token: $("#hf-token").value || "",
        exo: collectExoSettings(),
    };
}

let saveTimer = null;
function scheduleAutoSave() {
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(async () => {
        try {
            await pywebview.api.save_settings(collectAllSettings());
        } catch (e) { /* ignore */ }
    }, 500);
}

function setupAutoSave() {
    // モーダル内の設定
    for (const id of ["model-size", "language", "num-speakers", "diarization-method"]) {
        $(`#${id}`).addEventListener("change", () => {
            if (id === "diarization-method") updateHfTokenVisibility();
            if (id === "num-speakers") {
                renderSpeakerColors();
                renderSpeakerTachie();
            }
            scheduleAutoSave();
        });
    }
    $("#hf-token").addEventListener("change", scheduleAutoSave);

    // exo設定
    const exoInputs = [
        "exo-font", "exo-font-size", "exo-spacing-x", "exo-spacing-y",
        "exo-display-speed", "exo-align", "exo-pos-x", "exo-pos-y",
        "exo-max-chars",
    ];
    for (const id of exoInputs) {
        $(`#${id}`).addEventListener("change", () => { markDirty(); scheduleAutoSave(); schedulePreviewRedraw(); });
    }
    for (const id of ["exo-bold", "exo-italic", "exo-soft-edge"]) {
        $(`#${id}`).addEventListener("change", () => { markDirty(); scheduleAutoSave(); schedulePreviewRedraw(); });
    }

    // 話者色・立ち絵
    $("#speaker-colors-list").addEventListener("input", () => { markDirty(); scheduleAutoSave(); schedulePreviewRedraw(); });
    $("#speaker-tachie-list").addEventListener("input", () => { markDirty(); scheduleAutoSave(); schedulePreviewRedraw(); });

    // Undo トリガー設定
    setupExoUndoListeners();
}

// --- ユーティリティ ---
function formatBytes(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
    if (bytes < 1073741824) return (bytes / 1048576).toFixed(1) + " MB";
    return (bytes / 1073741824).toFixed(2) + " GB";
}

// ============================================================
// exo シーンプレビュー
// ============================================================

let previewSegments = [];
let previewIndex = 0;

async function initExoPreview() {
    try {
        const mapping = Object.keys(currentMapping).length > 0 ? currentMapping : null;
        const res = await pywebview.api.get_preview_segments(mapping);
        if (!res || !res.success) return;

        previewSegments = res.segments;
        previewIndex = Math.min(previewIndex, Math.max(0, previewSegments.length - 1));

        // プレースホルダーを消してプレビュー表示
        hide($("#preview-placeholder"));
        await renderPreviewImage();
        updatePreviewNav();
        populateSegmentEditor();
        renderSegmentTable();
    } catch (e) {
        console.error("プレビュー初期化エラー:", e);
    }
}

async function renderPreviewImage() {
    const img = $("#preview-image");
    if (previewSegments.length === 0) return;

    try {
        const settings = collectExoSettings();
        const res = await pywebview.api.render_preview_frame(previewIndex, settings);
        if (res && res.success) {
            img.src = res.data_url;
        }
    } catch (e) {
        console.error("プレビューレンダリングエラー:", e);
    }
}

function navigatePreview(delta) {
    const newIdx = previewIndex + delta;
    if (newIdx < 0 || newIdx >= previewSegments.length) return;
    _segTextEditStarted = false;
    clearTimeout(_segTextDebounceTimer);
    previewIndex = newIdx;
    renderPreviewImage();
    updatePreviewNav();
    populateSegmentEditor();
    highlightSegmentTableRow();
}

function updatePreviewNav() {
    const total = previewSegments.length;
    const idx = previewIndex;
    $("#preview-seg-info").textContent = total > 0 ? `${idx + 1} / ${total}` : "- / -";
    $("#btn-prev-seg").disabled = idx <= 0;
    $("#btn-next-seg").disabled = idx >= total - 1;
}

function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function formatTimeDetailed(seconds) {
    const m = Math.floor(seconds / 60);
    const s = (seconds % 60).toFixed(1);
    return `${String(m).padStart(2, "0")}:${s.padStart(4, "0")}`;
}

function getSpeakerIndex(speakerName) {
    const match = speakerName?.match(/Speaker (\d+)/);
    return match ? parseInt(match[1]) - 1 : 0;
}

function schedulePreviewRedraw() {
    if (previewSegments.length === 0) return;
    renderPreviewImage();
}

// ============================================================
// セグメント一覧テーブル
// ============================================================

function renderSegmentTable() {
    const tbody = $("#segment-table-body");
    const empty = $("#segment-table-empty");
    const colors = exoDefaults?.speaker_colors || DEFAULT_SPEAKER_COLORS;

    if (previewSegments.length === 0) {
        tbody.innerHTML = "";
        show(empty);
        return;
    }

    hide(empty);
    tbody.innerHTML = "";

    previewSegments.forEach((seg, i) => {
        const tr = document.createElement("tr");
        if (i === previewIndex) tr.classList.add("active");

        const spkIdx = getSpeakerIndex(seg.speaker);
        const color = colors[spkIdx % colors.length];

        tr.innerHTML = `
            <td class="col-time">${formatTimeDetailed(seg.start)} → ${formatTimeDetailed(seg.end)}</td>
            <td class="col-speaker"><span style="color:#${color}">●</span> ${seg.speaker}</td>
            <td class="col-text">${escapeHtml(seg.text)}</td>
        `;

        tr.addEventListener("click", () => {
            previewIndex = i;
            renderPreviewImage();
            updatePreviewNav();
            populateSegmentEditor();
            highlightSegmentTableRow();
        });

        tbody.appendChild(tr);
    });
}

function highlightSegmentTableRow() {
    const rows = document.querySelectorAll("#segment-table-body tr");
    rows.forEach((tr, i) => {
        tr.classList.toggle("active", i === previewIndex);
    });

    // アクティブ行をスクロールに入れる
    const activeRow = document.querySelector("#segment-table-body tr.active");
    if (activeRow) {
        activeRow.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================
// セグメント編集
// ============================================================

function populateSegmentEditor() {
    const editor = $("#seg-editor");
    if (previewSegments.length === 0) {
        hide(editor);
        hide($("#seg-tts-status-panel"));
        selectedTtsStatusRequestId += 1;
        return;
    }
    show(editor);
    show($("#seg-tts-status-panel"));
    const seg = previewSegments[previewIndex];

    const select = $("#seg-edit-speaker");
    const speakers = getKnownSpeakers();
    select.innerHTML = "";
    for (const spk of speakers) {
        const opt = document.createElement("option");
        opt.value = spk;
        opt.textContent = spk;
        if (spk === seg.speaker) opt.selected = true;
        select.appendChild(opt);
    }

    $("#seg-edit-start").value = seg.start.toFixed(2);
    $("#seg-edit-end").value = seg.end.toFixed(2);
    $("#seg-edit-text").value = seg.text;

    const curSpeaker = seg.speaker || "Speaker 1";
    const prevSeg = previewIndex > 0 ? previewSegments[previewIndex - 1] : null;
    const nextSeg = previewIndex < previewSegments.length - 1 ? previewSegments[previewIndex + 1] : null;
    $("#btn-seg-merge-prev").disabled = !(prevSeg && (prevSeg.speaker || "Speaker 1") === curSpeaker);
    $("#btn-seg-merge-next").disabled = !(nextSeg && (nextSeg.speaker || "Speaker 1") === curSpeaker);
    scheduleSelectedTtsStatusRefresh();
}

function getKnownSpeakers() {
    const set = new Set();
    for (const seg of previewSegments) {
        set.add(seg.speaker || "Speaker 1");
    }
    const numValue = $("#num-speakers").value;
    const numSpeakers = numValue === "auto" ? 2 : parseInt(numValue);
    for (let i = 1; i <= Math.max(numSpeakers, set.size); i++) {
        set.add(`Speaker ${i}`);
    }
    return [...set].sort();
}

async function applySegmentEdit(skipUndo = false) {
    if (previewSegments.length === 0) return;
    if (!skipUndo) pushUndo();
    const speaker = $("#seg-edit-speaker").value;
    const text = $("#seg-edit-text").value;
    const start = parseFloat($("#seg-edit-start").value);
    const end = parseFloat($("#seg-edit-end").value);

    try {
        const res = await pywebview.api.update_segment(
            previewIndex, speaker, text, start, end
        );
        if (res && res.success) {
            handleSegmentEditResponse(res);
        } else {
            alert("更新エラー: " + (res?.error || "不明"));
        }
    } catch (e) {
        console.error("セグメント更新エラー:", e);
    }
}

async function addSegment() {
    pushUndo();
    let defaultStart = 0;
    let defaultEnd = 1;
    if (previewSegments.length > 0) {
        const cur = previewSegments[previewIndex];
        defaultStart = cur.end;
        defaultEnd = cur.end + 2.0;
    }

    const startStr = prompt("開始時刻（秒）", defaultStart.toFixed(2));
    if (startStr === null) return;
    const endStr = prompt("終了時刻（秒）", defaultEnd.toFixed(2));
    if (endStr === null) return;
    const text = prompt("テキスト", "");
    if (text === null) return;

    const start = parseFloat(startStr);
    const end = parseFloat(endStr);
    if (isNaN(start) || isNaN(end)) {
        alert("無効な時刻です");
        return;
    }

    const speakers = getKnownSpeakers();
    const speaker = speakers[0] || "Speaker 1";

    try {
        const res = await pywebview.api.add_segment(start, end, text, speaker);
        if (res && res.success) {
            handleSegmentEditResponse(res);
            if (res.inserted_index != null) {
                previewIndex = res.inserted_index;
            }
            renderPreviewImage();
            updatePreviewNav();
            populateSegmentEditor();
            renderSegmentTable();
        } else {
            alert("追加エラー: " + (res?.error || "不明"));
        }
    } catch (e) {
        console.error("セグメント追加エラー:", e);
    }
}

async function deleteSegment() {
    if (previewSegments.length <= 1) {
        alert("最後のセグメントは削除できません");
        return;
    }
    if (!confirm(`セグメント ${previewIndex + 1} を削除しますか？`)) return;
    pushUndo();

    try {
        const res = await pywebview.api.delete_segment(previewIndex);
        if (res && res.success) {
            handleSegmentEditResponse(res);
            if (previewIndex >= previewSegments.length) {
                previewIndex = previewSegments.length - 1;
            }
            renderPreviewImage();
            updatePreviewNav();
            populateSegmentEditor();
            renderSegmentTable();
        } else {
            alert("削除エラー: " + (res?.error || "不明"));
        }
    } catch (e) {
        console.error("セグメント削除エラー:", e);
    }
}

async function mergePrevSegment() {
    if (previewIndex <= 0) return;
    pushUndo();
    try {
        const res = await pywebview.api.merge_segments(previewIndex - 1);
        if (res && res.success) {
            previewIndex = res.merged_index;
            handleSegmentEditResponse(res);
        } else {
            alert("結合エラー: " + (res?.error || "不明"));
        }
    } catch (e) {
        console.error("セグメント結合エラー:", e);
    }
}

async function mergeNextSegment() {
    if (previewIndex >= previewSegments.length - 1) return;
    pushUndo();
    try {
        const res = await pywebview.api.merge_segments(previewIndex);
        if (res && res.success) {
            previewIndex = res.merged_index;
            handleSegmentEditResponse(res);
        } else {
            alert("結合エラー: " + (res?.error || "不明"));
        }
    } catch (e) {
        console.error("セグメント結合エラー:", e);
    }
}

async function playSegmentAudio() {
    if (previewSegments.length === 0) return;
    try {
        const res = await pywebview.api.play_segment_audio(previewIndex);
        if (res && !res.success) {
            console.warn("音声再生エラー:", res.error);
        }
    } catch (e) {
        console.error("音声再生エラー:", e);
    }
}

// --- ElevenLabs TTS（メインのセグメント編集状態から独立） ---
const TTS_STATUS_LABELS = {
    checking: "確認中",
    not_generated: "未生成",
    generated: "生成済み",
    needs_regeneration: "要再生成",
    error: "エラー",
};

function renderSelectedTtsStatus(item, fallbackStatus = "checking") {
    if (previewSegments.length === 0) {
        hide($("#seg-tts-status-panel"));
        selectedTtsStatusItem = null;
        return;
    }
    show($("#seg-tts-status-panel"));
    selectedTtsStatusItem = item || null;
    const status = item?.status || fallbackStatus;
    const badge = $("#seg-tts-status");
    badge.className = "seg-tts-status-badge tts-status-" + (
        status === "checking" ? "not_generated" : status
    );
    badge.textContent = TTS_STATUS_LABELS[status] || status;

    let detail = "";
    if (item?.chunk_index != null) {
        detail = "チャンク " + String(item.chunk_index + 1);
    }
    if (item?.error) detail = item.error;
    $("#seg-tts-status-detail").textContent = detail;
    $("#btn-seg-tts-play").disabled = (
        ttsBusy || !item?.audio_path
    );
    $("#btn-seg-tts-regenerate").disabled = ttsBusy;
}

function updateSelectedTtsPanelFromStatuses(statuses) {
    const item = (statuses || []).find(
        status => Number(status.index) === previewIndex
    );
    renderSelectedTtsStatus(item, item ? item.status : "not_generated");
}

async function refreshSelectedTtsStatus() {
    if (previewSegments.length === 0) {
        renderSelectedTtsStatus(null, "not_generated");
        return;
    }
    const requestId = ++selectedTtsStatusRequestId;
    const requestedIndex = previewIndex;
    renderSelectedTtsStatus(null, "checking");
    try {
        const result = await pywebview.api.get_tts_status(
            collectTtsSettings()
        );
        if (
            requestId !== selectedTtsStatusRequestId
            || requestedIndex !== previewIndex
        ) {
            return;
        }
        if (result?.success) {
            updateSelectedTtsPanelFromStatuses(result.segments);
        } else {
            renderSelectedTtsStatus(
                { status: "error", error: result?.error || "状態取得失敗" },
                "error"
            );
        }
    } catch (e) {
        if (
            requestId === selectedTtsStatusRequestId
            && requestedIndex === previewIndex
        ) {
            renderSelectedTtsStatus(
                { status: "error", error: String(e) },
                "error"
            );
        }
    }
}

function scheduleSelectedTtsStatusRefresh() {
    if (selectedTtsStatusTimer) clearTimeout(selectedTtsStatusTimer);
    selectedTtsStatusTimer = setTimeout(
        refreshSelectedTtsStatus, 150
    );
}

function collectTtsSettings() {
    const voiceIds = { ...savedTtsVoiceIds };
    document.querySelectorAll(".tts-voice-id").forEach(input => {
        voiceIds[input.dataset.speaker] = input.value.trim();
    });
    savedTtsVoiceIds = voiceIds;
    return {
        api_key: $("#tts-api-key").value.trim(),
        model_id: $("#tts-model-id").value.trim() || "eleven_multilingual_v2",
        output_format: $("#tts-output-format").value.trim() || "mp3_44100_128",
        speaker_voice_ids: voiceIds,
        output_dir: $("#tts-output-dir").value.trim(),
        voice_settings: {
            stability: Number($("#tts-stability").value),
            similarity_boost: Number($("#tts-similarity-boost").value),
            style: Number($("#tts-style").value),
            use_speaker_boost: $("#tts-use-speaker-boost").checked,
            speed: Number($("#tts-speed").value),
        },
        apply_language_text_normalization: false,
        apply_text_normalization: "auto",
        generation_mode: $("#tts-generation-mode").value,
        chunk_settings: {
            max_chars_per_chunk: Number(
                $("#tts-max-chars-per-chunk").value
            ),
            max_segments_per_chunk: Number(
                $("#tts-max-segments-per-chunk").value
            ),
            split_on_speaker_change:
                $("#tts-split-on-speaker-change").checked,
        },
    };
}

function scheduleTtsSettingsSave() {
    if (ttsSaveTimer) clearTimeout(ttsSaveTimer);
    ttsSaveTimer = setTimeout(async () => {
        try {
            await pywebview.api.save_tts_settings(collectTtsSettings());
        } catch (e) {
            console.error("TTS設定保存エラー:", e);
        }
    }, 500);
}

function renderTtsVoiceSettings() {
    const speakers = [...new Set(
        previewSegments.map(seg => seg.speaker || "Speaker 1")
    )].sort();
    const container = $("#tts-voice-settings");
    container.innerHTML = "";
    for (const speaker of speakers) {
        const item = document.createElement("div");
        item.className = "setting-item";
        const label = document.createElement("label");
        label.textContent = speaker + " voice";
        const row = document.createElement("div");
        row.className = "tts-voice-select-row";
        const select = document.createElement("select");
        select.className = "tts-voice-id";
        select.dataset.speaker = speaker;
        const selectedVoiceId = savedTtsVoiceIds[speaker] || "";
        const emptyOption = document.createElement("option");
        emptyOption.value = "";
        emptyOption.textContent = "ボイスを選択";
        select.appendChild(emptyOption);
        let selectedFound = false;
        for (const voice of availableTtsVoices) {
            const option = document.createElement("option");
            option.value = voice.voice_id;
            option.textContent = voice.name;
            if (voice.voice_id === selectedVoiceId) selectedFound = true;
            select.appendChild(option);
        }
        if (selectedVoiceId && !selectedFound) {
            const option = document.createElement("option");
            option.value = selectedVoiceId;
            option.textContent = selectedVoiceId + "（保存済み）";
            select.appendChild(option);
        }
        select.value = selectedVoiceId;
        const previewButton = document.createElement("button");
        previewButton.type = "button";
        previewButton.className = "btn btn-small";
        previewButton.textContent = "試聴";
        previewButton.title = "ボイス試聴";
        const updatePreviewState = () => {
            const voice = availableTtsVoices.find(
                item => item.voice_id === select.value
            );
            previewButton.disabled = !voice?.preview_url;
        };
        select.addEventListener("change", async () => {
            savedTtsVoiceIds[speaker] = select.value;
            updatePreviewState();
            scheduleTtsSettingsSave();
            await refreshTtsStatus();
        });
        previewButton.addEventListener("click", () => {
            previewTtsVoice(select.value);
        });
        updatePreviewState();
        row.append(select, previewButton);
        item.append(label, row);
        container.appendChild(item);
    }
}

function renderTtsModelOptions(models, selectedModelId) {
    availableTtsModels = models;
    const select = $("#tts-model-id");
    select.innerHTML = "";
    const selected = selectedModelId || "eleven_multilingual_v2";
    let selectedFound = false;
    for (const model of models) {
        const option = document.createElement("option");
        option.value = model.model_id;
        option.textContent = model.name;
        if (model.model_id === selected) selectedFound = true;
        select.appendChild(option);
    }
    if (!selectedFound) {
        const option = document.createElement("option");
        option.value = selected;
        option.textContent = selected;
        select.appendChild(option);
    }
    select.value = selected;
    syncTtsGenerationMode();
}

function syncTtsGenerationMode() {
    const modelId = $("#tts-model-id").value;
    const isV3 = modelId === "eleven_v3";
    const isMultilingual = modelId === "eleven_multilingual_v2";
    $("#tts-generation-mode").value = isV3
        ? "chunk_v3"
        : "per_segment_context";
    $("#tts-v3-warning").classList.toggle("hidden", !isV3);
    $("#tts-multilingual-info").classList.toggle(
        "hidden", !isMultilingual
    );
    $("#tts-chunk-settings").classList.toggle("hidden", !isV3);
    $("#btn-generate-tts-selected").textContent = isV3
        ? "選択チャンクTTS生成"
        : "選択セグメントTTS生成";
}

async function fetchTtsVoices() {
    const button = $("#btn-fetch-tts-voices");
    button.disabled = true;
    const originalText = button.textContent;
    button.textContent = "取得中...";
    try {
        const currentVoiceIds = {};
        document.querySelectorAll(".tts-voice-id").forEach(select => {
            currentVoiceIds[select.dataset.speaker] = select.value;
        });
        const result = await pywebview.api.list_elevenlabs_voices(
            $("#tts-api-key").value.trim() || null
        );
        if (!result?.success) {
            alert("ボイス一覧取得エラー: " + (result?.error || "不明"));
            return;
        }
        availableTtsVoices = result.voices || [];
        savedTtsVoiceIds = currentVoiceIds;
        renderTtsVoiceSettings();
    } catch (e) {
        alert("ボイス一覧取得エラー: " + e);
    } finally {
        button.disabled = false;
        button.textContent = originalText;
    }
}

async function fetchTtsModels() {
    const button = $("#btn-fetch-tts-models");
    button.disabled = true;
    const originalText = button.textContent;
    button.textContent = "取得中...";
    try {
        const selected = $("#tts-model-id").value
            || "eleven_multilingual_v2";
        const result = await pywebview.api.list_elevenlabs_models(
            $("#tts-api-key").value.trim() || null
        );
        if (!result?.success) {
            alert("モデル一覧取得エラー: " + (result?.error || "不明"));
            return;
        }
        const models = result.models || [];
        const selectedIsAvailable = models.some(
            model => model.model_id === selected
        );
        const defaultIsAvailable = models.some(
            model => model.model_id === "eleven_multilingual_v2"
        );
        const nextSelected = selectedIsAvailable
            ? selected
            : defaultIsAvailable
                ? "eleven_multilingual_v2"
                : (models[0]?.model_id || "eleven_multilingual_v2");
        renderTtsModelOptions(models, nextSelected);
        scheduleTtsSettingsSave();
        await refreshTtsStatus();
    } catch (e) {
        alert("モデル一覧取得エラー: " + e);
    } finally {
        button.disabled = false;
        button.textContent = originalText;
    }
}

function previewTtsVoice(voiceId) {
    const voice = availableTtsVoices.find(
        item => item.voice_id === voiceId
    );
    if (!voice?.preview_url) return;
    if (ttsVoicePreviewAudio) ttsVoicePreviewAudio.pause();
    ttsVoicePreviewAudio = new Audio(voice.preview_url);
    ttsVoicePreviewAudio.play().catch(error => {
        alert("ボイス試聴エラー: " + error);
    });
}

async function openTtsModal() {
    if (previewSegments.length === 0) return;
    ttsSelectedIndex = Math.min(previewIndex, previewSegments.length - 1);
    renderTtsVoiceSettings();
    syncTtsGenerationMode();
    for (const id of [
        "tts-api-key", "tts-model-id", "tts-output-format", "tts-output-dir",
        "tts-stability", "tts-similarity-boost", "tts-style", "tts-speed",
        "tts-use-speaker-boost",
        "tts-max-chars-per-chunk", "tts-max-segments-per-chunk",
        "tts-split-on-speaker-change",
    ]) {
        const input = $("#" + id);
        if (!input.dataset.ttsSaveBound) {
            input.addEventListener("change", async () => {
                if (id === "tts-model-id") syncTtsGenerationMode();
                scheduleTtsSettingsSave();
                if (id !== "tts-api-key" && id !== "tts-output-dir") {
                    await refreshTtsStatus();
                }
            });
            input.dataset.ttsSaveBound = "1";
        }
    }
    show($("#tts-modal"));
    await refreshTtsStatus();
}

function closeTtsModal() {
    if (ttsBusy) return;
    if (ttsVoicePreviewAudio) {
        ttsVoicePreviewAudio.pause();
        ttsVoicePreviewAudio = null;
    }
    scheduleTtsSettingsSave();
    hide($("#tts-modal"));
}

async function selectTtsOutputDir() {
    try {
        const path = await pywebview.api.select_tts_output_dir();
        if (path) {
            $("#tts-output-dir").value = path;
            scheduleTtsSettingsSave();
            await refreshTtsStatus();
        }
    } catch (e) {
        alert("出力フォルダの選択に失敗しました: " + e);
    }
}

async function saveTtsScript(formatType) {
    try {
        const result = await pywebview.api.save_tts_script(
            formatType, collectTtsSettings()
        );
        if (result && !result.success && result.error !== "キャンセルされました") {
            alert("台本保存エラー: " + result.error);
        }
    } catch (e) {
        alert("台本保存エラー: " + e);
    }
}

async function saveCombinedTtsWav() {
    if (ttsBusy) return;
    setTtsBusy(true);
    hide($("#btn-cancel-tts"));
    updateTtsProgress({ progress: 0, message: "生成済みTTS音声を結合中..." });
    try {
        const result = await pywebview.api.save_combined_tts_wav(
            collectTtsSettings()
        );
        if (!result?.success) {
            if (result?.error !== "キャンセルされました") {
                alert("結合WAV保存エラー: " + (result?.error || "不明なエラー"));
            }
            return;
        }
        const skipped = result.skipped || [];
        let message = `結合WAVを保存しました（${result.included}件）\n${result.path}`;
        if (skipped.length) {
            const labels = skipped.map(item =>
                `${item.kind === "chunk" ? "チャンク" : "セグメント"} ${item.index + 1}`
            );
            message += `\n\n未生成・要再生成などのためスキップ: ${labels.join(", ")}`;
        }
        updateTtsProgress({ progress: 1, message: "結合WAVを保存しました" });
        alert(message);
    } catch (e) {
        alert("結合WAV保存エラー: " + e);
    } finally {
        setTtsBusy(false);
    }
}

function renderTtsTable(statuses) {
    const body = $("#tts-table-body");
    body.innerHTML = "";
    for (const item of statuses) {
        const row = document.createElement("tr");
        row.dataset.index = item.index;
        if (item.index === ttsSelectedIndex) row.classList.add("active");
        row.addEventListener("click", () => {
            ttsSelectedIndex = item.index;
            document.querySelectorAll("#tts-table-body tr").forEach(tr => {
                tr.classList.toggle(
                    "active", Number(tr.dataset.index) === ttsSelectedIndex
                );
            });
        });
        const indexCell = document.createElement("td");
        indexCell.textContent = String(item.index + 1);
        const speakerCell = document.createElement("td");
        speakerCell.textContent = item.speaker;
        const textCell = document.createElement("td");
        textCell.className = "tts-text-preview";
        textCell.textContent = item.text.slice(0, 80);
        textCell.title = item.text;
        const statusCell = document.createElement("td");
        statusCell.className = "tts-status-" + item.status;
        statusCell.textContent = TTS_STATUS_LABELS[item.status] || item.status;
        if (item.error) statusCell.title = item.error;
        const actionsCell = document.createElement("td");
        actionsCell.className = "tts-col-actions";
        const playButton = document.createElement("button");
        playButton.className = "btn btn-small";
        playButton.textContent = "▶";
        playButton.title = "TTS再生";
        playButton.disabled = !item.audio_path;
        playButton.addEventListener("click", event => {
            event.stopPropagation();
            playTts(item.index);
        });
        const regenerateButton = document.createElement("button");
        regenerateButton.className = "btn btn-small";
        regenerateButton.textContent = "再生成";
        regenerateButton.addEventListener("click", event => {
            event.stopPropagation();
            generateOneTts(item.index);
        });
        actionsCell.append(playButton, regenerateButton);
        row.append(indexCell, speakerCell, textCell, statusCell, actionsCell);
        body.appendChild(row);
    }
}

async function refreshTtsStatus() {
    try {
        const result = await pywebview.api.get_tts_status(collectTtsSettings());
        if (result?.success) {
            renderTtsTable(result.segments);
            updateSelectedTtsPanelFromStatuses(result.segments);
        } else if (result?.error) {
            alert("TTS状態更新エラー: " + result.error);
        }
    } catch (e) {
        alert("TTS状態更新エラー: " + e);
    }
}

function setTtsBusy(busy) {
    ttsBusy = busy;
    document.querySelectorAll(".tts-action-row button").forEach(button => {
        button.disabled = busy;
    });
    $("#btn-cancel-tts").disabled = false;
    if (busy) {
        show($("#btn-cancel-tts"));
        show($("#tts-progress-area"));
    } else {
        hide($("#btn-cancel-tts"));
    }
    renderSelectedTtsStatus(
        selectedTtsStatusItem,
        selectedTtsStatusItem?.status || "checking"
    );
}

function updateTtsProgress(status) {
    if (!status) return;
    const value = Math.max(0, Math.min(1, status.progress || 0));
    $("#tts-progress-fill").style.width = String(value * 100) + "%";
    $("#tts-progress-text").textContent = status.message || "";
}

async function runTtsGeneration(call) {
    if (ttsBusy) return;
    setTtsBusy(true);
    const timer = setInterval(async () => {
        try {
            updateTtsProgress(await pywebview.api.get_tts_progress());
        } catch (e) { /* operation may be finishing */ }
    }, 300);
    try {
        const result = await call();
        updateTtsProgress(await pywebview.api.get_tts_progress());
        if (result && !result.success && !result.cancelled) {
            const detail = result.error
                || ("生成エラー: " + (result.errors || 0) + "件");
            alert(detail);
        }
    } catch (e) {
        alert("TTS生成エラー: " + e);
    } finally {
        clearInterval(timer);
        setTtsBusy(false);
        await refreshTtsStatus();
    }
}

async function generateAllTts() {
    const ttsSettings = collectTtsSettings();
    await runTtsGeneration(
        () => pywebview.api.generate_tts_for_all(ttsSettings, false)
    );
}

async function generateOneTts(index) {
    ttsSelectedIndex = index;
    const ttsSettings = collectTtsSettings();
    await runTtsGeneration(
        () => pywebview.api.generate_tts_for_segment(index, ttsSettings, true)
    );
}

async function playTts(index) {
    try {
        ttsSelectedIndex = index;
        const result = await pywebview.api.play_tts_segment(
            index, collectTtsSettings()
        );
        if (result && !result.success) {
            alert("TTS再生エラー: " + result.error);
        }
    } catch (e) {
        alert("TTS再生エラー: " + e);
    }
}

function handleSegmentEditResponse(res) {
    previewSegments = res.segments;
    markDirty();
    renderPreviewImage();
    updatePreviewNav();
    populateSegmentEditor();
    renderSegmentTable();
}

// ============================================================
// Undo / Redo
// ============================================================

function captureSnapshot() {
    return {
        segments: JSON.parse(JSON.stringify(previewSegments)),
        exoSettings: collectExoSettings(),
        currentMapping: JSON.parse(JSON.stringify(currentMapping)),
    };
}

function pushUndo() {
    const snapshot = captureSnapshot();
    undoStack.push(snapshot);
    if (undoStack.length > MAX_UNDO) undoStack.shift();
    redoStack = [];
    pendingExoSnapshot = null;
    updateUndoRedoUI();
}

async function undo() {
    if (undoStack.length === 0) return;
    redoStack.push(captureSnapshot());
    const snapshot = undoStack.pop();
    await applySnapshot(snapshot);
    updateUndoRedoUI();
}

async function redo() {
    if (redoStack.length === 0) return;
    undoStack.push(captureSnapshot());
    const snapshot = redoStack.pop();
    await applySnapshot(snapshot);
    updateUndoRedoUI();
}

async function applySnapshot(snapshot) {
    // 1. バックエンドのセグメントを復元
    if (snapshot.segments && snapshot.segments.length > 0) {
        try {
            const res = await pywebview.api.restore_segments(snapshot.segments);
            if (res && res.success) {
                previewSegments = res.segments;
            } else {
                previewSegments = snapshot.segments;
            }
        } catch (e) {
            console.error("セグメント復元エラー:", e);
            previewSegments = snapshot.segments;
        }
    } else {
        previewSegments = snapshot.segments || [];
    }

    // 2. 話者マッピングを復元
    currentMapping = snapshot.currentMapping || {};

    // 3. exo設定・色・立ち絵・背景を復元
    if (snapshot.exoSettings) {
        applyExoSettingsToUI(snapshot.exoSettings);
        backgroundImage = snapshot.exoSettings.background_image || "";
        if (snapshot.exoSettings.speaker_images?.length > 0) {
            tachieData = snapshot.exoSettings.speaker_images.map(img => ({
                file: img.file || "",
                x: img.x || 0,
                y: img.y || 0,
                scale: img.scale || 100,
            }));
        }
        renderSpeakerTachie();
    }

    // 4. UIを更新
    previewIndex = Math.min(previewIndex, Math.max(0, previewSegments.length - 1));
    renderPreviewImage();
    updatePreviewNav();
    populateSegmentEditor();
    renderSegmentTable();
}

function updateUndoRedoUI() {
    const undoBtn = $("#menu-undo");
    const redoBtn = $("#menu-redo");
    if (undoBtn) undoBtn.disabled = undoStack.length === 0;
    if (redoBtn) redoBtn.disabled = redoStack.length === 0;
}

function clearUndoHistory() {
    undoStack = [];
    redoStack = [];
    pendingExoSnapshot = null;
    updateUndoRedoUI();
}

// exo設定変更の Undo トリガー設定
function setupExoUndoListeners() {
    // focusin: スナップショット保存
    // change: undoスタックに積む
    const onFocusin = () => {
        if (!pendingExoSnapshot) {
            pendingExoSnapshot = captureSnapshot();
        }
    };
    const onChangePush = () => {
        if (pendingExoSnapshot) {
            undoStack.push(pendingExoSnapshot);
            if (undoStack.length > MAX_UNDO) undoStack.shift();
            redoStack = [];
            pendingExoSnapshot = null;
            updateUndoRedoUI();
        }
    };

    const exoSection = $("#exo-settings-section");
    if (exoSection) {
        exoSection.addEventListener("focusin", onFocusin);
        exoSection.addEventListener("change", onChangePush);
        // カラーピッカーはpointerdownで事前キャプチャ
        exoSection.addEventListener("pointerdown", onFocusin);
    }

    const colorsSection = $("#speaker-colors-section");
    if (colorsSection) {
        colorsSection.addEventListener("focusin", onFocusin);
        colorsSection.addEventListener("change", onChangePush);
        colorsSection.addEventListener("pointerdown", onFocusin);
    }

    const tachieSection = $("#speaker-tachie-section");
    if (tachieSection) {
        tachieSection.addEventListener("focusin", onFocusin);
        tachieSection.addEventListener("change", onChangePush);
    }

    const bgSection = $("#bg-section");
    if (bgSection) {
        bgSection.addEventListener("focusin", onFocusin);
    }
}
