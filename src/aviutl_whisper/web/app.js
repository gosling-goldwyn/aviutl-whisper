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

const DEFAULT_SPEAKER_COLORS = [
    "ffffff", "00ffff", "00ff00", "ff00ff",
    "ffff00", "ff8000", "8080ff", "80ff80",
];

// --- 初期化 ---
document.addEventListener("DOMContentLoaded", () => {
    initEventListeners();
    updateHfTokenVisibility();
    renderSpeakerColors();
    renderSpeakerTachie();
});

// pywebview API が準備完了してからAPI呼び出し
window.addEventListener("pywebviewready", async () => {
    await loadDeviceInfo();
    await loadExoDefaults();
    await loadFonts();
    await loadSavedSettings();
});

function initEventListeners() {
    $("#btn-select-file").addEventListener("click", selectFile);
    $("#btn-open-settings").addEventListener("click", openTranscriptionModal);
    $("#btn-close-modal").addEventListener("click", closeTranscriptionModal);
    $("#btn-modal-ok").addEventListener("click", closeTranscriptionModal);
    $("#btn-start").addEventListener("click", startTranscription);
    $("#btn-cancel").addEventListener("click", cancelTranscription);
    $("#btn-save").addEventListener("click", saveResult);
    $("#btn-bg-image").addEventListener("click", selectBackgroundImage);
    $("#btn-bg-image-clear").addEventListener("click", clearBackgroundImage);
    $("#btn-prev-seg").addEventListener("click", () => navigatePreview(-1));
    $("#btn-next-seg").addEventListener("click", () => navigatePreview(1));
    $("#btn-seg-apply").addEventListener("click", applySegmentEdit);
    $("#btn-seg-play").addEventListener("click", playSegmentAudio);
    $("#btn-seg-add").addEventListener("click", addSegment);
    $("#btn-seg-merge-prev").addEventListener("click", mergePrevSegment);
    $("#btn-seg-merge-next").addEventListener("click", mergeNextSegment);
    $("#btn-seg-delete").addEventListener("click", deleteSegment);
    $("#btn-load-project").addEventListener("click", loadProject);
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

    // キーボードナビゲーション
    document.addEventListener("keydown", (e) => {
        if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;
        if (previewSegments.length === 0) return;
        if (e.key === "ArrowLeft") { navigatePreview(-1); e.preventDefault(); }
        if (e.key === "ArrowRight") { navigatePreview(1); e.preventDefault(); }
    });
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
            backgroundImage = result;
            const name = result.split(/[\\/]/).pop();
            $("#bg-image-name").textContent = name;
            autoSave();
            schedulePreviewRedraw();
        }
    } catch (e) {
        console.error("背景画像選択エラー:", e);
    }
}

function clearBackgroundImage() {
    backgroundImage = "";
    $("#bg-image-name").textContent = "未選択";
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
async function saveProject() {
    try {
        const projectData = {
            source_file: selectedFile || "",
            exo_settings: collectExoSettings(),
            preview_index: previewIndex,
        };
        const result = await pywebview.api.save_project(projectData);
        if (result && result.success) {
            alert("プロジェクトを保存しました: " + result.path);
        } else if (result && result.error && result.error !== "キャンセルされました") {
            alert("保存エラー: " + result.error);
        }
    } catch (e) {
        alert("プロジェクト保存エラー: " + e);
    }
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

        // ファイル情報を復元
        selectedFile = result.source_file || null;
        if (selectedFile) {
            const name = selectedFile.split(/[\\/]/).pop();
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
            // テキスト更新は不要（テーブルで表示）
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
        $(`#${id}`).addEventListener("change", () => { scheduleAutoSave(); schedulePreviewRedraw(); });
    }
    for (const id of ["exo-bold", "exo-italic", "exo-soft-edge"]) {
        $(`#${id}`).addEventListener("change", () => { scheduleAutoSave(); schedulePreviewRedraw(); });
    }

    // 話者色・立ち絵
    $("#speaker-colors-list").addEventListener("input", () => { scheduleAutoSave(); schedulePreviewRedraw(); });
    $("#speaker-tachie-list").addEventListener("input", () => { scheduleAutoSave(); schedulePreviewRedraw(); });
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
        return;
    }
    show(editor);
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

async function applySegmentEdit() {
    if (previewSegments.length === 0) return;
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

function handleSegmentEditResponse(res) {
    previewSegments = res.segments;
    renderPreviewImage();
    updatePreviewNav();
    populateSegmentEditor();
    renderSegmentTable();
}
