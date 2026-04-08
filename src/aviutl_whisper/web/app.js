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
    $("#output-format").addEventListener("change", updateExoSettingsVisibility);
    $("#diarization-method").addEventListener("change", updateHfTokenVisibility);
    $("#num-speakers").addEventListener("change", () => {
        renderSpeakerColors();
        renderSpeakerTachie();
    });
    updateExoSettingsVisibility();
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
    $("#btn-start").addEventListener("click", startTranscription);
    $("#btn-cancel").addEventListener("click", cancelTranscription);
    $("#btn-save").addEventListener("click", saveResult);
    $("#btn-copy").addEventListener("click", copyResult);
    $("#btn-bg-image").addEventListener("click", selectBackgroundImage);
    $("#btn-bg-image-clear").addEventListener("click", clearBackgroundImage);
    $("#btn-prev-seg").addEventListener("click", () => navigatePreview(-1));
    $("#btn-next-seg").addEventListener("click", () => navigatePreview(1));

    // キーボードナビゲーション（矢印キー）
    document.addEventListener("keydown", (e) => {
        if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;
        if ($("#exo-preview-area").classList.contains("hidden")) return;
        if (e.key === "ArrowLeft") { navigatePreview(-1); e.preventDefault(); }
        if (e.key === "ArrowRight") { navigatePreview(1); e.preventDefault(); }
    });
}

// --- exo設定パネルの表示/非表示 ---
function updateExoSettingsVisibility() {
    const exoSection = $("#exo-settings-section");
    if ($("#output-format").value === "exo") {
        show(exoSection);
        // 結果がある場合はプレビューも表示
        if (previewSegments.length > 0) {
            show($("#exo-preview-area"));
            drawPreviewFrame();
            updatePreviewNav();
        }
    } else {
        hide(exoSection);
        hide($("#exo-preview-area"));
    }
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

    // 既存データを維持しつつ、話者数に合わせてリサイズ
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

    // 画像選択ボタンのイベント
    container.querySelectorAll(".btn-tachie-select").forEach(btn => {
        btn.addEventListener("click", (e) => selectTachieImage(parseInt(e.target.dataset.index)));
    });

    // クリアボタンのイベント
    container.querySelectorAll(".btn-tachie-clear").forEach(btn => {
        btn.addEventListener("click", (e) => {
            const idx = parseInt(e.target.dataset.index);
            tachieData[idx].file = "";
            const label = container.querySelector(`.tachie-file-name[data-index="${idx}"]`);
            if (label) label.textContent = "未選択";
            scheduleAutoSave();
        });
    });

    // パラメータ入力のイベント
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
            await loadImageCached(path);
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
            // プレビュー用に画像をキャッシュして再描画
            await loadImageCached(result);
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
    hide($("#result-section"));

    setProgress(0, "処理を開始しています...");

    // 進捗ポーリング開始
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
            output_format: $("#output-format").value,
            diarization_method: $("#diarization-method").value,
            hf_token: $("#hf-token").value || "",
        };

        if (settings.output_format === "exo") {
            settings.exo_settings = collectExoSettings();
        }

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

    $("#result-text").value = result.text;
    show($("#result-section"));
    setProgress(100, "完了！");

    // 話者マッピングUI
    lastSpeakers = result.speakers || [];
    if (lastSpeakers.length > 1) {
        renderSpeakerMapping(lastSpeakers);
        show($("#speaker-mapping-area"));
    } else {
        hide($("#speaker-mapping-area"));
    }

    // exoプレビュー
    if ($("#output-format").value === "exo") {
        initExoPreview();
    } else {
        hide($("#exo-preview-area"));
    }
}

async function saveResult() {
    try {
        const format = $("#output-format").value;
        const exoSettings = format === "exo" ? collectExoSettings() : null;
        const mapping = Object.keys(currentMapping).length > 0 ? currentMapping : null;
        const result = await pywebview.api.save_result(format, exoSettings, mapping);
        if (result && result.success) {
            alert("保存しました: " + result.path);
        }
    } catch (e) {
        alert("保存エラー: " + e);
    }
}

async function copyResult() {
    const text = $("#result-text").value;
    try {
        await navigator.clipboard.writeText(text);
        const btn = $("#btn-copy");
        const original = btn.textContent;
        btn.textContent = "✅ コピーしました";
        setTimeout(() => { btn.textContent = original; }, 2000);
    } catch (e) {
        // フォールバック
        $("#result-text").select();
        document.execCommand("copy");
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

    // 再生ボタンのイベント
    container.querySelectorAll(".btn-play-sample").forEach(btn => {
        btn.addEventListener("click", () => playSpeakerSample(btn.dataset.speaker));
    });

    // セレクトボックスのイベント
    container.querySelectorAll(".mapping-select").forEach(sel => {
        sel.addEventListener("change", () => applyMapping());
    });

    // 2話者の場合は入れ替えボタンを表示
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

    // 色プレビュードットを更新
    const colors = exoDefaults?.speaker_colors || DEFAULT_SPEAKER_COLORS;
    selects.forEach(sel => {
        const slot = parseInt(sel.value);
        const row = sel.closest(".speaker-mapping-row");
        const dot = row.querySelector(".color-preview-dot");
        if (dot) dot.style.background = "#" + colors[slot % colors.length];
    });

    try {
        const format = $("#output-format").value;
        const exoSettings = format === "exo" ? collectExoSettings() : null;
        const result = await pywebview.api.remap_speakers(currentMapping, format, exoSettings);
        if (result && result.success) {
            $("#result-text").value = result.text;
        }
        // マッピング変更後にプレビュー再読み込み
        if (format === "exo") initExoPreview();
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

        // 基本設定の復元
        if (saved.model_size) $("#model-size").value = saved.model_size;
        if (saved.language) $("#language").value = saved.language;
        if (saved.num_speakers) $("#num-speakers").value = saved.num_speakers;
        if (saved.output_format) $("#output-format").value = saved.output_format;
        if (saved.diarization_method) $("#diarization-method").value = saved.diarization_method;
        if (saved.hf_token_decrypted) $("#hf-token").value = saved.hf_token_decrypted;

        // exo設定の復元
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

            // 話者色の復元はrenderSpeakerColorsで使うためexoDefaultsに反映
            if (exo.speaker_colors) {
                exoDefaults = exoDefaults || {};
                exoDefaults.speaker_colors = exo.speaker_colors;
            }
            if (exo.speaker_edge_colors && exo.speaker_edge_colors.length > 0) {
                exoDefaults = exoDefaults || {};
                exoDefaults.speaker_edge_colors = exo.speaker_edge_colors;
            }
        }

        // UI更新
        updateExoSettingsVisibility();
        updateHfTokenVisibility();
        renderSpeakerColors();

        // 立ち絵設定の復元
        if (exo?.speaker_images?.length > 0) {
            tachieData = exo.speaker_images.map(img => ({
                file: img.file || "",
                x: img.x || 0,
                y: img.y || 0,
                scale: img.scale || 100,
            }));
        }
        renderSpeakerTachie();

        // 背景画像の復元
        if (exo?.background_image) {
            backgroundImage = exo.background_image;
            const name = backgroundImage.split(/[\\/]/).pop();
            $("#bg-image-name").textContent = name;
        }

        // 話者色の復元（renderSpeakerColors後）
        if (exo?.speaker_edge_colors?.length > 0) {
            document.querySelectorAll(".speaker-edge-hex").forEach((el, i) => {
                if (i < exo.speaker_edge_colors.length) {
                    el.value = exo.speaker_edge_colors[i];
                    const picker = document.querySelector(`.speaker-edge-color[data-index="${i}"]`);
                    if (picker) picker.value = "#" + exo.speaker_edge_colors[i];
                }
            });
        }

        // 自動保存リスナーを登録
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
        output_format: $("#output-format").value,
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
    // 基本設定
    for (const id of ["model-size", "language", "num-speakers", "output-format", "diarization-method"]) {
        $(`#${id}`).addEventListener("change", () => {
            if (id === "output-format") updateExoSettingsVisibility();
            if (id === "diarization-method") updateHfTokenVisibility();
            if (id === "num-speakers") {
                renderSpeakerColors();
                renderSpeakerTachie();
            }
            scheduleAutoSave();
        });
    }
    // HFトークン
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
    // 話者色は動的なのでcontainerに委任
    $("#speaker-colors-list").addEventListener("input", () => { scheduleAutoSave(); schedulePreviewRedraw(); });
    // 立ち絵設定も動的
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
const previewImageCache = {};  // path → Image

async function initExoPreview() {
    const area = $("#exo-preview-area");
    try {
        const mapping = Object.keys(currentMapping).length > 0 ? currentMapping : null;
        const res = await pywebview.api.get_preview_segments(mapping);
        if (!res || !res.success) {
            hide(area);
            return;
        }
        previewSegments = res.segments;
        previewIndex = 0;
        show(area);
        await preloadPreviewImages();
        drawPreviewFrame();
        updatePreviewNav();
    } catch (e) {
        console.error("プレビュー初期化エラー:", e);
        hide(area);
    }
}

async function preloadPreviewImages() {
    const paths = new Set();
    if (backgroundImage) paths.add(backgroundImage);
    for (const td of tachieData) {
        if (td.file) paths.add(td.file);
    }
    const promises = [...paths].map(p => loadImageCached(p));
    await Promise.allSettled(promises);
}

async function loadImageCached(filePath) {
    if (previewImageCache[filePath]) return previewImageCache[filePath];
    try {
        const res = await pywebview.api.get_image_base64(filePath);
        if (!res || !res.success) return null;
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                previewImageCache[filePath] = img;
                resolve(img);
            };
            img.onerror = () => resolve(null);
            img.src = res.data_url;
        });
    } catch (e) {
        return null;
    }
}

function navigatePreview(delta) {
    const newIdx = previewIndex + delta;
    if (newIdx < 0 || newIdx >= previewSegments.length) return;
    previewIndex = newIdx;
    drawPreviewFrame();
    updatePreviewNav();
}

function updatePreviewNav() {
    const total = previewSegments.length;
    const idx = previewIndex;
    $("#preview-seg-info").textContent = `${idx + 1} / ${total}`;
    $("#btn-prev-seg").disabled = idx <= 0;
    $("#btn-next-seg").disabled = idx >= total - 1;

    if (total > 0 && previewSegments[idx]) {
        const seg = previewSegments[idx];
        const startStr = formatTime(seg.start);
        const endStr = formatTime(seg.end);
        const colors = exoDefaults?.speaker_colors || DEFAULT_SPEAKER_COLORS;
        const spkIdx = getSpeakerIndex(seg.speaker);
        const color = colors[spkIdx % colors.length];
        $("#preview-seg-detail").innerHTML =
            `<span style="color:#${color}">●</span> ${seg.speaker} ` +
            `<span style="color:var(--text-muted)">[${startStr} → ${endStr}]</span> ` +
            `${seg.text.substring(0, 60)}${seg.text.length > 60 ? "…" : ""}`;
    } else {
        $("#preview-seg-detail").textContent = "";
    }
}

function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function getSpeakerIndex(speakerName) {
    // "Speaker N" → N-1 (0-based)
    const match = speakerName?.match(/Speaker (\d+)/);
    return match ? parseInt(match[1]) - 1 : 0;
}

function drawPreviewFrame() {
    const canvas = $("#preview-canvas");
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;

    // クリア
    ctx.clearRect(0, 0, W, H);

    // 1. 背景
    drawBackground(ctx, W, H);

    // 2. 立ち絵
    if (previewSegments.length > 0) {
        const seg = previewSegments[previewIndex];
        drawTachie(ctx, W, H, seg.speaker);
    }

    // 3. 字幕テキスト
    if (previewSegments.length > 0) {
        const seg = previewSegments[previewIndex];
        drawSubtitle(ctx, W, H, seg);
    }
}

function drawBackground(ctx, W, H) {
    // 黒背景
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, W, H);

    // 背景画像
    if (backgroundImage && previewImageCache[backgroundImage]) {
        const img = previewImageCache[backgroundImage];
        // アスペクト比を維持してフィット（cover）
        const scale = Math.max(W / img.width, H / img.height);
        const dw = img.width * scale;
        const dh = img.height * scale;
        ctx.drawImage(img, (W - dw) / 2, (H - dh) / 2, dw, dh);
    }
}

function drawTachie(ctx, W, H, activeSpeaker) {
    const numValue = $("#num-speakers").value;
    const numSpeakers = numValue === "auto" ? 2 : parseInt(numValue);

    for (let i = 0; i < numSpeakers; i++) {
        const td = tachieData[i];
        if (!td || !td.file || !previewImageCache[td.file]) continue;

        const img = previewImageCache[td.file];
        const speakerName = `Speaker ${i + 1}`;
        const isActive = speakerName === activeSpeaker;
        const scale = (td.scale || 100) / 100;
        const dw = img.width * scale;
        const dh = img.height * scale;
        // AviUtlの座標系: 画面中央が(0,0)
        const dx = (W / 2) + (td.x || 0) - dw / 2;
        const dy = (H / 2) + (td.y || 0) - dh / 2;

        if (!isActive) {
            // グレースケール: オフスクリーンCanvasで彩度0を表現
            ctx.save();
            ctx.filter = "grayscale(100%)";
            ctx.drawImage(img, dx, dy, dw, dh);
            ctx.restore();
        } else {
            ctx.drawImage(img, dx, dy, dw, dh);
        }
    }
}

function drawSubtitle(ctx, W, H, seg) {
    const settings = collectExoSettings();
    const spkIdx = getSpeakerIndex(seg.speaker);
    const colors = settings.speaker_colors || DEFAULT_SPEAKER_COLORS;
    const textColor = "#" + (colors[spkIdx % colors.length] || "ffffff");
    const edgeColors = settings.speaker_edge_colors || [];
    const edgeColor = "#" + (edgeColors[spkIdx % (edgeColors.length || 1)] || "000000");

    const fontSize = settings.font_size || 34;
    const fontName = settings.font || "MS UI Gothic";
    const bold = settings.bold ? "bold " : "";
    const italic = settings.italic ? "italic " : "";
    ctx.font = `${italic}${bold}${fontSize}px "${fontName}", sans-serif`;

    // テキスト折り返し
    const maxChars = settings.max_chars_per_line || 0;
    const text = maxChars > 0 ? wrapText(seg.text, maxChars) : seg.text;
    const lines = text.split("\n");

    const lineHeight = fontSize + (settings.spacing_y || 0);
    const totalTextHeight = lines.length * lineHeight;

    // 寄せ方向 (align: 0-8, 3x3グリッド)
    const align = settings.align != null ? settings.align : 4;
    const colAlign = align % 3;  // 0=左, 1=中, 2=右
    const rowAlign = Math.floor(align / 3);  // 0=上, 1=中, 2=下

    ctx.textAlign = colAlign === 0 ? "left" : colAlign === 1 ? "center" : "right";

    // X基準点
    let baseX;
    if (colAlign === 0) baseX = 40;
    else if (colAlign === 1) baseX = W / 2;
    else baseX = W - 40;
    baseX += (settings.pos_x || 0);

    // Y基準点
    let baseY;
    if (rowAlign === 0) baseY = 40 + fontSize;
    else if (rowAlign === 1) baseY = (H - totalTextHeight) / 2 + fontSize;
    else baseY = H - 40 - totalTextHeight + fontSize;
    baseY += (settings.pos_y || 0);

    // 描画
    for (let i = 0; i < lines.length; i++) {
        const y = baseY + i * lineHeight;
        const line = lines[i];

        // 縁取り
        if (settings.soft_edge) {
            ctx.strokeStyle = edgeColor;
            ctx.lineWidth = Math.max(2, fontSize / 8);
            ctx.lineJoin = "round";
            ctx.strokeText(line, baseX, y);
        }

        // テキスト本体
        ctx.fillStyle = textColor;
        ctx.fillText(line, baseX, y);
    }
}

function wrapText(text, maxChars) {
    if (!maxChars || maxChars <= 0) return text;
    const result = [];
    for (const line of text.split("\n")) {
        for (let i = 0; i < line.length; i += maxChars) {
            result.push(line.substring(i, i + maxChars));
        }
        if (line.length === 0) result.push("");
    }
    return result.join("\n");
}

function schedulePreviewRedraw() {
    if ($("#exo-preview-area").classList.contains("hidden")) return;
    drawPreviewFrame();
}
