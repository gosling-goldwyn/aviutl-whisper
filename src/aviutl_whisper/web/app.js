// aviutl-whisper フロントエンド

const $ = (sel) => document.querySelector(sel);
const show = (el) => el.classList.remove("hidden");
const hide = (el) => el.classList.add("hidden");

let selectedFile = null;
let isProcessing = false;

// --- 初期化 ---
document.addEventListener("DOMContentLoaded", async () => {
    initEventListeners();
    await loadDeviceInfo();
});

function initEventListeners() {
    $("#btn-select-file").addEventListener("click", selectFile);
    $("#btn-start").addEventListener("click", startTranscription);
    $("#btn-cancel").addEventListener("click", cancelTranscription);
    $("#btn-save").addEventListener("click", saveResult);
    $("#btn-copy").addEventListener("click", copyResult);
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

    $("#result-text").value = result.text;
    show($("#result-section"));
    setProgress(100, "完了！");
}

async function saveResult() {
    try {
        const format = $("#output-format").value;
        const result = await pywebview.api.save_result(format);
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

// --- ユーティリティ ---
function formatBytes(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
    if (bytes < 1073741824) return (bytes / 1048576).toFixed(1) + " MB";
    return (bytes / 1073741824).toFixed(2) + " GB";
}
