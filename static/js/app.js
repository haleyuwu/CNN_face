// ===== Upload SINGLE image =====
const form = document.getElementById("upload-form");
const input = document.getElementById("image-input");
const gallery = document.getElementById("gallery");
const statusEl = document.getElementById("upload-status");
const btnUpload = document.getElementById("btn-upload");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = input.files?.[0];
  if (!file) {
    statusEl.textContent = "Chưa chọn ảnh.";
    return;
  }
  statusEl.textContent = "Đang xử lý...";
  btnUpload.disabled = true;

  try {
    const fd = new FormData();
    fd.append("image", file);
    const res = await fetch("/predict", { method: "POST", body: fd });
    const data = await res.json();

    gallery.innerHTML = "";
    const card = document.createElement("div");
    card.className = "result";
    const img = document.createElement("img");
    img.src = URL.createObjectURL(file);
    img.className = "thumb";
    const p = document.createElement("p");
    p.className = "muted";
    const name = data.top_class_name ?? "?";
    const prob = data.top_class_prob != null ? (data.top_class_prob * 100).toFixed(1) : "?";
    p.textContent = `Kết quả: ${name} (${prob}%)`;
    card.appendChild(img);
    card.appendChild(p);
    gallery.appendChild(card);

    statusEl.textContent = "Xong.";
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Lỗi nhận diện.";
  } finally {
    btnUpload.disabled = false;
  }
});

// ===== Webcam realtime =====
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const btnStart = document.getElementById("btn-start");
const btnStop = document.getElementById("btn-stop");
const fpsEl = document.getElementById("fps");

let stream = null;
let loopId = null;
let running = false;
let busy = false;
let controller = null;
let lastSend = 0;
const targetFPS = 10;
const sendInterval = 1000 / targetFPS;

function syncCanvasSize() {
  const w = video.videoWidth || 640;
  const h = video.videoHeight || 480;
  overlay.width = w;
  overlay.height = h;
}

function startLoop() {
  function loop(now = 0) {
    if (!running) return;
    if (!busy && now - lastSend >= sendInterval) {
      busy = true;
      lastSend = now;
      sendFrameAndDraw().finally(() => (busy = false));
      fpsEl.textContent = `~${targetFPS} FPS`;
    }
    loopId = requestAnimationFrame(loop);
  }
  loop();
}

async function startWebcam() {
  if (running) return;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    video.srcObject = stream;
    await video.play();
    syncCanvasSize();
    window.addEventListener("resize", syncCanvasSize);
    running = true;
    startLoop();
  } catch (err) {
    console.error("Cannot start webcam", err);
  }
}

function stopWebcam() {
  running = false;
  if (loopId) cancelAnimationFrame(loopId);
  loopId = null;

  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;

  if (controller) controller.abort();
  controller = null;
  busy = false;

  ctx.clearRect(0, 0, overlay.width, overlay.height);
  fpsEl.textContent = "";
}

async function sendFrameAndDraw() {
  const maxW = 384; // giảm kích thước gửi -> nhanh
  const vw = video.videoWidth || 640;
  const vh = video.videoHeight || 480;
  const scale = Math.min(1, maxW / vw);
  const tw = Math.round(vw * scale);
  const th = Math.round(vh * scale);

  const tmp = document.createElement("canvas");
  tmp.width = tw;
  tmp.height = th;
  const tctx = tmp.getContext("2d");
  tctx.drawImage(video, 0, 0, tw, th);
  const dataURL = tmp.toDataURL("image/jpeg", 0.6);

  controller = new AbortController();
  try {
    const res = await fetch("/predict_frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataURL }),
      signal: controller.signal,
    });
    const data = await res.json();

    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Nếu backend không trả mảng bbox (faces), vẫn vẽ label top-1 ở góc
    if (!data.faces || !Array.isArray(data.faces) || data.faces.length === 0) {
      const label = data.top_class_name ? `${data.top_class_name} ${(data.top_class_prob * 100).toFixed(0)}%` : "";
      if (label) {
        ctx.fillStyle = "rgba(34,197,94,0.18)";
        ctx.fillRect(10, 10, 160, 28);
        ctx.fillStyle = "#fff";
        ctx.font = "14px system-ui, sans-serif";
        ctx.fillText(label, 16, 30);
      }
      return;
    }

    // map bbox từ kích thước gửi -> kích thước video thực
    const sx = overlay.width / tw;
    const sy = overlay.height / th;
    data.faces.forEach((f) => {
      const x = Math.round(f.x * sx);
      const y = Math.round(f.y * sy);
      const w = Math.round(f.w * sx);
      const h = Math.round(f.h * sy);

      ctx.lineWidth = 2;
      ctx.strokeStyle = "#22c55e";
      ctx.strokeRect(x, y, w, h);

      ctx.fillStyle = "rgba(34,197,94,0.18)";
      ctx.fillRect(x, y - 22, Math.max(120, w), 22);

      ctx.fillStyle = "#fff";
      ctx.font = "14px system-ui, sans-serif";
      const label = f.label ? `${f.label} ${(f.score * 100).toFixed(0)}%` : "";
      ctx.fillText(label, x + 6, y - 6);
    });
  } catch (err) {
    if (err.name !== "AbortError") console.error(err);
  } finally {
    controller = null;
  }
}

document.getElementById("btn-start").addEventListener("click", startWebcam);
document.getElementById("btn-stop").addEventListener("click", stopWebcam);
document.getElementById("video").addEventListener("loadedmetadata", syncCanvasSize);
