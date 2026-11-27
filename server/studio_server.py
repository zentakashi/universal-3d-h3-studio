#!/usr/bin/env python3
"""
Universal 3D x MiniMax H3 (Ref2VA / FL2VA / T2VA SD) One-Stop Studio Server for Google Colab L4
  - 任意の 3D モデル (.glb/.obj) や任意の参照画像・参照動画・参照音声で利用可能
  - /api/health: GPU・ComfyUI 稼働状況
  - /api/generate:
      * 参照画像 (最大9枚)
      * 参照動画 (最大3本) -> 自動 ffmpeg SD解像度・24fps・最大5秒リエンコード
      * 参照音声 (最大3本)
      * Ref2VA / FL2VA / T2VA モードに応じて ComfyUI ワークフローを動的構築＆投入
  - /api/status/<job_id>: 進捗ポーリング & 完成 MP4 URL 返却
  - /api/shutdown_runtime: google.colab.runtime.unassign() による安全な L4 ランタイム終了
"""

import json
import os
import subprocess
import threading
import time
import urllib.request
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import uvicorn

COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", "/content/ComfyUI"))
COMFY_INPUT = COMFY_ROOT / "input"
COMFY_OUTPUT = COMFY_ROOT / "output"
STUDIO_HTML = Path(__file__).resolve().parent.parent / "frontend" / "studio.html"

app = FastAPI(title="Universal 3D x MiniMax H3 Studio")
JOBS: dict[str, dict] = {}


def detect_gpu() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            timeout=3,
        ).strip()
        return out.splitlines()[0] if out else "NVIDIA GPU"
    except Exception:
        return "CPU"


def is_comfy_ready() -> bool:
    try:
        urllib.request.urlopen(f"{COMFY_URL}/system_stats", timeout=1.5)
        return True
    except Exception:
        return False


def find_model(subdir: str, keywords: list[str]) -> str | None:
    folder = COMFY_ROOT / "models" / subdir
    if not folder.exists():
        return None
    files = sorted(folder.glob("*.safetensors"))
    for kw in keywords:
        for f in files:
            if kw.lower() in f.name.lower():
                return f.name
    return files[0].name if files else None


ASPECT_SPECS: dict[str, dict[str, tuple[int, int]]] = {
    "16:9": {"gen": (1280, 704), "720p": (1280, 720), "1080p": (1920, 1080)},
    "9:16": {"gen": (704, 1280), "720p": (720, 1280), "1080p": (1080, 1920)},
    "4:3":  {"gen": (960, 704),  "720p": (960, 720),  "1080p": (1440, 1080)},
    "3:4":  {"gen": (704, 960),  "720p": (720, 960),  "1080p": (1080, 1440)},
    "1:1":  {"gen": (736, 736),  "720p": (720, 720),  "1080p": (1080, 1080)},
}


def resolve_video_spec(cfg: dict) -> tuple[int, int, int, int, int]:
    """Return (gen_w, gen_h, target_w, target_h, target_fps)."""
    aspect = str(cfg.get("aspect_ratio", "16:9"))
    tier = str(cfg.get("resolution_tier", "720p")).lower()
    if aspect not in ASPECT_SPECS:
        aspect = "16:9"
    if tier not in ("720p", "1080p"):
        tier = "720p"
    gen_w, gen_h = ASPECT_SPECS[aspect]["gen"]
    target_w, target_h = ASPECT_SPECS[aspect][tier]
    target_fps = int(cfg.get("fps", 24))
    if target_fps not in (24, 30, 60):
        target_fps = 24
    return gen_w, gen_h, target_w, target_h, target_fps


def master_output_video(raw_mp4: Path, target_w: int, target_h: int, target_fps: int) -> Path:
    """Scale to exact 720p/1080p and interpolate to 30/60fps when needed."""
    out_mp4 = raw_mp4.with_name(f"{raw_mp4.stem}_{target_w}x{target_h}_{target_fps}fps.mp4")
    vf_parts = [f"scale={target_w}:{target_h}:flags=lanczos"]
    if target_fps > 24:
        vf_parts.append(f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:vsbmc=1")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(raw_mp4),
            "-vf", ",".join(vf_parts),
            "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            str(out_mp4),
        ],
        check=True,
    )
    return out_mp4


def build_comfy_workflow(
    cfg: dict,
    saved_imgs: list[str],
    saved_vids: list[str],
    saved_auds: list[str],
) -> dict:
    mode = cfg.get("mode", "ref2va").lower()
    gen_w, gen_h, _, _, _ = resolve_video_spec(cfg)
    frames = int(cfg.get("frames", 124))
    steps = int(cfg.get("steps", 8))
    prompt_text = cfg.get("prompt", "")

    dit_kw = ["ref2va", "fl2va"] if mode == "ref2va" else ["fl2va", "ref2va"]
    dit_name = find_model("diffusion_models", dit_kw)
    te_name = find_model("text_encoders", ["qwen3vl", "minimax"])
    vvae_name = find_model("vae", ["video_vae"])
    avae_name = find_model("vae", ["audio_vae"])
    lora_name = find_model("loras", ["hyperflow", "turbo"])

    wf: dict[str, dict] = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": dit_name, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": te_name, "type": "minimax_h3"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vvae_name}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": avae_name}},
    }

    model_ref = ["1", 0]
    if lora_name and steps <= 10:
        wf["5"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["1", 0], "lora_name": lora_name, "strength_model": 1.0},
        }
        model_ref = ["5", 0]

    nid = 10
    if mode == "ref2va":
        ref_inputs: dict = {
            "clip": ["2", 0],
            "video_vae": ["3", 0],
            "audio_vae": ["4", 0],
            "prompt": prompt_text,
            "width": gen_w,
            "height": gen_h,
            "length": frames,
        }
        for idx, fn in enumerate(saved_imgs[:9]):
            wf[str(nid)] = {"class_type": "LoadImage", "inputs": {"image": fn}}
            ref_inputs[f"ref_image_{idx}"] = [str(nid), 0]
            nid += 1
        for idx, fn in enumerate(saved_vids[:3]):
            wf[str(nid)] = {
                "class_type": "VHS_LoadVideo",
                "inputs": {"video": fn, "force_rate": 24, "frame_load_cap": frames},
            }
            ref_inputs[f"ref_video_{idx}"] = [str(nid), 0]
            nid += 1
        for idx, fn in enumerate(saved_auds[:3]):
            wf[str(nid)] = {"class_type": "LoadAudio", "inputs": {"audio": fn}}
            ref_inputs[f"ref_audio_{idx}"] = [str(nid), 0]
            nid += 1
        wf["30"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": ref_inputs}
    else:
        fl_inputs: dict = {
            "clip": ["2", 0],
            "video_vae": ["3", 0],
            "audio_vae": ["4", 0],
            "prompt": prompt_text,
            "width": gen_w,
            "height": gen_h,
            "length": frames,
        }
        if mode == "fl2va" and len(saved_imgs) >= 1:
            wf["10"] = {"class_type": "LoadImage", "inputs": {"image": saved_imgs[0]}}
            fl_inputs["start_image"] = ["10", 0]
        if mode == "fl2va" and len(saved_imgs) >= 2:
            wf["11"] = {"class_type": "LoadImage", "inputs": {"image": saved_imgs[1]}}
            fl_inputs["end_image"] = ["11", 0]
        wf["30"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": fl_inputs}

    wf["31"] = {"class_type": "BasicGuider", "inputs": {"model": model_ref, "conditioning": ["30", 0]}}
    wf["32"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    wf["33"] = {"class_type": "BasicScheduler", "inputs": {"model": model_ref, "scheduler": "simple", "steps": steps, "denoise": 1.0}}
    wf["34"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": int(time.time()) % 2**31}}
    wf["35"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["34", 0],
            "guider": ["31", 0],
            "sampler": ["32", 0],
            "sigmas": ["33", 0],
            "latent_image": ["30", 1],
        },
    }
    wf["36"] = {"class_type": "VAEDecode", "inputs": {"samples": ["35", 0], "vae": ["3", 0]}}
    wf["37"] = {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "images": ["36", 0],
            "frame_rate": 24,
            "loop_count": 0,
            "filename_prefix": f"h3_{mode}",
            "format": "video/h264-mp4",
            "save_output": True,
        },
    }
    return wf


def poll_job(job_id: str, prompt_id: str, prefix: str, target_w: int, target_h: int, target_fps: int):
    t0 = time.time()
    try:
        while True:
            time.sleep(2.5)
            el = time.time() - t0
            JOBS[job_id]["progress"] = min(88, 30 + int(el / 2.4))
            JOBS[job_id]["message"] = f"MiniMax H3 推論中... ({int(el)}秒経過)"

            hist = json.loads(urllib.request.urlopen(f"{COMFY_URL}/history/{prompt_id}", timeout=5).read().decode())
            if prompt_id in hist:
                mp4 = None
                for out in hist[prompt_id].get("outputs", {}).values():
                    for k in ("gifs", "videos", "mp4"):
                        for it in out.get(k, []):
                            if str(it.get("filename", "")).endswith(".mp4"):
                                mp4 = it["filename"]
                                break
                if not mp4:
                    cands = sorted(COMFY_OUTPUT.glob(f"{prefix}*.mp4"), key=os.path.getmtime)
                    if cands:
                        mp4 = cands[-1].name
                if mp4:
                    raw_path = COMFY_OUTPUT / Path(mp4).name
                    JOBS[job_id]["progress"] = 94
                    JOBS[job_id]["message"] = f"{target_w}×{target_h} / {target_fps}fps マスタリング中..."
                    final_path = master_output_video(raw_path, target_w, target_h, target_fps)
                    JOBS[job_id].update({
                        "state": "done",
                        "progress": 100,
                        "message": f"生成完了 ({target_w}×{target_h} · {target_fps}fps)",
                        "video_url": f"/api/output/{final_path.name}",
                    })
                else:
                    JOBS[job_id].update({"state": "error", "error": "出力MP4が見つかりませんでした。"})
                return
    except Exception as e:
        JOBS[job_id].update({"state": "error", "error": str(e)})


@app.get("/", response_class=HTMLResponse)
def index():
    return STUDIO_HTML.read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    return {"status": "ok", "gpu": detect_gpu(), "comfy_ready": is_comfy_ready()}


@app.post("/api/generate")
async def generate(request: Request):
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    form = await request.form()
    cfg = json.loads(str(form.get("config", "{}")))
    gen_w, gen_h, target_w, target_h, target_fps = resolve_video_spec(cfg)

    imgs, vids, auds = [], [], []
    for key, val in form.multi_items():
        if key == "config" or not hasattr(val, "read"):
            continue
        raw = await val.read()
        orig = Path(val.filename or f"{key}.bin").name
        if key.startswith("ref_image_"):
            dst = COMFY_INPUT / f"{uuid.uuid4().hex[:8]}_{orig}"
            dst.write_bytes(raw)
            imgs.append(dst.name)
        elif key.startswith("ref_video_"):
            tmp = COMFY_INPUT / f"raw_{uuid.uuid4().hex[:8]}_{orig}"
            tmp.write_bytes(raw)
            out_name = f"ref_24fps_{uuid.uuid4().hex[:8]}.mp4"
            out_path = COMFY_INPUT / out_name
            subprocess.run([
                "ffmpeg", "-y", "-i", str(tmp), "-t", "5.0",
                "-vf", f"scale={gen_w}:{gen_h}:force_original_aspect_ratio=decrease,pad={gen_w}:{gen_h}:(ow-iw)/2:(oh-ih)/2,fps=24",
                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(out_path),
            ], check=True)
            tmp.unlink(missing_ok=True)
            vids.append(out_name)
        elif key.startswith("ref_audio_"):
            dst = COMFY_INPUT / f"{uuid.uuid4().hex[:8]}_{orig}"
            dst.write_bytes(raw)
            auds.append(dst.name)

    wf = build_comfy_workflow(cfg, imgs, vids, auds)
    req = urllib.request.Request(
        f"{COMFY_URL}/prompt",
        data=json.dumps({"prompt": wf}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        prompt_id = json.loads(resp.read().decode())["prompt_id"]

    job_id = uuid.uuid4().hex[:10]
    mode = cfg.get("mode", "ref2va").lower()
    JOBS[job_id] = {"state": "running", "progress": 25, "message": "ComfyUI キューに投入しました"}
    threading.Thread(
        target=poll_job,
        args=(job_id, prompt_id, f"h3_{mode}", target_w, target_h, target_fps),
        daemon=True,
    ).start()
    return {"job_id": job_id, "prompt_id": prompt_id}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    return JOBS.get(job_id, {"state": "error", "error": "Unknown job_id"})


@app.get("/api/output/{filename}")
def get_output(filename: str):
    target = COMFY_OUTPUT / Path(filename).name
    if not target.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(str(target), media_type="video/mp4", filename=target.name)


@app.post("/api/shutdown_runtime")
def shutdown_runtime():
    def _stop():
        time.sleep(1.0)
        try:
            from google.colab import runtime
            runtime.unassign()
        except Exception:
            os._exit(0)
    threading.Thread(target=_stop, daemon=True).start()
    return {"status": "shutting_down"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7860)
