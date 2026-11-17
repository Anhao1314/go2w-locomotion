#!/usr/bin/env python3
"""实时观察面板：训练进度 / loss 曲线 / checkpoint / 最近评测 / GPU。

只读监控，不触碰训练进程。纯标准库实现（无额外依赖），
Windows 侧运行（训练日志在 D 盘，直接读取，不经 WSL）。

用法：python scripts/dashboard_server.py --port 8765
浏览器打开 http://localhost:8765
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "vla"

LINE_RE = re.compile(
    r"step:(?P<step>[\d.]+)(?P<unit>[KM]?)\s+smpl:.*?"
    r"loss:(?P<loss>[\d.]+)\s+grdn:(?P<grdn>[\d.]+)\s+lr:(?P<lr>[\d.eE+-]+)"
)
PROG_RE = re.compile(r"Training:\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[")
SMP_RE = re.compile(r"smp/s:(?P<smp>[\d.]+)")


def latest_train_log() -> Path | None:
    logs = list(LOG_DIR.glob("*train*.log"))
    if not logs:
        return None
    return max(logs, key=lambda p: p.stat().st_mtime)


def parse_log(path: Path) -> dict:
    raw = path.read_bytes().decode("utf-8", errors="ignore").replace("\r", "\n")
    points = []
    prog = None
    smp = None
    for line in raw.split("\n"):
        m = LINE_RE.search(line)
        if m:
            step = float(m.group("step")) * (1000 if m.group("unit") == "K" else
                                             1000000 if m.group("unit") == "M" else 1)
            points.append({
                "step": step,
                "loss": float(m.group("loss")),
                "lr": float(m.group("lr")),
                "grdn": float(m.group("grdn")),
            })
        p2 = PROG_RE.search(line)
        if p2:
            prog = {"pct": int(p2.group(1)), "step": int(p2.group(2)),
                    "total": int(p2.group(3))}
        m3 = SMP_RE.search(line)
        if m3:
            smp = float(m3.group("smp"))
    return {"points": points, "progress": prog, "smp_s": smp,
            "mtime": path.stat().st_mtime, "size": path.stat().st_size}


_CACHE: dict = {"key": None, "data": None}


def cached_parse(path: Path) -> dict:
    st = path.stat()
    key = (str(path), st.st_mtime, st.st_size)
    if _CACHE["key"] != key:
        _CACHE["key"] = key
        _CACHE["data"] = parse_log(path)
    return _CACHE["data"]


def gpu_info() -> dict | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        parts = out.stdout.strip().split(", ")
        if len(parts) == 3:
            return {"util": int(parts[0]), "mem_used": int(parts[1]),
                    "mem_total": int(parts[2])}
    except Exception:
        return None
    return None


def checkpoints() -> list[str]:
    best = None
    for d in LOG_DIR.glob("*/checkpoints"):
        if d.is_dir():
            m = max(p.stat().st_mtime for p in d.iterdir() if p.is_dir())
            if best is None or m > best[0]:
                best = (m, d)
    if best is None:
        return []
    names = sorted(p.name for p in best[1].iterdir() if p.is_dir())
    return names


def recent_evals(k: int = 6) -> list[dict]:
    rows = []
    for p in LOG_DIR.glob("**/suite_summary.json"):
        rows.append((p.stat().st_mtime, p, "suite"))
    for p in LOG_DIR.glob("**/libero_eval_summary.json"):
        rows.append((p.stat().st_mtime, p, "single"))
    rows.sort(reverse=True)
    out = []
    for mtime, p, kind in rows[:k]:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            out.append({
                "name": p.parent.name,
                "when": time.strftime("%m-%d %H:%M", time.localtime(mtime)),
                "kind": kind,
                "success_rate": d.get("success_rate"),
                "episodes": d.get("episodes"),
                "successes": d.get("successes"),
                "fps": d.get("fps"),
                "goal_image": d.get("goal_image"),
                "flow_steps": d.get("flow_steps"),
            })
        except Exception:
            continue
    return out


def tail_lines(path: Path, k: int = 12) -> list[str]:
    raw = path.read_bytes().decode("utf-8", errors="ignore").replace("\r", "\n")
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    return lines[-k:]


def status() -> dict:
    log = latest_train_log()
    if log is None:
        return {"state": "no_log"}
    parsed = cached_parse(log)
    age = time.time() - parsed["mtime"]
    state = "running" if age < 45 else "stalled_or_done"
    return {
        "state": state,
        "log": log.name,
        "age_s": round(age, 1),
        "points": parsed["points"][-300:],
        "progress": parsed["progress"],
        "smp_s": parsed["smp_s"],
        "checkpoints": checkpoints(),
        "evals": recent_evals(),
        "gpu": gpu_info(),
        "tail": tail_lines(log),
    }


PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>go2w 实时面板</title>
<style>
  body { background:#0f1115; color:#d8dee9; font: 14px/1.5 "Segoe UI", sans-serif; margin:0; }
  .wrap { max-width: 1080px; margin: 0 auto; padding: 20px; }
  h1 { font-size: 18px; margin: 0 0 4px; color:#88c0d0; }
  .sub { color:#6b7280; font-size: 12px; margin-bottom: 16px; }
  .grid { display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; margin-bottom:16px; }
  .card { background:#171a21; border:1px solid #262b36; border-radius:10px; padding:14px; }
  .k { color:#6b7280; font-size:11px; text-transform:uppercase; letter-spacing:1px; }
  .v { font-size:24px; font-weight:600; margin-top:4px; color:#eceff4; }
  .v small { font-size:12px; color:#6b7280; }
  .bar { height:14px; background:#262b36; border-radius:7px; overflow:hidden; margin-top:10px; }
  .bar>i { display:block; height:100%; background:linear-gradient(90deg,#5e81ac,#88c0d0); }
  canvas { width:100%; height:180px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  td,th { padding:6px 8px; border-bottom:1px solid #262b36; text-align:left; }
  th { color:#6b7280; font-weight:500; font-size:11px; text-transform:uppercase; }
  .tail { font:11px/1.6 Consolas, monospace; color:#9ca3af; white-space:pre-wrap; word-break:break-all;
          max-height:180px; overflow:auto; background:#10131a; padding:10px; border-radius:8px; }
  .pill { display:inline-block; padding:2px 10px; border-radius:10px; font-size:12px; }
  .run { background:#2e5339; color:#a3d9a5; } .stop { background:#533335; color:#e0a3a3; }
  .badge { color:#ebcb8b; font-size:12px; }
</style></head><body><div class="wrap">
<h1>go2w-locomotion · VLA 实时面板</h1>
<div class="sub">只读监控 · 每 3 秒自动刷新 · <span id="state" class="pill stop">-</span>
 · 日志 <span id="logname">-</span> <span id="clock" class="badge"></span></div>

<div class="bar"><i id="bar" style="width:0%"></i></div>
<div class="grid" style="margin-top:16px">
  <div class="card"><div class="k">训练步数</div><div class="v" id="step">-<small> / -</small></div></div>
  <div class="card"><div class="k">当前 loss</div><div class="v" id="loss">-</div></div>
  <div class="card"><div class="k">学习率</div><div class="v" id="lr">-</div></div>
  <div class="card"><div class="k">速度</div><div class="v" id="speed">-<small> step/s</small></div></div>
</div>
<div class="grid">
  <div class="card" style="grid-column:span 2"><div class="k">loss 曲线</div><canvas id="losscv" width="500" height="180"></canvas></div>
  <div class="card" style="grid-column:span 2"><div class="k">学习率曲线</div><canvas id="lrcv" width="500" height="180"></canvas></div>
</div>
<div class="grid">
  <div class="card"><div class="k">GPU</div><div class="v" id="gpu">-</div></div>
  <div class="card"><div class="k">已存 checkpoint</div><div class="v" id="ck">-</div></div>
  <div class="card" style="grid-column:span 2"><div class="k">日志尾部</div><div class="tail" id="tail">-</div></div>
</div>
<div class="grid">
  <div class="card" style="grid-column:span 2">
    <div class="k">仿真环境 · 最近一集录像（每 20 秒刷新）</div>
    <video id="simvideo" width="480" style="width:100%;border-radius:8px;margin-top:8px" controls muted autoplay loop></video>
    <div class="sub" id="viewerinfo">观察器状态加载中…</div>
  </div>
  <div class="card" style="grid-column:span 2">
    <div class="k">当前帧快照</div>
    <img id="snap" style="width:100%;border-radius:8px;margin-top:8px" alt="等待观察器…">
    <div class="sub" id="snapinfo">-</div>
  </div>
</div>
<div class="card"><div class="k">最近评测（logs/vla）</div>
<table><thead><tr><th>时间</th><th>名称</th><th>成功率</th><th>成功/集数</th><th>备注</th></tr></thead>
<tbody id="evals"></tbody></table></div>
</div>
<script>
function line(cv, pts, color, logY) {
  const c = document.getElementById(cv), g = c.getContext("2d");
  g.clearRect(0,0,c.width,c.height);
  if (pts.length < 2) return;
  const xs = pts.map(p=>p[0]), ys = pts.map(p=>p[1]);
  const x0=Math.min(...xs), x1=Math.max(...xs);
  let y0=Math.min(...ys), y1=Math.max(...ys);
  if (logY) { ys.forEach((v,i)=>ys[i]=Math.log10(Math.max(v,1e-7))); y0=Math.min(...ys); y1=Math.max(...ys); }
  if (y1-y0 < 1e-9) { y1 = y0 + 1e-9; }
  g.strokeStyle="#262b36"; g.beginPath(); g.moveTo(0,c.height-1); g.lineTo(c.width,c.height-1); g.stroke();
  g.strokeStyle=color; g.lineWidth=1.6; g.beginPath();
  pts.forEach((p,i)=>{ const x=(p[0]-x0)/(x1-x0||1)*(c.width-10)+5;
    const y=c.height-5-(ys[i]-y0)/(y1-y0)*(c.height-14);
    i?g.lineTo(x,y):g.moveTo(x,y); });
  g.stroke();
  g.fillStyle="#6b7280"; g.font="10px sans-serif";
  g.fillText(x1>=1000? (x1/1000)+"k" : Math.round(x1), c.width-40, 12);
}
async function poll() {
  try {
    const r = await fetch("/api/status"); const s = await r.json();
    document.getElementById("state").textContent = s.state;
    document.getElementById("state").className = "pill " + (s.state==="running"?"run":"stop");
    document.getElementById("logname").textContent = s.log || "-";
    document.getElementById("clock").textContent = new Date().toLocaleTimeString();
    const p = s.progress || {};
    document.getElementById("bar").style.width = (p.pct||0) + "%";
    document.getElementById("step").innerHTML = (p.step||"-") + "<small> / " + (p.total||"-") + "</small>";
    const pts = s.points || [];
    if (pts.length) {
      document.getElementById("loss").textContent = pts[pts.length-1].loss.toFixed(3);
      document.getElementById("lr").textContent = pts[pts.length-1].lr.toExponential(1);
    }
    document.getElementById("speed").innerHTML = (s.smp_s||"-") + "<small> 样本/s</small>";
    line("losscv", pts.map(p=>[p.step, p.loss]), "#88c0d0", false);
    line("lrcv", pts.map(p=>[p.step, p.lr]), "#ebcb8b", true);
    const g = s.gpu;
    document.getElementById("gpu").innerHTML = g ? g.util + "<small>% util · " + g.mem_used + "/" + g.mem_total + " MB</small>" : "-";
    document.getElementById("ck").innerHTML = (s.checkpoints||[]).length;
    document.getElementById("tail").textContent = (s.tail||[]).join("\\n");
    // 仿真录像与观察器状态
    fetch("/api/viewer").then(r=>r.json()).then(v=>{
      document.getElementById("viewerinfo").textContent =
        v.viewer === "offline" ? "观察器离线" :
        ("EP " + v.episode + " · seed " + v.seed + " · " + v.outcome + " · " + v.steps + " 步 · 更新 " + v.updated);
    }).catch(()=>{});
    fetch("/api/livefile").then(r=>r.json()).then(f=>{
      if (f.found) {
        const vid = document.getElementById("simvideo");
        if (!vid.dataset.src || vid.dataset.src !== f.name) {
          vid.dataset.src = f.name;
          vid.src = "/media/" + f.name + "?t=" + Date.now();
          vid.load(); vid.play().catch(()=>{});
        }
        document.getElementById("snap").src = "/media/latest.jpg?t=" + Date.now();
        document.getElementById("snapinfo").textContent = "更新 " + f.updated;
      }
    }).catch(()=>{});
    document.getElementById("evals").innerHTML = (s.evals||[]).map(e=>{
      const note = [e.fps?("fps="+e.fps):"", e.goal_image?"goal图":"", e.flow_steps?("flow="+e.flow_steps):""].filter(Boolean).join(" · ");
      return "<tr><td>"+e.when+"</td><td>"+e.name+"</td><td>"+(e.success_rate!=null? (e.success_rate*100).toFixed(0)+"%":"-")+
             "</td><td>"+(e.successes!=null? e.successes+"/"+e.episodes:"-")+"</td><td>"+note+"</td></tr>";
    }).join("");
  } catch(e) {}
}
poll(); setInterval(poll, 3000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/media/latest_episode.mp4"):
            f = LOG_DIR / "live_episodes" / "latest_episode.mp4"
            if not f.exists():
                self.send_response(404)
                self.end_headers()
                return
            data = f.read_bytes()
            rng = self.headers.get("Range")
            if rng and rng.startswith("bytes="):
                start = int(rng.split("=")[1].split("-")[0] or 0)
                chunk = data[start:start + 2_000_000]
                self.send_response(206)
                self.send_header("Content-Range",
                                 f"bytes {start}-{start + len(chunk) - 1}/{len(data)}")
            else:
                chunk = data
                self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(chunk)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(chunk)
        elif self.path.startswith("/media/latest.jpg"):
            f = LOG_DIR / "live_episodes" / "latest.jpg"
            if f.exists():
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(f.stat().st_size))
                self.end_headers()
                self.wfile.write(f.read_bytes())
            else:
                self.send_response(404)
                self.end_headers()
        elif self.path.startswith("/api/viewer"):
            # 转发 WSL 观察器状态（Windows → WSL localhost 转发）
            try:
                import urllib.request

                with urllib.request.urlopen("http://localhost:8766/status", timeout=3) as r:
                    body = r.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                body = json.dumps({"viewer": "offline", "err": str(exc)[:80]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        elif self.path.startswith("/api/livefile"):
            d = LOG_DIR / "live_episodes"
            status_f = d / "latest_status.json"
            if status_f.exists():
                st = json.loads(status_f.read_text(encoding="utf-8"))
                body = json.dumps({"found": True, "name": "latest_episode.mp4",
                                   "updated": st.get("updated", ""),
                                   "outcome": st.get("outcome")}).encode()
            else:
                body = json.dumps({"found": False}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/status"):
            body = json.dumps(status(), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(PAGE.encode("utf-8"))

    def log_message(self, format: str, *args: object) -> None:  # 静默访问日志
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"dashboard: http://localhost:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
