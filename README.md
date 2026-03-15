# Auto-Video-Agent

基于多 Agent 的全自动视频剪辑工作流：Vision → Director → Audio → Editor，一条命令完成从原始视频到成片 MP4 的生产。

## 功能概览

- Vision Agent：镜头切分（PySceneDetect 优先，OpenCV fallback）+ 每个 Shot 1~3 张关键帧视觉理解，输出 `visual_log.json`
- Director Agent：基于视觉日志生成旁白脚本 + timeline 规划，输出 `director_plan.json`（并强制时间轴校验）
- Audio Agent：默认 edge-tts 生成逐句 wav + 合并配音，读取真实时长回写，并按真实时长对齐 timeline；BGM 从本地库随机选择
- Editor Agent：MoviePy 合成（转场/字幕/三轨混音+ducking）并导出 `output.mp4`，显式 close 释放资源防 OOM

## 环境要求

- Python 3.10+（推荐 3.11/3.12）
- FFmpeg（MoviePy 导出与音频编码需要）
- 可选：ImageMagick（部分 MoviePy 版本的 TextClip 需要；若没有会跳过字幕渲染）

## 安装依赖

在项目根目录创建虚拟环境并安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install openai opencv-python-headless scenedetect edge-tts moviepy pytest
```

如果你当前使用的是系统 Anaconda 且没有写入权限（例如 `/opt/anaconda3`），可以改用：

```bash
python -m pip install --user opencv-python-headless
```

macOS 安装 FFmpeg（示例）：

```bash
brew install ffmpeg
```

如果字幕渲染失败提示与 ImageMagick 相关（示例）：

```bash
brew install imagemagick
```

## 必要环境变量

- `OPENAI_API_KEY`：Vision/Director 阶段调用 OpenAI 需要
- `AUTO_VIDEO_FONT`：字幕中文字体名称或字体文件路径（推荐设置，避免乱码）

示例（macOS）：

```bash
export OPENAI_API_KEY="YOUR_KEY"
export AUTO_VIDEO_FONT="PingFang SC"
```

## BGM 素材库（MVP）

将 `.wav` 或 `.mp3` 放入目录：

```text
media/bgm_library/
```

Audio Agent 会随机挑选一个作为 BGM；目录为空则静音（不会报错）。

## 一键跑通（端到端）

入口脚本：`scripts/run_pipeline.py`

```bash
python scripts/run_pipeline.py \
  --input "/absolute/path/to/input.mp4" \
  --workspace "/absolute/path/to/workspace" \
  --style_hint "专业技术讲解"
```

运行完成后会生成：

```text
<workspace>/
  runs/
    run_YYYYMMDD_HHMMSS_xxxxxxxx/
      frames/              # 抽帧
      visual_log.json      # 视觉日志
      director_plan.json   # 导演规划（Audio/Utils 会回写真实时长与对齐结果）
      audio/
        vo_0.wav ...
        voiceover.wav
        bgm.wav|bgm.mp3
      output.mp4           # 最终成片
```

## 常见问题

- 字幕中文乱码/不显示：
  - 设置 `AUTO_VIDEO_FONT`，例如 `PingFang SC`、`Heiti SC`，或传入字体文件路径（`.ttf/.otf`）。
  - 若仍失败，可能缺少 ImageMagick 或系统字体不可用；此时程序会跳过字幕渲染继续导出视频。
- 导出失败提示找不到 ffmpeg：
  - 安装 FFmpeg 并确保 `ffmpeg` 在 PATH 中。
- 内存占用过高：
  - 已在 Editor Agent 中对 VideoFileClip/AudioFileClip 显式 `close()`，但长视频仍建议分段或降低分辨率与 fps。

## 运行测试

```bash
python -m pytest -q
```
