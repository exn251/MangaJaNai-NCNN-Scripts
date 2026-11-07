# MangaJaNai-NCNN-Scripts
Personal scripts for converting and running MangaJaNai on AMD via NCNN
#

An automated batch-processing pipeline for upscaling manga and webtoons using the [upscayl-ncnn](https://github.com/upscayl/upscayl-ncnn) CLI binary and **MangaJaNai** ESRGAN models. 

It automatically detects color vs. black-and-white pages, handles long webtoon strips, splits double-page spreads, and resizes using print-standard color transforms to eliminate screentone moiré.

---

## Features

- **Webtoon & Long-Strip Support:** Detects tall vertical strips, intelligently slices them at blank panel gutters (so VRAM does not run out), upscales each chunk, and seamlessly re-stitches them back into one continuous image.
- **Smart Model Routing:** Separates color pages (RealESRGAN) from black-and-white pages (MangaJaNai), auto-sorting B&W pages into the best resolution bucket (1200p–2048p).
- **Spread Splitting:** Splits two-page landscape spreads into right-to-left individual pages while keeping the original full spread intact.
- **Tone & Ink Control:** Dedicated gamma and black-level controls for B&W pages to preserve deep inks and screentone density without muddying or crushing color pages.
- **MangaJaNai Post-Processing:** Dot Gain 20% ICC color profiles and anti-aliasing pre-blurring to avoid moiré patterns.
- **Built-in Model Converter:** Easily convert PyTorch `.pth` model checkpoints to Upscayl-compatible NCNN (`.param` / `.bin`) formats.

---

## Expected Folder Structure

The script determines paths relative to where it sits. If `MangaUpscale.py` is placed inside a subfolder (e.g., `scripts/`), your directory layout should look like this:

```text
MyProject/
│
├── bin/
│   └── upscayl-bin.exe                 <-- upscayl-ncnn CLI binary
│
├── models/                             <-- NCNN models (.bin and .param files)
│   ├── batch_model_convert_PTH2NCNN.py <-- Model conversion helper
│   ├── RealESRGAN_x2plus.bin
│   ├── RealESRGAN_x2plus.param
│   ├── 2x_MangaJaNai_1200p_...bin
│   ├── 2x_MangaJaNai_1200p_...param
│   └── ...
│
├── Upscayl-Input/                      <-- Drop your source images here
│   │
│   ├── My Series Title/                <-- Option A: Series folder containing chapters
│   │   ├── Chapter 01/
│   │   │   ├── 001.png
│   │   │   └── 002.png
│   │   └── Chapter 02/
│   │
│   └── Solo Chapter/                   <-- Option B: Chapter directly in root
│       ├── 001.png
│       └── 002.png
│
├── Upscayl-Output/                     <-- Processed output will appear here
│
└── scripts/
    ├── MangaUpscale.py                 <-- Main pipeline script
    ├── Dot Gain 20%.icc                <-- (Optional) ICC profile
    └── Custom Gray Gamma 1.0.icc       <-- (Optional) ICC profile
```

*(Note: If you run `MangaUpscale.py` directly from the project root rather than a subfolder, adjust `BASE_DIR = os.path.dirname(...)` at the top of the file).*

---

## Prerequisites & Installation

### 1. Upscayl Binary
Download the precompiled command-line binary from the [upscayl-ncnn Releases](https://github.com/upscayl/upscayl-ncnn/releases) and place `upscayl-bin.exe` in the `bin/` directory.

### 2. Download the Models (`.pth`)
Place the downloaded `.pth` files inside the `models/` folder:
- **Black & White Manga Models:** [MangaJaNai v1.0.0 Releases](https://github.com/the-database/MangaJaNai/releases/tag/1.0.0) *(download the `2x_MangaJaNai_*p_V1_ESRGAN_*k.pth` models for 1200p to 2048p)*.
- **Color Model:** [RealESRGAN_x2plus.pth (v0.2.1)](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.1).

### 3. Install ImageMagick (System Requirement)
`Wand` requires ImageMagick installed on your system:
- **Windows:** Download the installer from [imagemagick.org](https://imagemagick.org/script/download.php). During setup, check **"Install development headers and libraries for C and C++"**.
- **Linux:** `sudo apt-get install libmagickwand-dev`
- **macOS:** `brew install imagemagick`

### 4. Install Python Dependencies
```bash
pip install opencv-python numpy pillow wand
```

---

## Converting Models (`.pth` to NCNN)

[upscayl-ncnn](https://github.com/upscayl/upscayl-ncnn) requires models in **NCNN format** (`.param` and `.bin` pairs). Convert your downloaded `.pth` models using `batch_model_convert_PTH2NCNN.py`.

The script uses **Spandrel** to auto-detect model architectures, exports them via **PNNX (FP16)**, and renames the input/output layers to match Upscayl's expected `data` and `output` blobs.

### How to Convert:
1. Place `batch_model_convert_PTH2NCNN.py` inside your `models/` directory alongside your `.pth` files.
2. Run it inside the `models/` folder:

**Using `uv` (Fastest — installs dependencies automatically):**
```bash
cd models
uv run batch_model_convert_PTH2NCNN.py
```

**Using standard `pip`:**
```bash
pip install torch pnnx spandrel requests numpy
cd models
python batch_model_convert_PTH2NCNN.py
```

Once converted, the `.param` and `.bin` files will be saved in `models/` ready for use.

---

## Why is `Wand` Required?

Standard Python imaging libraries (like OpenCV or Pillow) lack the specialized resampling filters and color space conversions required for manga:

1. **Catmull-Rom (`catrom`) Resampling:** Standard Lanczos/Bicubic filters create ringing artifacts and severe **moiré patterns** on manga screentones (halftones). `Wand` connects to ImageMagick to provide Catmull-Rom filtering and precise linear light resizing.
2. **Modern Format Encoding:** `Wand` handles fast, fine-tuned compression for modern web formats like **AVIF** and **WebP**.

---

## How to Run the Pipeline

1. Ensure your converted NCNN models (`.param` and `.bin`) are in `models/` and `upscayl-bin.exe` is in `bin/`.
2. Place manga or webtoon folders into `Upscayl-Input/`.
3. Run the script:
   ```bash
   python scripts/MangaUpscale.py
   ```
4. Upscaled results will be generated in `Upscayl-Output/`.

---

## Key Settings (Inside `MangaUpscale.py`)

You can customize these variables directly near the top of the file:

### Tone Adjustment (Separated for B&W vs. Color)
Keeps deep ink lines and screentone density on B&W pages without crushing shadows or distorting skin tones on color pages:
```python
# --- TONE ADJUSTMENT (SEPARATED FOR B&W VS COLOR) ---
# Black & White pages: Keeps your preferred deep ink & screentone density
OUTPUT_GAMMA_BW = 1.35
BLACK_LEVEL_OFFSET_BW = 14

# Color pages: Untouched / natural colors
OUTPUT_GAMMA_COLOR = 1.00
BLACK_LEVEL_OFFSET_COLOR = 0
```

> **Disclaimer:** By default, B&W pages output slightly **darker** with richer black tones and denser screentones than the official MangaJaNai GUI due to personal preference. If you want the stock/official MangaJaNai look, simply reset the B&W settings to match the color defaults:
> ```python
> OUTPUT_GAMMA_BW = 1.00
> BLACK_LEVEL_OFFSET_BW = 0
> ```

### Webtoon & Sizing Controls
- `ENABLE_LONG_STRIP_SPLIT` *(default: `True`)*: Auto-detects and slices tall webtoon pages at panel gutters.
- `STITCH_LONG_STRIPS` *(default: `True`)*: Re-joins processed slices into one continuous image.
- `LONG_STRIP_OUTPUT_FORMAT` *(default: `"png"`)*: Format for stitched webtoons (`"png"` or `"jpeg"`).
- `TARGET_WIDTH_WEBTOON` *(default: `750`)*: Target downscaled width strictly for webtoons.
- `TARGET_HEIGHT` *(default: `1700`)*: Target downscaled height for standard manga pages.
- `SPLIT_DOUBLE_PAGES` *(default: `True`)*: Splits two-page spreads into individual portrait pages (Japanese reading order: Right, then Left) while preserving the original spread.
