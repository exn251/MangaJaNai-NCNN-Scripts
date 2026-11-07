import os
import subprocess
import shutil
import sys
import time
import hashlib
import io
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
from PIL import Image, ImageCms, ImageFilter, ImageOps

# Prevent Pillow DecompressionBomb warnings/errors for ultra-tall images
Image.MAX_IMAGE_PIXELS = None

# Try importing Wand for ImageMagick formatting/conversion
try:
    from wand.image import Image as WandImage
except ImportError:
    print("\n[ERROR] Wand library not found.")
    print("Please run: pip install Wand")
    sys.exit(1)

# ================= CONFIGURATION =================

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_ROOT = os.path.join(BASE_DIR, "Upscayl-Input")
OUTPUT_ROOT = os.path.join(BASE_DIR, "Upscayl-Output")
UPSCAYL_BIN = os.path.join(BASE_DIR, "bin", "upscayl-bin.exe")
MODELS_PATH = os.path.join(BASE_DIR, "models")

# --- ICC PROFILES (Optional: place in same directory as this script) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GAMMA1_ICC_PATH = os.path.join(SCRIPT_DIR, "Custom Gray Gamma 1.0.icc")
DOT20_ICC_PATH = os.path.join(SCRIPT_DIR, "Dot Gain 20%.icc")

# --- MODEL SELECTION ---
MODEL_NAME_COLOR = "RealESRGAN_x2plus" 

BW_MODEL_MAPPING = {
    1200: "2x_MangaJaNai_1200p_V1_ESRGAN_70k",
    1300: "2x_MangaJaNai_1300p_V1_ESRGAN_75k",
    1400: "2x_MangaJaNai_1400p_V1_ESRGAN_70k",
    1500: "2x_MangaJaNai_1500p_V1_ESRGAN_90k",
    1600: "2x_MangaJaNai_1600p_V1_ESRGAN_90k",
    1920: "2x_MangaJaNai_1920p_V1_ESRGAN_70k",
    2048: "2x_MangaJaNai_2048p_V1_ESRGAN_95k",
}

# --- LONG STRIP / MANHUA / WEBTOON SETTINGS ---
ENABLE_LONG_STRIP_SPLIT = True       # Detect and split long strip images for Upscayl
LONG_STRIP_ASPECT_RATIO = 2.2        # Height/Width ratio threshold to trigger splitting
MAX_SLICE_HEIGHT = 2048              # Target slice height for temporary Upscayl chunks
GUTTER_SEARCH_RANGE = 350            # Search range (px) to find blank panel gutters
STITCH_LONG_STRIPS = True            # Re-stitch slices into 1 single continuous image
LONG_STRIP_OUTPUT_FORMAT = "png"     # Format for stitched webtoons: "png" or "jpeg"
LONG_STRIP_OUTPUT_QUALITY = 95       # Quality setting for stitched output (1-100 scale)

# --- SUPPORTED FILE EXTENSIONS ---
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.avif', '.bmp')

# --- POST-PROCESSING SETTINGS (Official MangaJaNai Specs) ---
ENABLE_AUTO_LEVELS = True      
GRAYSCALE_THRESHOLD = 12        # Channel diff threshold for grayscale vs color (Official = 12)
TARGET_HEIGHT = 1700            # Target height for all manga pages (singles, splits, & spreads)
TARGET_WIDTH_WEBTOON = 750      # Target width strictly for webtoons / long strips
SPLIT_DOUBLE_PAGES = True       # Split double-page spreads into 2 single pages (and keep spread)

# --- TONE ADJUSTMENT (SEPARATED FOR B&W VS COLOR) ---
# Black & White pages: Keeps your preferred deep ink & screentone density
OUTPUT_GAMMA_BW = 1.35
BLACK_LEVEL_OFFSET_BW = 14

# Color pages: Untouched/natural colors (no crushed shadows or muddy skin tones)
OUTPUT_GAMMA_COLOR = 1.00
BLACK_LEVEL_OFFSET_COLOR = 0

# --- RESIZING KERNELS (ImageMagick / Wand) ---
ResizeKernelBW = "catrom"       # Official choice: Catmull-Rom ("catrom") for B&W
ResizeKernelColor = "lanczos"   # Official choice: Lanczos ("lanczos") for Color

OUTPUT_SCALE_BW = 2      
OUTPUT_SCALE_COLOR = 2   
TILE_SIZE_BW = 256      
TILE_SIZE_COLOR = 256   
GPU_ID = "auto"
THREAD_COUNT = "2:4:4"
ENABLE_TTA = False
MAX_WORKERS = 2    
SUFFIX = " - Upscaled"
OUTPUT_QUALITY = 65             # Quality setting for normal single manga pages
OUTPUT_FORMAT = "avif"          # Format: AVIF preserves screentones best
VERBOSE = False

# =================================================

# --- SETUP OFFICIAL ICC TRANSFORMS ---
DOTGAIN_TRANSFORMS_AVAILABLE = False
if os.path.exists(GAMMA1_ICC_PATH) and os.path.exists(DOT20_ICC_PATH):
    try:
        gamma1icc = ImageCms.getOpenProfile(GAMMA1_ICC_PATH)
        dotgain20icc = ImageCms.getOpenProfile(DOT20_ICC_PATH)
        dotgain20togamma1transform = ImageCms.buildTransformFromOpenProfiles(
            dotgain20icc, gamma1icc, "L", "L"
        )
        gamma1todotgain20transform = ImageCms.buildTransformFromOpenProfiles(
            gamma1icc, dotgain20icc, "L", "L"
        )
        DOTGAIN_TRANSFORMS_AVAILABLE = True
        print("[INFO] Initialized Official Dot Gain 20% ICC Color Profiles.")
    except Exception as e:
        print(f"[WARNING] ICC Profile initialization failed: {e}")

if not DOTGAIN_TRANSFORMS_AVAILABLE:
    print("[INFO] ICC profiles not found; using Gamma 1.78 mathematical fallback.")


def get_closest_model_height(img_height):
    """Official MangaJaNai 2.0 resolution bucket thresholds."""
    if img_height <= 1250:
        return 1200
    elif img_height <= 1350:
        return 1300
    elif img_height <= 1450:
        return 1400
    elif img_height <= 1550:
        return 1500
    elif img_height <= 1760:
        return 1600
    elif img_height <= 1984:
        return 1920
    else:
        return 2048


# ================= OFFICIAL PRE-PROCESSING =================

def cv_image_is_grayscale(image_np, user_threshold=GRAYSCALE_THRESHOLD):
    """
    Official MangaJaNai Grayscale Detection logic using OpenCV.
    Excludes pure black and pure white pixels to avoid skewing color variance.
    """
    if image_np.ndim == 2 or (image_np.ndim == 3 and image_np.shape[2] == 1):
        return True

    rgb = image_np[:, :, :3]
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    ignore_threshold = user_threshold

    r_g = cv2.subtract(cv2.absdiff(r, g), ignore_threshold)
    r_b = cv2.subtract(cv2.absdiff(r, b), ignore_threshold)
    g_b = cv2.subtract(cv2.absdiff(g, b), ignore_threshold)

    pure_black_mask = np.logical_and.reduce((r == 0, g == 0, b == 0))
    pure_white_mask = np.logical_and.reduce((r == 255, g == 255, b == 255))
    exclude_mask = np.logical_or(pure_black_mask, pure_white_mask)

    diff_sum = np.sum(np.where(exclude_mask, 0, r_g + r_b + g_b))
    size_without_black_and_white = np.sum(~exclude_mask) * 3

    if size_without_black_and_white == 0:
        return True

    ratio = diff_sum / size_without_black_and_white
    return ratio <= (user_threshold / 12.0)


def enhance_contrast(image_np):
    """Histogram Contrast Adjustment."""
    image_p = Image.fromarray(image_np).convert("L")
    hist = image_p.histogram()

    new_black_level = 0
    global_max_black = hist[0]
    for i in range(1, 31):
        if hist[i] > global_max_black:
            global_max_black = hist[i]
            new_black_level = i

    continuous_count = 0
    for i in range(31, 256):
        if hist[i] > global_max_black:
            continuous_count = 0
            global_max_black = hist[i]
            new_black_level = i
        elif hist[i] < global_max_black:
            continuous_count += 1
            if continuous_count > 1:
                break

    new_white_level = 255
    global_max_white = hist[255]
    for i in range(254, 224, -1):
        if hist[i] > global_max_white:
            global_max_white = hist[i]
            new_white_level = i

    continuous_count = 0
    for i in range(223, -1, -1):
        if hist[i] > global_max_white:
            continuous_count = 0
            global_max_white = hist[i]
            new_white_level = i
        elif hist[i] < global_max_white:
            continuous_count += 1
            if continuous_count > 1:
                break

    if new_white_level <= new_black_level:
        new_black_level, new_white_level = 0, 255

    image_array = np.array(image_p, dtype=np.float32)
    image_array = np.maximum(image_array - new_black_level, 0.0) / max(1.0, (new_white_level - new_black_level))
    return (np.clip(image_array, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def apply_gamma(image_np, gamma, black_offset):
    """Darkens midtones and deepens ink lines using an ultra-fast OpenCV LUT."""
    if gamma == 1.0 and black_offset == 0:
        return image_np
    
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        val = max(0.0, i - black_offset)
        norm = val / max(1.0, 255.0 - black_offset)
        lut[i] = int(round(np.clip((norm ** gamma) * 255.0, 0.0, 255.0)))
        
    return cv2.LUT(image_np, lut)


# ================= OFFICIAL POST-PROCESSING =================

def wand_resize_pil(pil_img, target_w, target_h, kernel_name):
    filter_str = "cubic" if kernel_name.lower() == "bicubic" else kernel_name.lower()
    
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    
    with WandImage(blob=buf.getvalue()) as w_img:
        w_img.resize(width=target_w, height=target_h, filter=filter_str)
        w_img.format = "png"
        with Image.open(io.BytesIO(w_img.make_blob())) as res_pil:
            return np.array(res_pil.convert(pil_img.mode))


def dotgain20_resize(image_np, new_size, kernel_name=ResizeKernelBW):
    h, w = image_np.shape[:2]
    target_w, target_h = new_size
    size_ratio = h / max(1, target_h)

    pil_image = Image.fromarray(image_np, mode="L")

    # Anti-aliasing Gaussian pre-blur to eliminate screentone moire when downscaling
    if size_ratio > 1.0:
        blur_size = (size_ratio - 1.0) / 3.5
        if blur_size >= 0.1:
            blur_size = min(blur_size, 250.0)
            pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_size))

    if DOTGAIN_TRANSFORMS_AVAILABLE:
        pil_image = ImageCms.applyTransform(pil_image, dotgain20togamma1transform, False)
        resized_np = wand_resize_pil(pil_image, target_w, target_h, kernel_name)
        pil_resized = Image.fromarray(resized_np, mode="L")
        pil_resized = ImageCms.applyTransform(pil_resized, gamma1todotgain20transform, False)
        return np.array(pil_resized)
    else:
        arr = np.array(pil_image, dtype=np.float32) / 255.0
        arr_linear = np.power(arr, 1.78)

        pil_linear = Image.fromarray((arr_linear * 255.0).round().astype(np.uint8), mode="L")
        resized_linear_np = wand_resize_pil(pil_linear, target_w, target_h, kernel_name)

        arr_resized = resized_linear_np.astype(np.float32) / 255.0
        arr_dotgain = np.power(arr_resized, 1.0 / 1.78)
        return (np.clip(arr_dotgain, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def standard_resize(image_np, new_size, kernel_name=ResizeKernelColor):
    target_w, target_h = new_size
    pil_img = Image.fromarray(image_np, mode="RGB")
    return wand_resize_pil(pil_img, target_w, target_h, kernel_name)


# ================= HELPER UTILITIES =================

def slice_long_image(img_np, target_height=MAX_SLICE_HEIGHT, search_range=GUTTER_SEARCH_RANGE):
    h, w = img_np.shape[:2]
    if h <= target_height:
        return [img_np]

    slices = []
    curr_y = 0

    while curr_y < h:
        if h - curr_y <= target_height:
            slices.append(img_np[curr_y:h])
            break

        search_start = max(curr_y + target_height - search_range, curr_y + 100)
        search_end = min(curr_y + target_height + search_range, h - 10)

        best_cut = -1
        if search_start < search_end:
            if img_np.ndim == 3:
                search_region = np.mean(img_np[search_start:search_end], axis=2)
            else:
                search_region = img_np[search_start:search_end]

            row_stds = np.std(search_region, axis=1)
            min_idx = np.argmin(row_stds)
            
            if row_stds[min_idx] < 5.0:
                best_cut = search_start + min_idx

        if best_cut != -1:
            slices.append(img_np[curr_y:best_cut])
            curr_y = best_cut
        else:
            end_y = curr_y + target_height
            slices.append(img_np[curr_y:end_y])
            curr_y = end_y

    return slices


def prepare_single_image(args):
    filename, input_folder, base_temp_dir, dir_col = args
    src_path = os.path.join(input_folder, filename)
    
    try:
        with open(src_path, "rb") as f:
            file_bytes = f.read()

        try:
            with Image.open(io.BytesIO(file_bytes)) as img:
                img = ImageOps.exif_transpose(img)
                img_np = np.array(img.convert("RGB"))
        except Exception:
            with WandImage(blob=file_bytes) as w_img:
                w_img.format = 'png'
                blob = w_img.make_blob()
                with Image.open(io.BytesIO(blob)) as img:
                    img = ImageOps.exif_transpose(img)
                    img_np = np.array(img.convert("RGB"))

        h, w = img_np.shape[:2]
        aspect_ratio = h / max(1, w)

        is_bw_full_image = cv_image_is_grayscale(img_np)

        if ENABLE_LONG_STRIP_SPLIT and aspect_ratio >= LONG_STRIP_ASPECT_RATIO:
            strip_slices = slice_long_image(img_np)
        else:
            strip_slices = [img_np]

        base_name = os.path.splitext(filename)[0]
        created_types = []

        for i, slice_np in enumerate(strip_slices):
            slice_filename = f"{base_name}_strip_{i+1:03d}.png" if len(strip_slices) > 1 else f"{base_name}.png"

            if is_bw_full_image:
                slice_h = slice_np.shape[0]
                closest_h = get_closest_model_height(slice_h)
                dest_folder = os.path.join(base_temp_dir, f"bw_{closest_h}")
                os.makedirs(dest_folder, exist_ok=True)
                dest_path = os.path.join(dest_folder, slice_filename)
                
                if ENABLE_AUTO_LEVELS:
                    processed_np = enhance_contrast(slice_np)
                else:
                    processed_np = cv2.cvtColor(slice_np, cv2.COLOR_RGB2GRAY)

                Image.fromarray(processed_np).convert("RGB").save(dest_path, format="PNG")
                created_types.append(f"BW_{closest_h}")
            else:
                dest_path = os.path.join(dir_col, slice_filename)
                Image.fromarray(slice_np).save(dest_path, format="PNG")
                created_types.append("COL")

        return created_types

    except Exception as e:
        return [f"ERR: {e}"]


def resize_single_image(args):
    filename, source_dir, dest_dir, model_name, resample_filter = args
    img_path = os.path.join(source_dir, filename)
    
    try:
        is_bw = model_name in BW_MODEL_MAPPING.values()
        name_root = os.path.splitext(filename)[0]
        base_output_name = f"{name_root}-{model_name}-{resample_filter}"

        with Image.open(img_path) as raw_img:
            img_np = np.array(raw_img.convert("L") if is_bw else raw_img.convert("RGB"))

        original_height, original_width = img_np.shape[:2]
        is_spread = original_width > original_height
        is_webtoon = ("_strip_" in filename) or ((original_height / max(1, original_width)) >= 1.8)

        def save_and_format(proc_np, suffix=""):
            # Apply tone curve based on whether this page is B&W or Color
            gamma = OUTPUT_GAMMA_BW if is_bw else OUTPUT_GAMMA_COLOR
            offset = BLACK_LEVEL_OFFSET_BW if is_bw else BLACK_LEVEL_OFFSET_COLOR
            
            if gamma != 1.0 or offset != 0:
                proc_np = apply_gamma(proc_np, gamma, offset)

            final_name = f"{base_output_name}{suffix}.{OUTPUT_FORMAT}"
            final_path = os.path.join(dest_dir, final_name)

            ram_buffer = io.BytesIO()
            mode_str = "L" if is_bw else "RGB"
            Image.fromarray(proc_np, mode=mode_str).save(ram_buffer, format="PNG")

            with WandImage(blob=ram_buffer.getvalue()) as wand_img:
                wand_img.format = OUTPUT_FORMAT
                wand_img.compression_quality = OUTPUT_QUALITY
                if OUTPUT_FORMAT.lower() == 'avif':
                    wand_img.options['heic:speed'] = '6'
                elif OUTPUT_FORMAT.lower() == 'webp':
                    wand_img.options['webp:method'] = '6'
                wand_img.save(filename=final_path)

            return final_name

        def resize_to_target(np_img):
            cur_h, cur_w = np_img.shape[:2]
            if is_webtoon:
                if cur_w > TARGET_WIDTH_WEBTOON:
                    new_w = TARGET_WIDTH_WEBTOON
                    new_h = int(round(cur_h * (TARGET_WIDTH_WEBTOON / max(1, cur_w))))
                else:
                    new_w = cur_w
                    new_h = cur_h
            else:
                if cur_h > TARGET_HEIGHT:
                    new_h = TARGET_HEIGHT
                    new_w = int(round(cur_w * (TARGET_HEIGHT / max(1, cur_h))))
                else:
                    new_h = cur_h
                    new_w = cur_w

            if is_bw:
                return dotgain20_resize(np_img, (new_w, new_h), resample_filter)
            else:
                return standard_resize(np_img, (new_w, new_h), resample_filter)

        # Handle Double-Page Spread Splitting (Saves split1, split2, AND the full spread)
        if is_spread and SPLIT_DOUBLE_PAGES and not is_webtoon:
            mid_point = original_width // 2

            def process_split(crop_np, suffix):
                resized_crop = resize_to_target(crop_np)
                save_and_format(resized_crop, suffix)

            # Japanese Manga layout: Split 1 = Right Page, Split 2 = Left Page
            process_split(img_np[:, mid_point:], "-split1")
            process_split(img_np[:, :mid_point], "-split2")

        # Process Main / Full Unsplit Page
        main_processed = resize_to_target(img_np)
        saved_name = save_and_format(main_processed)
        return f"Saved: {saved_name}"

    except Exception as e:
        return f"Error processing {filename}: {e}"


def stitch_long_strips_in_folder(dest_dir):
    if not STITCH_LONG_STRIPS:
        return

    target_format = LONG_STRIP_OUTPUT_FORMAT.lower()
    ext = "jpg" if target_format in ["jpg", "jpeg"] else "png"

    files = [f for f in os.listdir(dest_dir) if f.lower().endswith(f".{OUTPUT_FORMAT}")]
    strip_groups = {}

    for f in files:
        if "_strip_" in f:
            parts = f.split("_strip_")
            base_name = parts[0]
            
            sub_parts = parts[1].split("-", 1)
            slice_num_str = sub_parts[0]
            suffix_str = "-" + sub_parts[1] if len(sub_parts) > 1 else ""
            
            group_key = (base_name, suffix_str)
            if group_key not in strip_groups:
                strip_groups[group_key] = []
            
            strip_groups[group_key].append((slice_num_str, f))

    for (base_name, suffix_str), slice_tuples in strip_groups.items():
        if not slice_tuples:
            continue

        slice_tuples.sort(key=lambda x: x[0])
        sorted_files = [t[1] for t in slice_tuples]

        print(f"   -> Re-stitching {len(sorted_files)} slices into continuous image for {base_name}...")

        try:
            images = [Image.open(os.path.join(dest_dir, sf)) for sf in sorted_files]
            
            total_height = sum(img.height for img in images)
            max_width = max(img.width for img in images)

            stitched_img = Image.new(images[0].mode, (max_width, total_height))
            y_offset = 0
            for img in images:
                stitched_img.paste(img, (0, y_offset))
                y_offset += img.height
                img.close()

            base_without_ext = os.path.splitext(f"{base_name}{suffix_str}")[0]
            stitched_filename = f"{base_without_ext}.{ext}"
            final_stitched_path = os.path.join(dest_dir, stitched_filename)

            if ext in ["jpg", "jpeg"]:
                quality_val = max(1, min(100, LONG_STRIP_OUTPUT_QUALITY))
                stitched_img.convert("RGB").save(
                    final_stitched_path,
                    format="JPEG",
                    quality=quality_val,
                    optimize=True
                )
            else:
                png_compress_level = max(1, min(9, int((100 - LONG_STRIP_OUTPUT_QUALITY) / 10)))
                stitched_img.convert("RGB").save(
                    final_stitched_path,
                    format="PNG",
                    compress_level=png_compress_level,
                    optimize=True
                )

            stitched_img.close()

            for sf in sorted_files:
                try:
                    os.remove(os.path.join(dest_dir, sf))
                except Exception:
                    pass

            print(f"   [SUCCESS] Saved stitched image: {stitched_filename} ({max_width}x{total_height}px)")

        except Exception as e:
            print(f"   [ERROR] Failed to re-stitch slices for {base_name}: {e}")


def prepare_images_parallel(input_folder, base_temp_dir, dir_col):
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(SUPPORTED_EXTENSIONS)]
    print(f"Analyzing and Sorting {len(files)} images (Parallel)...")
    tasks = [(f, input_folder, base_temp_dir, dir_col) for f in files]

    results_summary = {"COL": 0}
    for k in BW_MODEL_MAPPING.keys():
        results_summary[f"BW_{k}"] = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(prepare_single_image, tasks))
        for res_list in results:
            for res in res_list:
                if res.startswith("BW_"):
                    results_summary[res] = results_summary.get(res, 0) + 1
                elif res == "COL":
                    results_summary["COL"] += 1
                elif res.startswith("ERR"): 
                    print(f"Skipping corrupt file/slice: {res}")
    return results_summary


def run_upscayl(input_dir, output_dir, model_name, tile_size, output_scale):
    is_2x_model = "2x" in model_name.lower() or "x2" in model_name.lower()
    current_scale = 2 if is_2x_model else output_scale

    cmd = [
        UPSCAYL_BIN,
        "-i", input_dir,
        "-o", output_dir,
        "-m", MODELS_PATH,
        "-n", model_name,
        "-f", "png",
    ]
    if current_scale: cmd.extend(["-s", str(current_scale)])
    if is_2x_model: cmd.extend(["-z", "2"])
    if tile_size > 0: cmd.extend(["-t", str(tile_size)])
    if GPU_ID and GPU_ID != "auto": cmd.extend(["-g", GPU_ID])
    if THREAD_COUNT: cmd.extend(["-j", THREAD_COUNT])
    if ENABLE_TTA: cmd.append("-x")
    if VERBOSE: cmd.append("-v")

    print(f"   -> Executing Upscayl using {model_name}...")
    subprocess.run(cmd, check=True) 


def process_final_images_parallel(source_dir, dest_dir, model_name, resample_filter):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    if not os.path.exists(source_dir):
        print(f"   [ERROR] Source dir missing: {source_dir}")
        return

    files = [f for f in os.listdir(source_dir) if f.lower().endswith(SUPPORTED_EXTENSIONS)]
    total = len(files)
    
    if total == 0:
        print(f"   [WARNING] No images found in {source_dir}.")
        return

    print(f"   -> Resizing, Post-processing, and Converting {total} images (Parallel)...")
    tasks = [(f, source_dir, dest_dir, model_name, resample_filter) for f in files]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i, result in enumerate(executor.map(resize_single_image, tasks)):
            if result.startswith("Error"):
                print(f"      [!] {result}")
            elif (i+1) % 5 == 0 or (i+1) == total:
                print(f"   [{i+1}/{total}] Processed")


def process_chapter(chapter_path, series_name, chapter_name, output_series_folder):
    output_chapter_folder = os.path.join(output_series_folder, f"{chapter_name}{SUFFIX}")
    if not os.path.exists(output_chapter_folder):
        os.makedirs(output_chapter_folder)
    
    print(f"\n{'='*60}")
    print(f"Processing: {series_name} / {chapter_name}")
    print(f"{'='*60}")
    
    job_id_str = f"{series_name}_{chapter_name}".encode('utf-8')
    job_hash = hashlib.md5(job_id_str).hexdigest()[:10]
    base_temp = os.path.join(OUTPUT_ROOT, "temp", f"job_{job_hash}")
    
    dir_col_in = os.path.join(base_temp, "col_in")
    dir_col_out = os.path.join(base_temp, "col_out")
    
    if os.path.exists(base_temp):
        shutil.rmtree(base_temp)
    
    os.makedirs(dir_col_in)
    os.makedirs(dir_col_out)

    bw_folders = {} 
    for height_key in BW_MODEL_MAPPING.keys():
        in_path = os.path.join(base_temp, f"bw_{height_key}")
        out_path = os.path.join(base_temp, f"bw_{height_key}_out")
        os.makedirs(in_path)
        os.makedirs(out_path)
        bw_folders[height_key] = {"in": in_path, "out": out_path}

    print(f"\n--- Step 1: Sorting and Pre-Processing ---")
    counts = prepare_images_parallel(chapter_path, base_temp, dir_col_in)
    
    total_bw = sum(counts[k] for k in counts if k.startswith("BW_"))
    total_col = counts["COL"]
    print(f"Sorted: {total_bw} B&W chunks, {total_col} Color chunks.")

    if total_bw == 0 and total_col == 0: return
    
    # Step 2: RUN B&W BATCHES
    for height_key, model_name in BW_MODEL_MAPPING.items():
        count = counts.get(f"BW_{height_key}", 0)
        
        if count > 0:
            print(f"\n--- Processing B&W Bucket: {height_key}p ({count} images) ---")
            in_dir = bw_folders[height_key]["in"]
            out_dir = bw_folders[height_key]["out"]
            
            try:
                run_upscayl(in_dir, out_dir, model_name, TILE_SIZE_BW, OUTPUT_SCALE_BW)
                process_final_images_parallel(out_dir, output_chapter_folder, model_name, ResizeKernelBW)
            except Exception as e:
                print(f"Error in B&W {height_key}p batch: {e}")

    if total_bw > 0 and total_col > 0:
        print("\n--- Cooling down GPU (3s) ---")
        time.sleep(3)

    # Step 3: RUN COLOR BATCH
    if total_col > 0:
        print(f"\n--- Step 3: Processing Color Batch ---")
        try:
            run_upscayl(dir_col_in, dir_col_out, MODEL_NAME_COLOR, TILE_SIZE_COLOR, OUTPUT_SCALE_COLOR)
            process_final_images_parallel(dir_col_out, output_chapter_folder, MODEL_NAME_COLOR, ResizeKernelColor)
        except Exception as e:
            print(f"Error in Color batch: {e}")

    # Step 4: Re-stitching Slices into 1 Continuous File
    if STITCH_LONG_STRIPS:
        print("\n--- Step 4: Re-stitching Long Strips ---")
        stitch_long_strips_in_folder(output_chapter_folder)

    # CLEANUP
    print(f"\n--- Cleanup Temp ---")
    try:
        shutil.rmtree(base_temp)
    except Exception as e:
        print(f"Warning: Could not remove temp folder: {e}")


def main():
    print(f"Checking for input in: {INPUT_ROOT}")
    if not os.path.exists(INPUT_ROOT):
        print(f"[ERROR] Input folder not found: {INPUT_ROOT}")
        return
    
    if not os.path.exists(OUTPUT_ROOT): os.makedirs(OUTPUT_ROOT)
    
    global_temp = os.path.join(OUTPUT_ROOT, "temp")
    if os.path.exists(global_temp): shutil.rmtree(global_temp)
    os.makedirs(global_temp)
    
    input_items = [d for d in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, d))]
    
    if not input_items:
        print("[WARNING] Input folder is empty or contains no folders.")
        return
    
    for item_name in input_items:
        item_path = os.path.join(INPUT_ROOT, item_name)
        sub_dirs = [d for d in os.listdir(item_path) if os.path.isdir(os.path.join(item_path, d))]
        images = [f for f in os.listdir(item_path) if f.lower().endswith(SUPPORTED_EXTENSIONS)]
        
        if sub_dirs:
            series_name = item_name
            output_series_folder = os.path.join(OUTPUT_ROOT, f"{series_name}{SUFFIX}")
            if not os.path.exists(output_series_folder): os.makedirs(output_series_folder)
            
            print(f"\n[INFO] Found Series: {series_name}")
            for chapter_name in sub_dirs:
                chapter_path = os.path.join(item_path, chapter_name)
                process_chapter(chapter_path, series_name, chapter_name, output_series_folder)
        
        elif images:
            print(f"\n[INFO] Found Single Chapter: {item_name}")
            process_chapter(item_path, "Single", item_name, OUTPUT_ROOT)
        else:
            print(f"[SKIP] '{item_name}' contains no images or subfolders.")
    
    try:
        shutil.rmtree(global_temp)
    except:
        pass
    print(f"\nDone! Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()