"""
Created on Wed Apr 15 12:32:28 2026

@author: AXILLIOS
"""
# ============================= Fast path: copy outputs when same window =============================
def copy_outputs_between_classifiers(series_of_experiments,sampleRate,src_classifier,dst_classifier):
    """Copy Stage 0-4 outputs from src_classifier to dst_classifier (rename filenames accordingly).
    Uses multithreading (I/O bound) for faster copy on disks.
    """
    import os
    import shutil
    from concurrent.futures import ThreadPoolExecutor, as_completed

    stage_folders = [
        "1_CLEAN",
        "2_FEATS_PREPROCESSSED",
        "3_FEATS",
        "4_FEATS_COMBINED",
        "5_FEATS_SELECTION",
    ]

    def _copy_tree_rename(src_root: str, dst_root: str):
        if not os.path.isdir(src_root):
            return 0

        jobs = []
        n_copied = 0
        os.makedirs(dst_root, exist_ok=True)

        def _copy_one(src_file: str, dst_file: str):
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            shutil.copy2(src_file, dst_file)

        # Collect all files
        for dirpath, _, filenames in os.walk(src_root):
            for fn in filenames:
                src_file = os.path.join(dirpath, fn)
                rel = os.path.relpath(src_file, src_root)
                dst_file = os.path.join(dst_root, rel)

                # Rename only by classifier token in path/filename
                dst_file = dst_file.replace(src_classifier, dst_classifier)

                jobs.append((src_file, dst_file))

        # Copy in parallel (I/O bound)
        max_workers = min(32, (os.cpu_count() or 4) * 4)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_copy_one, s, d) for s, d in jobs]
            for f in as_completed(futs):
                f.result()
                n_copied += 1
        return n_copied

    total = 0
    for folder in stage_folders:
        src_root = os.path.join(".", folder, f"{sampleRate}_Hz_sampling", src_classifier)
        dst_root = os.path.join(".", folder, f"{sampleRate}_Hz_sampling", dst_classifier)
        total += _copy_tree_rename(src_root, dst_root)

    # Also copy Stage-2 FFT best index artifacts if they exist (they live in project root in this script)
    root_artifacts = [
        f"{series_of_experiments}best_fft_index{src_classifier}.json",
        f"{series_of_experiments}best_fft_index{src_classifier}.h",
    ]
    for src_name in root_artifacts:
        src_path = os.path.join(".", src_name)
        if os.path.isfile(src_path):
            dst_path = os.path.join(".", src_name.replace(src_classifier, dst_classifier))
            try:
                shutil.copy2(src_path, dst_path)
                total += 1
            except Exception as e:
                print(f"[COPY] Failed {src_path} -> {dst_path}: {e}")

    print(f"[COPY] Copied/renamed {total} files: {src_classifier} -> {dst_classifier}")
    return total
