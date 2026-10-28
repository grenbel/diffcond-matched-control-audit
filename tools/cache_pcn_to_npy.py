"""Write a float32 .npy copy of every PCN point cloud next to its .pcd file.

The reported PCN runs read this copy (cfgs/dataset_configs/PCN_npy.yaml). It was made as here:
each .pcd file is read with open3d.io.read_point_cloud and its points are saved with numpy as a
float32 array under the same name with the .npy extension. The files are those the PCN loader
reads for the train split (the complete cloud and 8 partial views per model) and the val and
test splits (the complete cloud and 1 view), as listed in the category file; files that do not
exist are left out, and existing .npy files are kept. Needs open3d and numpy.

Usage (from the repository root, with the PCN data in data/PCN):
  python tools/cache_pcn_to_npy.py --data-root data/PCN --category-file data/PCN/PCN.json
  python tools/cache_pcn_to_npy.py --dry-run      # count the files only
"""
import argparse
import collections
import json
import os
import sys
import time
from multiprocessing import Pool


def read_points(pcd_path):
    import numpy as np
    import open3d as o3d
    return np.asarray(o3d.io.read_point_cloud(pcd_path).points, dtype=np.float32)


def convert(pcd_path, reader=read_points):
    """Return (npy_path, status); status is 'written', 'skipped' or 'failed: <reason>'."""
    import numpy as np
    npy_path = os.path.splitext(pcd_path)[0] + ".npy"
    if os.path.exists(npy_path):
        return npy_path, "skipped"
    try:
        np.save(npy_path, reader(pcd_path))
        return npy_path, "written"
    except Exception as e:  # reported per file and counted in the summary
        return npy_path, "failed: %s" % e


def pcd_files(data_root, category_file):
    """The .pcd files of the train, test and val splits named in the category file."""
    with open(category_file) as f:
        categories = json.load(f)
    files = []
    for cat in categories:
        tid = cat["taxonomy_id"]
        for split in ("train", "test", "val"):
            views = 8 if split == "train" else 1
            for model_id in cat.get(split, []):
                files.append(os.path.join(data_root, split, "complete", tid, model_id + ".pcd"))
                files += [os.path.join(data_root, split, "partial", tid, model_id, "%02d.pcd" % v) for v in range(views)]
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default="data/PCN")
    ap.add_argument("--category-file", default="data/PCN/PCN.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="count the files, write nothing")
    args = ap.parse_args(argv)
    listed = pcd_files(args.data_root, args.category_file)
    files = [p for p in listed if os.path.exists(p)]
    print("%d .pcd files listed, %d found" % (len(listed), len(files)), flush=True)
    if args.dry_run:
        return 0
    start, results = time.time(), []
    with Pool(args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(convert, files, chunksize=100), 1):
            results.append(result)
            if i % 5000 == 0:
                print("  %d/%d (%.0f s)" % (i, len(files), time.time() - start), flush=True)
    status = collections.Counter("failed" if s.startswith("failed") else s for _, s in results)
    for path, s in results:
        if s.startswith("failed"):
            print(path, s)
    print("written %d, skipped %d, failed %d" % (status["written"], status["skipped"], status["failed"]))
    return 1 if status["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
