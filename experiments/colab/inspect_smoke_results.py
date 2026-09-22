import glob
import json
import os

results_dir = "/content/results/fairshare"
files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
print(f"Found {len(files)} result files in {results_dir}:")
for f in files:
    print(" -", f)

for f in files:
    fname = os.path.basename(f)
    print("\n" + "=" * 60)
    print(f"FILE: {fname}")
    print("=" * 60)
    try:
        with open(f) as fp:
            data = json.load(fp)
        manifest = data.get("manifest", {})
        print(f"Task: {manifest.get('task')}")
        print(f"Dataset: {manifest.get('dataset')} | Seeds: {manifest.get('seeds', manifest.get('num_seeds'))} | Device: {manifest.get('device')}")
        print("Summary:")
        print(json.dumps(data.get("summary"), indent=2))
    except Exception as e:
        print(f"Error reading {f}: {e}")
