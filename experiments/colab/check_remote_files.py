import os
import glob

print("Listing /content/results/fairshare:")
for f in glob.glob("/content/results/fairshare/*"):
    print(" -", f, os.path.getsize(f), "bytes")
