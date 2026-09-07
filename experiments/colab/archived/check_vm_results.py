"""Check VM results file.
"""
import os
import json
import glob

print("Checking /content/remediation_report.json:")
if os.path.exists("/content/remediation_report.json"):
    print("Found /content/remediation_report.json!")
    with open("/content/remediation_report.json") as f:
        print(f.read()[:500])
else:
    print("Not found /content/remediation_report.json")

print("\nChecking /content/FedFairGNN/results/:")
results = glob.glob("/content/FedFairGNN/results/*")
print("Results files:", results)
for r in results:
    print(f"--- {r} (size {os.path.getsize(r)}) ---")
    try:
        with open(r) as f:
            data = json.load(f)
            print("Keys:", list(data.keys()))
    except Exception as e:
        print("Error reading:", e)
