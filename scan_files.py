import os

print("=== Scanning Att3 ===")
for root, dirs, files in os.walk('/root/E_Problem/DATA/Att3-Incomplete-Modality-Features'):
    pkls = [f for f in files if f.endswith('.pkl')]
    if pkls:
        print(f"Dir: {root}")
        print(f"pkl count: {len(pkls)}")
        print(f"samples: {sorted(pkls)[:5]}")

print("=== Scanning Att4 ===")
for root, dirs, files in os.walk('/root/E_Problem/DATA/Att4-Interpretable-Video-Samples-Features'):
    pkls = [f for f in files if f.endswith('.pkl')]
    if pkls:
        print(f"Dir: {root}")
        print(f"pkl count: {len(pkls)}")
        print(f"samples: {sorted(pkls)[:5]}")
