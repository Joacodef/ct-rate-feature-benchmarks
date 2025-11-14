from pathlib import Path
import pandas as pd

feature_dir = Path(r"data/features/CT-CLIP_v2_corrected/image")  # adjust to your folder
output_csv = Path(r"data/manifests/ct_clip_v2_corrected_split.csv")

volume_names = sorted(p.stem for p in feature_dir.glob("*.npz"))  # change suffix if needed
df = pd.DataFrame({"VolumeName": [f"{name}.nii.gz" for name in volume_names]})

output_csv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_csv, index=False)
print(f"Wrote {len(df)} rows to {output_csv}")