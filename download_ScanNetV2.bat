@echo off
set OUTDIR=C:\Users\Administrator\Desktop\Open3DSG-main\open3dsg\data\SCANNET\scannet_3d\data
set SCENES=scene0191_00 scene0568_00 scene0707_00
set TYPES=.aggregation.json .txt _vh_clean.aggregation.json _vh_clean_2.0.010000.segs.json _vh_clean_2.labels.ply _vh_clean_2.ply

for %%s in (%SCENES%) do (
  for %%t in (%TYPES%) do (
    python download_ScanNetV2.py --out_dir %OUTDIR% --id %%s --type %%t
  )
)