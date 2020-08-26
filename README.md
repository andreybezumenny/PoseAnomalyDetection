# PoseAnomalyDetection

## install requirements:

```shell script
    pip install -r requirements.txt
```

## Install AlphaPose

### CPU usage installation:

https://github.com/nickgrealy/AlphaPose/tree/pytorch-cpu

### Change some AlphaPose code:
https://github.com/MVIG-SJTU/AlphaPose/issues/335#issuecomment-508972808

## preprocess 
run preprocessing/convert_to_frames.py 
Has two options (ffmpeg / cv2)

## pose tracking 
run AlphaPose/demo.py
```shell script
python3 demo.py --indir ${img_directory} --outdir examples/res 
```
## Anomaly detection

run anomaly_detection/find_outliers.py
```python

```