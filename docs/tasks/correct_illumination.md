# Illumination Correction

This task performs a global illumination normalization across all tiles of a multitile dataset. It ensures that tile intensities match consistently in overlapping regions, reducing brightness differences that arise from acquisition variability.

The correction operates by analyzing all tile–tile overlaps, estimating multiplicative coefficients that bring tiles to a common illumination level, and applying these coefficients to the entire dataset. 

An optional Z-axis correction is available to compensate for Z band artifacts.

---

## Parameters

### Z Correction

By setting this parameter to `True`, the task will perform a z-correction to compensate for Z band artifacts.

---    

## Local Run Command

```bash
python run_illumination_correction.py 
```
