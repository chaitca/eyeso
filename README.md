# eyeso

## Manual ellipse annotator

Use `manual_ellipse_annotator.py` to visually mark the iris and pupil with two ellipses.

```powershell
conda activate pytorch
cd C:\Users\14312\VsCode_Project\Limu\_push_eyeso_repo
python manual_ellipse_annotator.py
```

Click `Open Image` to load the original eye image. Mark five boundary points for `Iris`, switch to `Pupil`, then mark five more points. After each ellipse is fitted, drag the center point or square handles to adjust it.

Click `Save` to export:

- `*_ellipses.csv`: ellipse center, width, height, and angle
- `*_ellipses.json`: ellipse data and the five clicked points
- `*_ellipses.png`: image with the two ellipses drawn on top
