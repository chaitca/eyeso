# eyeso

## Manual ellipse annotator

Use `manual_ellipse_annotator.py` to visually mark the iris and pupil with two ellipses.

```powershell
conda activate pytorch
cd C:\Users\14312\VsCode_Project\Limu\_push_eyeso_repo
python manual_ellipse_annotator.py
```

Click `Open Images` to load one or more original eye images. Mark five boundary points for `Iris`, switch to `Pupil`, then mark five more points. After each ellipse is fitted, drag the center point or square handles to adjust it.

For batch annotation, select multiple images in the file picker, then use `Save+Next` to save the current image and move to the next one. You can also use `Prev` and `Next` to move through the selected images.

Click `Save` to export:

- `*_ellipses.csv`: ellipse center, width, height, and angle
- `*_ellipses.json`: ellipse data and the five clicked points
- `*_ellipses.png`: image with the two ellipses drawn on top
