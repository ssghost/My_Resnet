# My_Resnet
An Automatic Compiled Resnet Deep Learning Algorithm Model.

Usage:
+    For training:

``$python train.py --images=[image_path] --labels=[label_path] --resize=[integer_square_side_length] --crop=[integer_square_side_length]``

Both 'images' and 'labels' can't be empty, at least one of 'resize' and 'crop' can't be empty.

+    For testing:

``$python test.py --images=[image_path] --outdir=[output_path] --resize=[integer_square_side_length] --crop=[integer_square_side_length]``

Both 'images' and 'outdir' can't be empty, at least one of 'resize' and 'crop' can't be empty.
