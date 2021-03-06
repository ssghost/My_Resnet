[![CircleCI](https://circleci.com/gh/ssghost/My_Resnet.svg?style=svg)](https://circleci.com/gh/ssghost/My_Resnet)
# My_Resnet
An Automatic Compiled Resnet Deep Learning Algorithm Model.

Usage:
+    For training:

``$python train.py --images=[image_path] --labels=[label_path] --resize=[integer_square_side_length] --crop=[integer_square_side_length]``

Both 'images' and 'labels' can't be empty, at least one of 'resize' and 'crop' can't be empty.

+    For testing:

``$python test.py --inpath=[image_path] --outpath=[output_path] --resize=[integer_square_side_length] --crop=[integer_square_side_length] --modelpath=[load_compiled_models]``

Both 'inpath' and 'outpath' can't be empty, at least one of 'resize' and 'crop' can't be empty.

On Kaggle image classification contests, this Resnet model earned an average test accuracy of around 0.80.
