# CNN Step-by-Step Report

This report summarizes the CNN work completed so far on the CIFAR-10 project, from data understanding through the improved CNN comparison.

## 1. Project Goal

The goal of this part of the project was to train a Convolutional Neural Network (CNN) to classify CIFAR-10 images into 10 classes:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

Each image is a small `32 x 32` RGB image, so the CNN needed to learn useful visual features from limited spatial resolution.

## 2. We Explored the Dataset First

Before training the CNN, we inspected the CIFAR-10 data to understand what the model would see.

Key outputs:

- Sample image grid: [sample_images.png](../Outputs/eda/sample_images.png)
- Class distribution chart: [class_distribution.png](../Outputs/eda/class_distribution.png)
- Dataset summary: [cifar10_summary.json](../Outputs/eda/cifar10_summary.json)

These visuals helped confirm that:

- the dataset contains a wide variety of object categories
- the images are small and visually challenging
- the classes are evenly distributed, which is helpful for training and evaluation

### Reference Image: CIFAR-10 Sample Images

![CIFAR-10 sample images](../Outputs/eda/sample_images.png)

### Reference Graph: CIFAR-10 Class Distribution

![CIFAR-10 class distribution](../Outputs/eda/class_distribution.png)

## 3. We Built the First CNN Baseline

The first CNN was implemented in [SRC/cnn_model.py](../SRC/cnn_model.py) as `SimpleCIFAR10CNN`.

Main design choices:

1. Three convolution stages.
2. Feature depth grows from `32 -> 64 -> 128`.
3. `MaxPool2d` is used to reduce spatial size.
4. `AdaptiveAvgPool2d((1, 1))` compresses the final feature map.
5. A dropout layer and final linear layer produce class logits.

This first model was meant to be a clean, understandable baseline rather than a highly optimized CNN.

## 4. We Trained the First CNN Baseline

The baseline CNN training script is [SRC/train_cnn_baseline.py](../SRC/train_cnn_baseline.py).

Saved run:

- [summary.json](../Data/cnn_baseline_outputs/run_20260325_130955/summary.json)
- [training_curves.png](../Data/cnn_baseline_outputs/run_20260325_130955/training_curves.png)

Baseline training settings:

1. `epochs = 5`
2. `batch_size = 256`
3. `learning_rate = 0.001`
4. `weight_decay = 0.0001`
5. `dropout = 0.4`
6. `augment = False`
7. optimizer: `Adam`

Baseline result:

1. Best validation accuracy: `0.5848`
2. Test loss: `1.1409`
3. Test accuracy: `0.5769`

### Reference Graph: Baseline CNN Training Curves

![Baseline CNN training curves](../Data/cnn_baseline_outputs/run_20260325_130955/training_curves.png)

## 5. We Visualized CNN Feature Maps

After training the first CNN, we inspected feature maps to better understand what the network was learning internally.

Saved visualizations:

- [input_image.png](../Data/cnn_feature_maps_trained/run_20260325_130955/input_image.png)
- [conv1_1_feature_maps.png](../Data/cnn_feature_maps_trained/run_20260325_130955/conv1_1_feature_maps.png)
- [pool1_feature_maps.png](../Data/cnn_feature_maps_trained/run_20260325_130955/pool1_feature_maps.png)
- [conv2_1_feature_maps.png](../Data/cnn_feature_maps_trained/run_20260325_130955/conv2_1_feature_maps.png)
- [pool2_feature_maps.png](../Data/cnn_feature_maps_trained/run_20260325_130955/pool2_feature_maps.png)
- [conv3_2_feature_maps.png](../Data/cnn_feature_maps_trained/run_20260325_130955/conv3_2_feature_maps.png)

These images helped show:

1. early layers respond to edges, color transitions, and local texture patterns
2. deeper layers become more abstract and selective
3. pooling reduces spatial detail while preserving stronger signals

### Reference Image: Input Example

![Feature-map input image](../Data/cnn_feature_maps_trained/run_20260325_130955/input_image.png)

### Reference Image: Early Feature Maps

![conv1_1 feature maps](../Data/cnn_feature_maps_trained/run_20260325_130955/conv1_1_feature_maps.png)

### Reference Image: Deeper Feature Maps

![conv3_2 feature maps](../Data/cnn_feature_maps_trained/run_20260325_130955/conv3_2_feature_maps.png)

## 6. We Designed an Improved CNN

The improved model was implemented in [SRC/improved_cnn.py](../SRC/improved_cnn.py) as `DeeperCIFAR10CNN`, and trained with [SRC/train_improved_cnn.py](../SRC/train_improved_cnn.py).

Main improvements over the first CNN:

1. A deeper architecture with more configurable stages.
2. Wider channels: `48, 96, 192, 256`.
3. Configurable blocks per stage: `2, 2, 1, 1`.
4. `Conv + BatchNorm + ReLU` blocks instead of plain convolution blocks.
5. A larger classifier head with a hidden layer of size `256`.
6. Support for configurable pooling types.
7. Use of `AdamW` instead of `Adam`.

Important training changes:

1. `learning_rate` lowered from `0.001` to `0.0005`
2. `weight_decay` increased from `0.0001` to `0.0005`
3. `dropout` reduced from `0.4` to `0.3`
4. `batch_size` stayed at `256`
5. `augment` stayed `False`

## 7. We Trained the Improved CNN

Completed improved run:

- [summary.json](../Data/improved_cnn_outputs/run_20260326_151826/summary.json)
- [training_curves.png](../Data/improved_cnn_outputs/run_20260326_151826/training_curves.png)

Improved CNN result:

1. Best validation accuracy: `0.7590`
2. Test loss: `0.7474`
3. Test accuracy: `0.7447`

This run trained on CPU and still achieved a much stronger result than the first CNN baseline.

### Reference Graph: Improved CNN Training Curves

![Improved CNN training curves](../Data/improved_cnn_outputs/run_20260326_151826/training_curves.png)

## 8. We Compared the First CNN Against the Improved CNN

Comparison notebook:

- [improved_cnn_comparison.ipynb](./improved_cnn_comparison.ipynb)

Main outcome:

1. Baseline CNN test accuracy: `0.5769`
2. Improved CNN test accuracy: `0.7447`
3. Absolute improvement: `0.1678`
4. Relative improvement over baseline: about `29.09%`

This shows that the improved CNN clearly outperformed the first CNN.

## 9. We Upgraded the CNN Training Again

After that, we added another round of upgrades to the training process to try to push accuracy even higher.

In simple terms, these upgrades meant:

1. The model saw more varied versions of the same training images.
2. The learning rate changed more smoothly during training instead of staying fixed.
3. Some training images were mixed together so the model would learn more general patterns.
4. Parts of images were randomly erased so the model would not rely too heavily on one small detail.
5. The labels were slightly softened so the model would be less overconfident.
6. The model was trained for longer, giving it more time to learn.

The main upgrades added were:

1. `TrivialAugmentWide`
2. crop padding and horizontal flipping
3. `RandomErasing`
4. `MixUp`
5. `CutMix`
6. label smoothing
7. cosine learning-rate scheduling with warmup

Latest upgraded run:

- [summary.json](../Data/improved_cnn_outputs/run_20260326_161003/summary.json)
- [training_curves.png](../Data/improved_cnn_outputs/run_20260326_161003/training_curves.png)

Latest upgraded result:

1. Best validation accuracy: `0.8040`
2. Test loss: `0.9944`
3. Test accuracy: `0.7979`

This was better than the earlier improved CNN:

1. Previous improved CNN test accuracy: `0.7447`
2. New upgraded CNN test accuracy: `0.7979`
3. Additional gain: `0.0532`

### Reference Graph: Latest Upgraded CNN Training Curves

![Latest upgraded CNN training curves](../Data/improved_cnn_outputs/run_20260326_161003/training_curves.png)

## 10. We Trained CNN Take 4

After that, we built and trained **CNN Take 4**, which was our strongest version so far.

In simple terms, the main changes were:

1. We kept the stronger training recipe from the previous run.
2. We changed the downsampling to `stride` so the network could learn how to shrink feature maps instead of using fixed pooling.
3. We added explicit color augmentation, including brightness, contrast, saturation, and small hue changes.
4. We also added a small amount of random grayscale augmentation so the model would not depend too much on color alone.
5. We resumed the interrupted Take 4 run and finished all 12 epochs.

CNN Take 4 run:

- [summary.json](../Data/improved_cnn_outputs/cnn_take_4_20260326_234323/summary.json)
- [training_curves.png](../Data/improved_cnn_outputs/cnn_take_4_20260326_234323/training_curves.png)
- [best_improved_cnn.pt](../Data/improved_cnn_outputs/cnn_take_4_20260326_234323/best_improved_cnn.pt)

CNN Take 4 result:

1. Best validation accuracy: `0.8252`
2. Test loss: `0.9340`
3. Test accuracy: `0.8218`

This was better than the previous best CNN:

1. Previous best test accuracy: `0.7979`
2. CNN Take 4 test accuracy: `0.8218`
3. Additional gain: `0.0239`

### Reference Graph: CNN Take 4 Training Curves

![CNN Take 4 training curves](../Data/improved_cnn_outputs/cnn_take_4_20260326_234323/training_curves.png)

## 11. Final Conclusion

Step by step, the CNN work so far has followed this path:

1. Understand the CIFAR-10 dataset with images and class-distribution graphs.
2. Build a simple first CNN baseline.
3. Train and evaluate the baseline.
4. Visualize feature maps to interpret what the CNN learned.
5. Design a deeper and better-regularized CNN.
6. Train the improved CNN and compare it fairly with the first model.
7. Upgrade the training recipe with stronger augmentation and a better learning schedule.
8. Build CNN Take 4 with learned stride downsampling and stronger color augmentation.
9. Test transfer learning using a pretrained ResNet18 and compare against the scratch CNN.

Final conclusion:

CNN Take 4 produced the best result so far and raised test accuracy to `0.8218`.

Transfer learning was added as a final comparison. A pretrained ResNet18 was fine-tuned on CIFAR-10 (feature extraction after resizing to `128x128` for speed). The first epoch already reached `0.7822` test accuracy, which is strong but still below CNN Take 4.

## 12. Transfer Learning Summary

Transfer learning run:

- [summary.json](../Data/transfer_learning_outputs/run_20260329_030438/summary.json)
- [best_transfer_model.pt](../Data/transfer_learning_outputs/run_20260329_030438/best_transfer_model.pt)

Result so far:

1. Best validation accuracy: `0.7792`
2. Test loss: `0.6587`
3. Test accuracy: `0.7822`

## 13. Key File References

- [SRC/cnn_model.py](../SRC/cnn_model.py)
- [SRC/improved_cnn.py](../SRC/improved_cnn.py)
- [SRC/train_cnn_baseline.py](../SRC/train_cnn_baseline.py)
- [SRC/train_improved_cnn.py](../SRC/train_improved_cnn.py)
- [baseline summary.json](../Data/cnn_baseline_outputs/run_20260325_130955/summary.json)
- [baseline training_curves.png](../Data/cnn_baseline_outputs/run_20260325_130955/training_curves.png)
- [improved summary.json](../Data/improved_cnn_outputs/run_20260326_151826/summary.json)
- [improved training_curves.png](../Data/improved_cnn_outputs/run_20260326_151826/training_curves.png)
- [upgraded summary.json](../Data/improved_cnn_outputs/run_20260326_161003/summary.json)
- [upgraded training_curves.png](../Data/improved_cnn_outputs/run_20260326_161003/training_curves.png)
- [take 4 summary.json](../Data/improved_cnn_outputs/cnn_take_4_20260326_234323/summary.json)
- [take 4 training_curves.png](../Data/improved_cnn_outputs/cnn_take_4_20260326_234323/training_curves.png)
- [transfer summary.json](../Data/transfer_learning_outputs/run_20260329_030438/summary.json)
- [transfer best_transfer_model.pt](../Data/transfer_learning_outputs/run_20260329_030438/best_transfer_model.pt)
- [conv1_1_feature_maps.png](../Data/cnn_feature_maps_trained/run_20260325_130955/conv1_1_feature_maps.png)
- [conv3_2_feature_maps.png](../Data/cnn_feature_maps_trained/run_20260325_130955/conv3_2_feature_maps.png)
- [sample_images.png](../Outputs/eda/sample_images.png)
- [class_distribution.png](../Outputs/eda/class_distribution.png)
