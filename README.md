# FoodSeg103
[DataSet](https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1) 

# A Large-Scale Benchmark for Food Image Segmentation

By [Xiongwei Wu](http://xiongweiwu.github.io/), [Xin Fu](https://xinfu607.github.io/), Ying Liu, [Ee-Peng Lim](http://www.mysmu.edu/faculty/eplim/), [Steven C.H. Hoi](https://sites.google.com/view/stevenhoi/home/), [Qianru Sun](https://qianrusun.com/).
  
![image](https://github.com/carrycooldude/FoodSeg103/assets/41143496/c7f8e2e1-8f23-4640-9c4a-b44374e5580d)
<br />

## Introduction

We build a new food image dataset FoodSeg103 containing 7,118 images. We annotate these images with 104 ingredient classes and each image has an average of 6 ingredient labels and pixel-wise masks.
In addition, we propose a multi-modality pre-training approach called ReLeM that explicitly equips a segmentation model with rich and semantic food knowledge.

In this software, we use three popular semantic segmentation methods (i.e., Dilated Convolution based, Feature Pyramid based, and Vision Transformer based) as baselines, and evaluate them as well as ReLeM on our new datasets. We believe that the FoodSeg103 and the pre-trained models using ReLeM can serve as a benchmark to facilitate future works on fine-grained food image understanding. 

Please refer our [paper](https://arxiv.org/abs/2105.05409) and our [homepage](https://xiongweiwu.github.io/foodseg103.html) for more details.


![Screenshot from 2024-07-03 01-16-25](https://github.com/carrycooldude/FoodSeg103/assets/41143496/04bb0cee-1279-408b-9e53-0e5ca6610b38)

### Task 1: Food Object Detection Algorithm

#### 1. Architecture Diagram and Documentation

**a) Possible Solution Approaches**

1. **Convolutional Neural Networks (CNN)**
   - **Approach**: Use a standard CNN architecture such as U-Net or Mask R-CNN for pixel-wise segmentation of food items.
   - **Pros**: Well-suited for image segmentation tasks, robust performance on large datasets, and availability of pre-trained models.
   - **Cons**: Computationally intensive, requires significant training time, and might require fine-tuning for specific datasets.

2. **Fully Convolutional Networks (FCN)**
   - **Approach**: Extend CNNs by replacing fully connected layers with convolutional layers, enabling the network to output spatial maps instead of class scores.
   - **Pros**: Efficient for segmentation tasks, end-to-end trainable.
   - **Cons**: Requires extensive data augmentation, potential overfitting on small datasets.

3. **SegNet**
   - **Approach**: Encoder-decoder architecture designed for pixel-wise segmentation.
   - **Pros**: Efficient in terms of memory and computational requirements, effective for semantic segmentation.
   - **Cons**: May not capture fine details as effectively as other architectures.

4. **DeepLabV3+**
   - **Approach**: Combines atrous convolution with spatial pyramid pooling for accurate segmentation.
   - **Pros**: High accuracy, effective for capturing multi-scale context.
   - **Cons**: High computational cost, complex to implement and tune.

**b) Possible Problems with These Solutions**

- **Data Augmentation**: Essential to avoid overfitting but can be computationally expensive.
- **Computational Resources**: High computational power and memory requirements.
- **Fine-tuning**: Pre-trained models might require extensive fine-tuning for specific datasets.
- **Class Imbalance**: Handling imbalance in the dataset where some food categories might be underrepresented.
- **Real-time Performance**: Ensuring the model performs well in real-time applications.

#### 2. Implementing a Workable Solution

We'll implement the U-Net architecture as a workable solution for this task. U-Net is a well-known architecture for image segmentation and is suitable for our dataset and task requirements.

**Implementation Steps:**

1. **Install Necessary Libraries**
   ```python
   !pip install tensorflow opencv-python-headless
   ```

2. **Import Libraries**
   ```python
   import tensorflow as tf
   from tensorflow.keras.models import Model
   from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
   from tensorflow.keras.preprocessing.image import ImageDataGenerator
   import numpy as np
   import cv2
   import os
   ```

3. **Define U-Net Architecture**
   ```python
   def unet(input_size=(256, 256, 3)):
       inputs = Input(input_size)
       conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
       conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
       pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

       # Add more layers here to complete the U-Net architecture...
       # Example final layers:
       up8 = concatenate([UpSampling2D(size=(2, 2))(conv1), conv1], axis=3)
       conv8 = Conv2D(64, 3, activation='relu', padding='same')(up8)
       conv8 = Conv2D(64, 3, activation='relu', padding='same')(conv8)
       
       conv9 = Conv2D(1, 1, activation='sigmoid')(conv8)
       
       model = Model(inputs=[inputs], outputs=[conv9])
       return model
   ```

4. **Load Images and Masks**
   ```python
   def load_images_and_masks(image_paths, mask_paths, target_size=(256, 256)):
       images = []
       masks = []
       for img_path, mask_path in zip(image_paths, mask_paths):
           img = cv2.imread(img_path)
           if img is None:
               print(f"Warning: Failed to load image at {img_path}")
               continue
           img = cv2.resize(img, target_size)
           img = img / 255.0  # Normalize to [0, 1]
           images.append(img)

           mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
           if mask is None:
               print(f"Warning: Failed to load mask at {mask_path}")
               continue
           mask = cv2.resize(mask, target_size)
           mask = mask / 255.0  # Normalize to [0, 1]
           masks.append(mask)
       return np.array(images), np.array(masks)

   train_image_paths = [os.path.join('Images/img_dir/train', f) for f in os.listdir('Images/img_dir/train')]
   train_mask_paths = [os.path.join('Images/ann_dir/train', f) for f in os.listdir('Images/ann_dir/train')]
   test_image_paths = [os.path.join('Images/img_dir/test', f) for f in os.listdir('Images/img_dir/test')]
   test_mask_paths = [os.path.join('Images/ann_dir/test', f) for f in os.listdir('Images/ann_dir/test')]

   train_images, train_masks = load_images_and_masks(train_image_paths, train_mask_paths)
   test_images, test_masks = load_images_and_masks(test_image_paths, test_mask_paths)

   if train_images.size == 0 or train_masks.size == 0:
       raise ValueError("No training data loaded. Please check your dataset paths and file extensions.")
   ```

5. **Data Augmentation**
   ```python
   data_gen_args = dict(rotation_range=15,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        shear_range=0.01,
                        zoom_range=[0.9, 1.25],
                        horizontal_flip=True,
                        vertical_flip=True,
                        fill_mode='reflect')

   image_datagen = ImageDataGenerator(**data_gen_args)
   mask_datagen = ImageDataGenerator(**data_gen_args)

   seed = 1
   image_datagen.fit(train_images, augment=True, seed=seed)
   mask_datagen.fit(train_masks, augment=True, seed=seed)
   ```

6. **Create Generators**
   ```python
   def create_generators(image_datagen, mask_datagen, train_images, train_masks, batch_size=8):
       image_generator = image_datagen.flow(train_images, batch_size=batch_size, seed=seed)
       mask_generator = mask_datagen.flow(train_masks, batch_size=batch_size, seed=seed)
       train_generator = zip(image_generator, mask_generator)
       return train_generator

   train_generator = create_generators(image_datagen, mask_datagen, train_images, train_masks)
   validation_generator = create_generators(image_datagen, mask_datagen, test_images, test_masks)
   ```

7. **Train the Model**
   ```python
   model = unet()
   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

   model.fit(train_generator, epochs=10, validation_data=validation_generator)
   ```

This implementation provides a basic setup for training a U-Net model on the FoodSeg103 dataset. Further fine-tuning and optimizations can be done based on performance evaluations and resource availability.


### Further Fine-Tuning and Optimizations

Fine-tuning and optimizing a U-Net model for food object detection involves several strategies to improve performance and efficiency. Here are some approaches:

#### 1. **Hyperparameter Tuning**
- **Learning Rate**: Experiment with different learning rates and schedules (e.g., step decay, exponential decay).
- **Batch Size**: Try different batch sizes to see which one yields better performance.
- **Optimizer**: Test different optimizers such as SGD with momentum, RMSprop, or Adam with different parameter settings.

#### 2. **Data Augmentation**
- **Augmentation Techniques**: Increase the variety of augmentation techniques (e.g., color jitter, contrast adjustment) to make the model more robust.
- **Augmentation Intensity**: Adjust the intensity and probability of each augmentation technique.

#### 3. **Model Architecture**
- **Depth and Width**: Experiment with deeper or wider architectures by adding more layers or increasing the number of filters.
- **Skip Connections**: Modify the skip connections in the U-Net to better capture fine details.

#### 4. **Loss Function**
- **Custom Loss Functions**: Design custom loss functions that better handle class imbalance or specific characteristics of the dataset.
- **Dice Loss**: Use Dice coefficient loss, IoU loss, or a combination of these with binary cross-entropy.

#### 5. **Regularization**
- **Dropout**: Introduce dropout layers to prevent overfitting.
- **Weight Decay**: Apply L2 regularization to the weights of the model.

#### 6. **Post-Processing**
- **Conditional Random Fields (CRFs)**: Use CRFs or other post-processing techniques to refine the segmentation results.
- **Smoothing**: Apply smoothing techniques to the predicted masks.

#### 7. **Transfer Learning**
- **Pre-trained Models**: Start with a pre-trained model on a similar dataset and fine-tune it on the FoodSeg103 dataset.
- **Layer Freezing**: Freeze the initial layers of the pre-trained model to retain learned features and fine-tune the later layers.

#### 8. **Training Strategies**
- **Class Balancing**: Implement techniques to balance the classes during training, such as oversampling minority classes.
- **Curriculum Learning**: Start training with simpler examples and gradually increase the complexity.

#### 9. **Evaluation and Validation**
- **Cross-Validation**: Use k-fold cross-validation to better estimate model performance.
- **Validation Metrics**: Track additional metrics such as precision, recall, F1-score, and IoU during training.

#### 10. **Resource Optimization**
- **Mixed Precision Training**: Use mixed precision training to speed up computation and reduce memory usage.
- **Distributed Training**: Utilize multiple GPUs or distributed computing resources to speed up training.

### Implementation of Fine-Tuning and Optimizations

Here are some examples of how you might implement these optimizations in code:

**Learning Rate Scheduler Example**
```python
from tensorflow.keras.callbacks import LearningRateScheduler

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

callback = LearningRateScheduler(scheduler)
```

**Data Augmentation Example**
```python
data_gen_args = dict(rotation_range=20,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     shear_range=0.02,
                     zoom_range=[0.8, 1.2],
                     horizontal_flip=True,
                     vertical_flip=True,
                     brightness_range=[0.8, 1.2],
                     fill_mode='reflect')

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)
```

**Custom Loss Function Example**
```python
import tensorflow as tf

def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - (numerator / (denominator + tf.keras.backend.epsilon()))

model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy'])
```

**Dropout and Regularization Example**
```python
from tensorflow.keras.layers import Dropout

def unet(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    conv1 = Dropout(0.5)(conv1)  # Dropout layer
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Add more layers here to complete the U-Net architecture...
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv1), conv1], axis=3)
    conv8 = Conv2D(64, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same')(conv8)
    conv8 = Dropout(0.5)(conv8)  # Dropout layer

    conv9 = Conv2D(1, 1, activation='sigmoid')(conv8)

    model = Model(inputs=[inputs], outputs=[conv9])
    return model
```

### Summary
Fine-tuning and optimizing a U-Net model for food object detection involves a combination of hyperparameter tuning, data augmentation, architectural adjustments, custom loss functions, regularization techniques, post-processing methods, transfer learning, advanced training strategies, comprehensive evaluation, and resource optimization. Implementing these strategies can significantly improve the performance and efficiency of the model.
