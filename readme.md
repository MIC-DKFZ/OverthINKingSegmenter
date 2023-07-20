# OverthINKingSegmenter

This was our contribution for 7th place in the [Vesuvius Challenge - Ink Detection](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection).

We are very grateful for the organization of this very interesting competition. Sincere thanks to the hosts and the Kaggle team. A big thank you to Yannick and Max, it was lots of fun working with you! For all of us it was the first Kaggle competition. 

## Summary

In this competition submission, we used the [nnU-Net framework](https://github.com/MIC-DKFZ/nnUNet), a recognized powerhouse in the medical imaging domain. The success of nnU-Net is validated by its adoption in winning solutions of 9 out of 10 challenges at [MICCAI 2020](https://arxiv.org/abs/2101.00232), 5 out of 7 in MICCAI 2021 and the first place in the [AMOS 2022](https://amos22.grand-challenge.org/final-ranking/) challange.

To tackle the three-dimensional nature of the given task, we designed a custom network architecture: a 3D Encoder 2D Decoder U-Net model using Squeeze-and-Excitation (SE) blocks within the skip connections. We used fragment-wise normalization and selected 32 slices for training, enabling us to extend the patch size to 32x512x512. We divided each fragment into 25 pieces and trained five folds. We only submitted models that performed well on our validation data. For the final submission, we ensambled the weights of two folds (zero and two) from two respective models. One model was trained with a batch size of 2 and a weight decay of 3e-5, while the other was trained with a batch size of 4 and a weight decay of 1e-4. During test time augmentation, we implemented mirroring along all axes and in-plane rotation of 90°, resulting in a total of eight separate predictions per model for each patch. 

For the two submissions we chose two different postprcessing techniques. The first approach involved setting the threshold of the softmax outputs from the network from 0.5 to 0.6. As a second step we conducted a connected component analysis to eliminate all instances with a softmax 95th percentile value below 0.8. The second approach involved utilizing an off-the-shelf 2D U-Net model with a patch size of 2048x2048 on the softmax outputs of the first model. The output was resized to 1024x1024 for inference and then scaled up to 2048x2048. The intention behind this step was to capture more structural elements, such as the shape of letters, due to the higher resolution of the input. We regret both since they only improved results for public testset.

## Model Architecture

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F6289140%2Fbf19626f416be872fbb0b56b92a6c585%2FModel.png?generation=1686847814703959&alt=media)

As been mentioned, we chose a 3D Encoder 2D Decoder U-Net model using SE blocks within the skip connections. Therefore, the selected slices were still seen as a 3D input volume by our network. After passing through the encoder, the features were mapped to 2D on all levels (i.e., skip connections) using a specific weighting. One unique aspect of the network to highlight is that the encoder contained four convolutions in each stage to process the difficult 3D input, whereas the decoder only had two convolutional blocks.

The mapping was initially performed using a simple average operation but was later refined with the use of Squeeze-and-Excitation. However, instead of applying the SE on the channel dimension — as is usually done to highlight important channels — we applied one SE Block per level (i.e., skip) to all channels, but on the x-dimension. This results in a weighting of the slices in feature space, so when aggregating with the average operation later, each slice has a different contribution.

## Preprocessing

In the preprocessing stage, we cropped each fragment into 25 parts and ensured they contain an equal amount of data points (area labeled as foreground in the mask.png). This process was performed to create five folds for training.

For the selection of the 32 slices, we calculated the intensity distributions for each individual fragment. From these distributions, we determined the minimum and maximum values and calculated the middle point between them. We then cropped 32 slices around this chosen central slice. The following plot shows the intensity distribution for the individual fragments. The vertical line in the plot represents the midpoint between the maximum and minimum intensity values.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F6289140%2F1fd4a8e4dbccd46f59d3ca92b19fb74d%2Fintesity_dist.png?generation=1686847846933995&alt=media)

To further preprocess the data, we applied z-scoring to normalize the intensity values. Additionally, we clipped the intensity values at the 0.5 and 99.5 percentiles to remove extreme outliers. Finally, we performed normalization on each individual fragment to ensure consistency across the dataset.

## Training

Here are some insights into our training pipeline.

### Augmentation

For the most part, we utilized the out-of-the-box augmentation techniques provided by nnU-Net, a framework specifically designed for medical image segmentation. These techniques formed the foundation of our data augmentation pipeline. However, we made certain modifications and additions to tailor the augmentation process to our specific task and data characteristics:

- Rotation: We performed rotations only in the plane, meaning we applied rotations along the y and z axes. Out-of-plane rotations were considered as a measure to ensure stability but were not implemented.
- Scaling: We introduced scaling augmentation, allowing the data to be randomly scaled within a certain range. This helped to increase the diversity of object sizes in the training data.
- Gaussian Noise: We added Gaussian noise to the data, which helps to simulate realistic variations in image acquisition and improve the model's ability to handle noise.
- Gaussian Blur: We applied Gaussian blur to the data, with varying levels of blurring intensity. This transformation aimed to capture the variations in image quality that can occur in different imaging settings.
- Brightness and Contrast: We incorporated brightness and contrast augmentation to simulate variations in lighting conditions.
- Simulate Low Resolution: We introduced a transformation to simulate low-resolution imaging by randomly zooming the data within a specified range. This augmentation aimed to make the model more robust to lower resolution images.
- Gamma Transformation: We applied gamma transformations to the data, which adjusted the pixel intensities to enhance or reduce image contrast. This augmentation technique helps the model adapt to different contrast levels in input images.
- Mirror Transform: We employed mirroring along specified axes to introduce further variations in object orientations and appearances.

### Training Curves for Submission Folds

Batch size of 4 and a weight decay of 1e-4. Fold 0.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F6289140%2Fb0086c8f134adc9b58a78097cec48151%2Fprogress_0_0.png?generation=1686847908240065&alt=media)
Batch size of 4 and a weight decay of 1e-4. Fold 2.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F6289140%2F5afa0a5b3cc9890b40155af6265c9617%2Fprogress_0_2.png?generation=1686847917423689&alt=media)
Batch size of 2 and a weight decay of 3e-5. Fold 0.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F6289140%2Facd870b03c91b9f4f6875b9d01a7858f%2Fprogress_1_0.png?generation=1686847926934043&alt=media)
Batch size of 2 and a weight decay of 3e-5. Fold 2.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F6289140%2F7cd38547bc3d2d76aa0050889c09ed2a%2Fprogress_1_2.png?generation=1686847935394605&alt=media)

## Inference

We made the decision to submit only the ensemble of fold 0 and fold 2 models due to their superior performance on the validation data compared to the other folds. The difference in Dice score between these two folds and the rest of the folds was substantial, with a margin of ~0.1. As a result, we deemed the ensemble of fold 0 and fold 2 models to be the most reliable and effective for our final submission (we fools).

Additionally, we made the decision to incorporate test time augmentation (TTA) techniques during the inference process. TTA involved applying mirroring along all axes and in-plane rotation of 90° to each patch. By performing these augmentations, we generated a total of eight separate predictions per model for each patch.

### Post Processing

In a moment of desperate determination to achieve better results on the public test set, one audacious team member decided to dive headfirst into the realm of advanced post-processing. This daring soul concocted a daring plan: raise the threshold of the softmax outputs from a mundane 0.5 to a daring 0.6. But that was just the beginning!

Undeterred by caution, the same intrepid individual embarked on a quest to conduct a connected component analysis, mercilessly discarding all instances with a lowly softmax 95th percentile value below the illustrious threshold of 0.8.

With fervor and a touch of madness, this brave adventurer tested countless combinations of thresholds, determined to find the golden ticket to enhanced validation scores across all folds. A relentless pursuit of validation improvement that knew no bounds.

On the public test set, this fearless undertaking delivered a substantial boost of 0.05 dice points, raising hopes and spirits across the team. The unexpected improvement injected a renewed sense of excitement and optimism.

However, as fate would have it, on the ultimate battlefield of the 50% final, the outcome took a peculiar twist. The gains dwindled ever so slightly, with a meager decrease of -0.002 dice points. Though the difference may seem minuscule, in the realm of fierce competition, every decimal point counts.

Sorry guys.

### 2D Unet Refinement 

The second approach involved employing a 2D U-Net model with a patch size of 2048x2048 on the softmax outputs generated by the first model. Subsequently, the model's output was resized to 1024x1024 for inference purposes and then scaled up to the original resolution of 2048x2048. The rationale behind this strategy was to leverage the higher resolution input data to capture finer structural details, including the intricate shapes of letters. The training data for our this model was derived from inferences made by our various trained models on the original training data.

Whose idea was this?

Sorry again.

## Steps to reproduce our submission

Our submission can be easily reproduced following the following steps. We will also share python scripts to easily do data preparation, training and inference with minimal manual intervention.

### Dataset preparation

After downloading the data from the [challenge website](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/data) you need to prepare the data for nnUNet. This is rather straightforward, as we have adopted the dataset structure for this challenge and use a custom reader writer class. The only thing you really need to do is the conversion of the ground truth labels from png to tif files. The scans are saved as folders containing all 65 slices in the original tif format.

```
  nnUNet_raw/Dataset800_vesuvius/
  ├── dataset.json
  ├── imagesTr
  │   ├── scroll_1_0000
  |   |   ├── 00.tif
  |   |   ├── 01.tif
  |   |   ├── ...
  |   |   └── 64.tif
  │   ├── scroll_2_0000
  |   |   ├── ...
  │   └── scroll_3_0000
  |   |   ├── ...
  └── labelsTr
      ├── scroll_1.tif
      ├── scroll_2.tif
      └── scroll_3.tif
```

The dataset.json should look similar to this:
```
{
    "channel_names": {
        "0": "CT16"
    },
    "description": "dataset for the vesuvius challenge",
    "file_ending": ".tif",
    "labels": {
        "background": 0,
        "foreground": 1
    },
    "overwrite_image_reader_writer": "Vesuvius3D2DIO",
    "name": "Dataset800_vesuvius",
    "numTraining": 3,
    "reference": "",
    "release": "0.0"
}
```
Note, that it is important to overwrite the image reader/writer such that nnUNet can handle the changed dataset structure!

### Data preprocessing

Preprocessing of the dataset can be done by running the `nnUNetv2_plan_and_preprocess` command with the following arguments:

`nnUNetv2_plan_and_preprocess -d 800 -fpe DatasetFingerprintExtractor3D2DSliceselect -npfp 1 -pl ExperimentPlanner3D2DSliceselect -preprocessor_name Preprocessor3D2DSliceselect -c 3d_fullres -np 1`

This command will perform the preprocessing outlined in our method description. It will take care of the extraction of the 32 slices used by the model as well as the individual intensity normalization. The data is saved in the nnUNet_preprocessed folder as npz files.

You will need at least 64GB RAM to run the preprocessing for this dataset! Do not change the number of processes unless you are sure that it will fit into memory!

### Splitting up the dataset

In order to enable training 5 fold cross validation the data needs to be split into parts. The Jupyter Notebook `dataset_splitting.ipynb` found in `nnunetv2/notebooks` takes care of this.

### Training of the model

For the final submission we ensembled two models, both trained on folds 0 and 2. To train these models yourself you can run the commands
```
nnUNetv2_train 801 3d_fullres 0 -tr nnUNetTrainer3DSqEx2D -p nnUNetPlans_large_4conv
nnUNetv2_train 801 3d_fullres 2 -tr nnUNetTrainer3DSqEx2D -p nnUNetPlans_large_4conv
nnUNetv2_train 801 3d_fullres 0 -tr nnUNetTrainer3DSqEx2D_wd -p nnUNetPlans_bs4_large_4conv
nnUNetv2_train 801 3d_fullres 2 -tr nnUNetTrainer3DSqEx2D_wd -p nnUNetPlans_bs4_large_4conv
```

The trained models will be saved into the `nnUNet_results` folder. nnUNet saves the best checkpoint (at the highest ema) and the final checkpoint and run a final validation and evaluation of the results on the whole validation split for both. The produced summary files can be used to determine which of the checkpoints to use for inference.

### Inference on test data

Inference is done with the custom predict_overthinking_segmenter_32ind_tta_helper function, which is a helper function for the predict_overthinking_segmenter_32ind_tta function. It can be imported by running

`from nnunetv2.inference.predict_overthinking_segmenter_32ind_tta import predict_overthinking_segmenter_32ind_tta_helper`

The function can then be called by
```
    predict_overthinking_segmenter_32ind_tta_helper(
        input_folder,
        output_folder,
        nnUNetTrainer,
        conf_dir
    )
```
where input_folder contains the cases in the challenge format and conf_dir contains all checkpoint files, the plans file and the dataset file.

### Pretrained weights

Pretrained weights, as well as scripts for (pre)-processing of the raw challenge data, training and inference and a detailed documentation can now be found at [Zenodo](https://zenodo.org/record/8169325).

## Preliminary Last Words

More details will follow, cheers!