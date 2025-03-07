# OCR Captchas

This repository demonstrates a practice of building an Optical Character Recognition (OCR) model to read 4-digit CAPTCHA images using TensorFlow and Keras.

## Overview

The project focuses on developing an OCR system capable of recognizing 4-digit CAPTCHAs. The approach involves:

-   **Data Preparation**: Utilizing a dataset of CAPTCHA images where each image's filename corresponds to its label.
-   **Model Architecture**: Implementing a Convolutional Recurrent Neural Network (CRNN) combined with Connectionist Temporal Classification (CTC) loss to effectively recognize sequences in images.
-   **Training and Evaluation**: Training the model on the prepared dataset and evaluating its performance on unseen data.

## Dataset

The dataset comprises CAPTCHA images stored in the `captcha_images` directory. Each image filename represents the correct 4-digit label, facilitating supervised learning.

## Dependencies

To set up the environment, install the necessary dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the OCR model:

1. Ensure your dataset is in the `captcha_images` directory, with each image named according to its 4-digit label.
2. Run the training script:

    ```bash
    python ocr_captcha.py
    ```

    This script will preprocess the data, define the CRNN model, and commence training. Upon completion, the trained model will be saved for future use.

### Predicting with the Model

To make predictions on new CAPTCHA images:

1. Place the image you wish to predict in the root directory and name it `test.png`.
2. Execute the prediction script:

    ```bash
    python predict_example.py
    ```

    The script will load the trained model and output the predicted 4-digit code for `test.png`.

## Results

The model's performance can be visualized through training metrics and sample predictions. Below is an example of the learning curve:

![Learning Curve](learning_curve.jpg)

And a visualization of the model's architecture:

![Model Structure](model_structure.jpg)

## References

This project is inspired by various OCR implementations for CAPTCHA recognition, including:

-   [EVOL-ution/Captcha-Recognition-using-CRNN](https://github.com/EVOL-ution/Captcha-Recognition-using-CRNN)
-   [nbswords/ocr-captchas](https://github.com/nbswords/ocr-captchas)
-   [Keras Example: OCR model for reading Captchas](https://keras.io/examples/vision/captcha_ocr/)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

_Note: This README is based on the structure and information from similar OCR CAPTCHA projects and may need adjustments to align perfectly with the specific implementations and results of this repository._
