# Human Segmentation System

This project provides a comprehensive framework for human segmentation, offering various models, loss functions, and dataset handling capabilities.

## Features

- **Multiple Models Supported**: Includes implementations of popular segmentation architectures such as UNet, DeepLabV3+, SegNet, and BiSeNet.
- **Flexible Loss Functions**: Supports a range of loss functions including Binary Cross-Entropy (BCE), Dice Loss, Focal Loss, and combinations like BCE-Dice Loss.
- **Dataset Handling**: Integrated support for datasets like CamVid and custom Portrait datasets.
- **Training and Inference Scripts**: Ready-to-use scripts for training models on specified datasets and performing inference on new images.
- **Tools for Experimentation**: Includes tools for learning rate range testing, common utilities, and evaluation metrics.

## Installation

To set up the development environment, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your_username/human-seg.git
    cd human-seg
    ```

2.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training

Training scripts are located in the `src/` directory.

-   **Training on Portrait Dataset**:
    ```bash
    python src/portrait_train.py --cfg config/portrait_config.py
    ```

-   **Training on CamVid Dataset**:
    ```bash
    python src/camvid_train.py --cfg config/camvid_config.py
    ```
    *(Note: Adjust `--cfg` path if your configuration files are located elsewhere or if you have custom configurations.)*

### Inference

Inference scripts are located in the `bins/` directory.

-   **BiSeNet Inference**:
    ```bash
    python bins/bisenet_inference.py --input_image_path /path/to/your/image.jpg --output_dir /path/to/output
    ```

-   **SegNet Inference**:
    ```bash
    python bins/segnet_inference.py --input_image_path /path/to/your/image.jpg --output_dir /path/to/output
    ```
    *(Note: These commands are examples. You may need to specify model weights, input/output paths, and other parameters according to the specific script's arguments.)*

## Datasets

The project is configured to work with the following datasets:

-   **CamVid Dataset**: Configuration can be found in `config/camvid_config.py` and dataset loader in `datasets/camvid_dataset.py`.
-   **Portrait Dataset**: Configuration can be found in `config/portrait_config.py` and dataset loader in `datasets/portrait_dataset.py`.

## Models

The following segmentation models are implemented in the `models/` directory:

-   **UNet**
-   **DeepLabV3+**
-   **SegNet**
-   **BiSeNet** (with configurable backbones like MobileNetV2)

## Contact

If you have any questions or need further assistance, please contact 1536825048@qq.com
