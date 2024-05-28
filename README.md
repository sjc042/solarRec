# WORK IN PROGRESS
# Satellite Imagery Analysis and Solar Capacity Estimation System 

## Introduction
This UW Capstone project (course EE497/598) is designed to acquire, process, and analyze satellite imagery to detect solar panel arrays on residential rooftops and estimate the solar energy generation capacity. It integrates with satellite imagery services to provide granular energy generation estimates for urban planning and renewable energy assessments.

## Getting Started

### Prerequisites
- Python 3.8+
- CUDA Toolkit 11.8+
- PyTorch 2.2.2+
- Ultralytics
- API key for accessing Google Solar API and Maps Static API

### Downloading Dataset
- For our instance segmentation dataset with images collected from Google Solar: https://universe.roboflow.com/solarrecmoreimage/solarrecseg-y4it9
- For bouding box dataset derived from [Duke University's dataset (Bradley et al.)](http://dx.doi.org/10.6084/m9.figshare.3385780.v1): https://universe.roboflow.com/solarrecmoreimage/duke-solar

<!-- TODO: include overview of custom instance segmentation dataset. -->

### Installation
Follow these steps to set up the project environment:

1. **Create a Python Environment with Conda**  
   Use Conda to create a new environment named `solarRec` with Python 3.8:
   ```bash
   conda create -n solarRec python=3.8
   ```
2. **Install PyTorch with CUDA Support**  
    Install the compatible version of PyTorch for your CUDA version. Visit PyTorch's official site for more details: https://pytorch.org/get-started/locally/. Example:
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
3. **Clone the Repository**  
    ```bash
    git clone https://github.com/sjc042/solarRec.git
    ```
4. **Install Required Python Packages**  
Navigate to the cloned project folder and install the necessary Python packages as specified in the requirements.txt:
    ```bash
    cd solarRec
    pip install -r requirements.txt
    ```
5. **Save Google Solar API to env file**  
Navigate to main folder and create .env file with added line: `'GOOGLE_SOLAR_API_KEY' = '{your_key}'`

**DEV NOTE**:  
- requirements.txt to be modified
- include overview of custom instance segmentation dataset


## Usage
1. Configure the API keys for Google Solar API in the configuration file.
**DEV NOTE: implement configuration for API.**
2. Run the main script to start the image processing and analysis:
    ```bash
    python main.py
    ```

## Features
- **Image Acquisition**: Downloads and processes GeoTiff images from satellite imagery services.
- **Object Detection**: Detects solar panels on rooftops from processed images.
- **Solar Capacity Estimation**: Estimates the BTM photovoltaic solar generation capacity from detected solar panels.

## Folder Structure
- `/solar-data-root`
    - `/images`
        - `rgb_address1.jpg`
        - `rgb_address2.jpg`
        - ...
   - `/addresses`
        - `/address1`
            - `monthlyFlux_address1.tif`
            - `DSM_address1.tif`
            - `mask_address1.tif`
            - `address1.json`
        - `/address2`
            - `monthlyFlux_address2.tif`
            - `DSM_address2.tif`
            - `mask_address2.tif`
            - `address2.json`
        - ...

## Documentation
- User Manual: Detailed instructions on system operation and configuration.
- API Documentation: Information on integrating and using various APIs with the system.

## Contributing
Contributions to the project are welcome. Please ensure to follow the best practices and coding standards outlined in the contribution guidelines.

## License
This project is licensed under the AGPL-3.0 License - see the [LICENSE.txt](LICENSE.txt) file for details.

## Acknowledgements
- Mentors: Anil Jampala (PSC), David Thomas (PSC), Daniel S Kirschen (UW)
- Capstone TA: Mingfei Chen
- Quality Assurance: Watt-Watchers team
- Developers and all contributors who have invested their time and effort.

## References
1. Bradbury, Kyle; Saboo, Raghav; Malof, Jordan; Johnson, Timothy; Devarajan, Arjun; Zhang, Wuming; et al. (2016). Distributed Solar Photovoltaic Array Location and Extent Data  Set for Remote Sensing Object Identification. figshare. Dataset. https://doi.org/10.6084/m9.figshare.3385780.v1
2. Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLO (Version 8.0.0) [Computer software]. https://github.com/ultralytics/ultralytics
3. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., ... & Girshick, R. (2023). Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 4015-4026).

## Contact
For any inquiries, please contact [cielsun042@gmail.com](mailto:cielsun042@gmail.com).

## Appendix
- **A**: Glossary
- **B**: Acronyms
    - BTM: Behind-The-Meter
    - PV: Photovoltaic
    - DSM: Digital Surface Model
    - AP: Average Precision
