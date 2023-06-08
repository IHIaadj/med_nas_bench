# MED-NAS-Bench

MED-NAS-Bench is a Neural Architecture Search (NAS) benchmark specifically designed for medical imaging tasks. It provides a comprehensive collection of architectures and their corresponding performance metrics on various medical imaging datasets. This repository serves as a central resource for researchers and practitioners in the field of medical imaging who are interested in exploring and comparing different NAS methodologies.

## Features

- **Diverse Architectures**: MED-NAS-Bench contains a wide range of neural architectures specifically designed for medical imaging tasks, including image classification, segmentation, and disease detection. The architectures cover different network depths, layer configurations, and architectural motifs commonly used in medical imaging applications.

- **Benchmark Datasets**: The benchmark includes popular medical imaging datasets, such as CIFAR-10, CIFAR-100, Colon, Hepatic Vessels, Spleen, and Chest X-Ray. These datasets cover a variety of imaging modalities, pathologies, and anatomical structures, allowing researchers to evaluate the generalization capabilities of different architectures across different medical imaging domains.

- **Performance Metrics**: For each architecture, MED-NAS-Bench provides a comprehensive set of performance metrics, including accuracy, dice score, sensitivity, specificity, and latency. These metrics enable researchers to assess the trade-offs between accuracy, computational efficiency, and other important factors in medical imaging tasks.

- **API and Evaluation Scripts**: The repository provides an easy-to-use API and evaluation scripts for querying the benchmark and computing performance metrics for user-defined architectures. This allows researchers to integrate MED-NAS-Bench into their own NAS frameworks or conduct comparative evaluations with their own architectures.

## Usage

To use MED-NAS-Bench, simply clone this repository and follow the instructions in the provided documentation. The documentation includes detailed information on the dataset format, architecture representation, API usage, and evaluation scripts.

```
git clone https://github.com/ihiaadj/MED-NAS-Bench.git
cd MED-NAS-Bench
```

## Contribution

We welcome contributions from the research community to enhance and expand MED-NAS-Bench. If you have new architectures, datasets, or performance metrics to contribute, please follow the guidelines outlined in the CONTRIBUTING.md file.


## License

MED-NAS-Bench is released under the [MIT License](LICENSE).

## Contact

For any inquiries or issues, please contact hadjer.benmeziane@uphf.fr

We hope that MED-NAS-Bench will serve as a valuable resource for researchers and practitioners in the field of medical imaging, facilitating advancements in neural architecture design and enabling breakthroughs in medical image analysis and diagnosis.

