# xprize-rainforest

This project is a pipeline for processing geospatial data, specifically for the purpose of detecting and segmenting trees in a forest. The pipeline includes several tasks such as tiling, detection, aggregation, and segmentation. Each task is configurable via YAML configuration files.

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/hugobaudchon/xprize-rainforest.git
cd pipelines-rainforest
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Configuration

The pipeline is configured using YAML files. Sample configuration files are provided in the `config/samples` directory. You can copy these files and modify them according to your needs.

Each configuration file has sections for different tasks. For example, the `xprize.yaml` file includes sections for tilerizer, detector, aggregator, and segmenter tasks. Each section contains parameters specific to that task.

## Usage

The main entry point of the pipeline is `main.py`. This script accepts command-line arguments specifying the task and subtask to perform, and the path to the configuration file.

Here is a basic usage example:

```bash
python main.py --task pipelines --config_path ./config/pipelines.yaml
```

## Tasks and Subtasks

The following table lists the available tasks and subtasks:

| Task       | Subtask    | Description                                                                                         | Config Sample                     |
|------------|------------|-----------------------------------------------------------------------------------------------------|-----------------------------------|
| pipeline   | xprize     | Runs the entire inference pipeline: pipeline-detector, pipeline-segmenter, pipeline-classifier.     | `pipeline_xprize_sample.yaml`     |
| pipeline   | detector   | Runs the detection inference pipeline: tilerizer, detector, aggregator, coco_to_geopackage.         | `pipeline_detector_sample.yaml`   |
| pipeline   | segmenter  | Runs the segmentation inference pipeline: tilerizer, segmenter, aggregator, and coco_to_geopackage. | `pipeline_segmenter_sample.yaml`  |
| pipeline   | classifier | Runs the classification inference pipeline: tiling, embedder, classifier.                           | `pipeline_classifier_sample.yaml` |
| tilerizer  | N/A        | Splits the input raster into tiles.                                                                 | `tilerizer_sample.yaml`           |
| detector   | train      | Trains the detector model.                                                                          | `detector_sample.yaml`            |
| detector   | score      | Scores the detector model.                                                                          | `detector_sample.yaml`                   |
| detector   | infer      | Runs inference using the detector model.                                                            | `detector_sample.yaml`                   |
| aggregator | N/A        | Aggregates the detection results.                                                                   | `aggregator_sample.yaml`          |
| segmenter  | infer      | Runs inference using the segmenter model.                                                           | `segmenter_sample.yaml`           |

For each task, you need to provide a configuration file with the appropriate sections filled out. For example, to run the detector 'train' subtask, you would use a command like this:

```bash
python main.py --task detector --subtask train --config_path ./config/detector.yaml
```

Please refer to the sample configuration files in the `config/samples` directory for examples of how to set up the configuration files.
