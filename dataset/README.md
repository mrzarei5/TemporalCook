# TemporalCook Benchmark Dataset

This repository contains the **TemporalCook** benchmark for evaluating temporal and procedural reasoning in image-based Visual Question Answering (VQA), specifically focused on cooking activities.

## Contents

- `TemporalCook.json`: The main dataset file in JSON format.  
- `images.zip`: Compressed folder containing all referenced images, organized by recipe type (~475MB). Downloadable as mentioned in the Images Download Instructions below.
- `recipe_video_ids.json`: File mapping recipe types to their corresponding YouCookII video IDs used as the knowledge source for retrieval-augmented generation (RAG).

## Images Download Instructions

Download the images from [Google Drive](https://drive.google.com/file/d/1a4WacU9CiesBezMHhU0ODqGREkhWIl_a/view?usp=sharing) and extract them to the `dataset/` directory:

```bash
cd dataset/
unzip images.zip
```

**Note**: The `images/` folder contains cooking activity images organized by recipe type (e.g., `pizza marghetta/`, `pad thai/`, `sushi/`, etc.).

## Dataset Structure

Each entry in `TemporalCook.json` maps a unique question ID to an object with the following fields:

| Field                     | Description                                                        |
|---------------------------|--------------------------------------------------------------------|
| `question`                | The question text.                                                 |
| `answer`                  | The ground truth answer to the question.                           |
| `image_name`              | The file name of the corresponding image.                          |
| `image_path`              | Relative path to the image file (within the `images/` directory).  |
| `category`                | The recipe or cooking activity related to this question.           |
| `recipe_type`             | The YouCookII recipe ID related to this question.                  |
| `video_id`                | The YouCookII video ID from which the question was generated.      |
| `videos_with_ground_truth`| List of instructional videos (from the knowledge base) containing the ground truth answer. Each item in this list is described below. |

### Details of `videos_with_ground_truth`

Each item in this list corresponds to a supporting instructional video, and has:

| Field                  | Description                                                         |
|------------------------|---------------------------------------------------------------------|
| `video_id`             | The YouCookII instructional video ID.                               |
| `procedure_step_ids`   | List of step ID(s) within the video that contain the ground truth answer (matching YouCookII step indices). |
| `procedure_step_texts` | List of the actual text of the procedure steps where the answer is found. |

## Details of `recipe_video_ids.json`

The `recipe_video_ids.json` file provides a mapping between each `recipe_type` and the corresponding YouCookII video IDs that are included in the video knowledge source for retrieval-augmented generation (RAG). Each video ID listed is the same as the official YouCookII video IDs.

**Structure:**
- Each entry consists of a `recipe_type` and a list of associated YouCookII `video_id`s.
- This file serves as an index of which videos are available as knowledge sources for each recipe type in the dataset.

Example format:
```
recipe_type_1: [video_id_1, video_id_2, ...]
recipe_type_2: [video_id_3, video_id_4, ...]
```

## TemporalCook Benchmark: Licensing and Usage

TemporalCook is constructed by extracting frames and annotations from the YouCookII instructional video dataset ([YouCookII website](https://youcook2.eecs.umich.edu/)), which as of 2024 makes all raw video files publicly available for **non-commercial, research purposes only**:

> "Due to requests and inaccessibility of online videos, we have begun to distribute the raw video files for non-commercial, research purposes only." —YouCookII

All images and video frames in this benchmark originate from YouCookII and are provided solely for academic research use under the same terms. **No commercial use is permitted.** Copyright for all video content remains with the original YouTube creators as credited in the YouCookII dataset.

If you are a copyright owner and wish to have your content removed, please contact us.

### Citation

If you use this dataset, please cite both our paper and the YouCookII dataset:

- Zhou et al., "Towards Automatic Learning of Procedures from Web Instructional Videos", AAAI 2018 ([YouCookII citation])

