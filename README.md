# Gaze Mapping Algorithm for Free Viewing Dynamic Virtual Environments

## Summary
This repository contains the source code for the paper [Gaze Mapping Algorithm for Free Viewing Dynamic Virtual Environments](https://www.frontiersin.org/articles/10.3389/frvir.2022.802318/full) a novel gaze mapping approach for free viewing conditions in dynamic immersive virtual environments. The algorithm projects eye fixation data from multiple users, who viewed the virtual environment from different perspectives, to the current view. This generates eye fixation maps that can be used as ground truth for training machine learning models to predict saliency and the user's gaze in immersive virtual reality environments. The algorithm utilizes a flexible image retrieval approach based on SIFT features and a vocabulary tree to efficiently match and project fixations from different viewpoints. The homography transformation re-projects the fixations to the current view, and a Gaussian filter is applied to calculate the final eye fixation map.

## Installation and Dependencies
### Required Software
To install the application please make sure you have the following programs installed:
- git
- gcc, clang, or msvc with at least C++17 support
- cmake >= 3.10

### Required Libraries
- OpenCV >= 4.2
- Boost with the system, thread, filesystem, program_options and iostream subpackages installed

We recommend installing the dependencies through [vcpkg](https://vcpkg.io/en/).

### Compile the Application

To compile the application run

```
git clone --recursive https://github.com/TheTimmy/gazemapping.git
cd gazemapping
cmake -Bbuild -DCMAKE_TOOLCHAIN_FILE="<vcpkg-root>/scripts/buildsystems/vcpkg.cmake"
cmake --build build --config Release
```

If you did not install the dependencies through vcpkg and cmake did not find them, you have to set the paths to OpenCV and boost manually.


## Usage
To run the application it requires to write a configuration file for each set of videos and gaze points.
These need to contain the following information, in order to run correctly.
Here, we provide a brief explaination of the configuration file. However, more can be found in the repository.

```
{
    # Offset between video and gaze data if the video starts earlier than the captured gaze. This is mainly for alignment
    "timestamps": [
        0,
        0,
        0,
    ],

    # Path to the gaze data in screen space coordinates from 0 to 1
    "gaze": [
        "gaze_data-0.txt",
        "gaze_data-1.txt",
        "gaze_data-2.txt",
    ], 

    # Path to the videos that were captured along with the gaze data
    "videos": [
        "video-0.mp4",
        "video-1.mp4",
        "video-2.mp4",
    ],

    # output file for the vocabulary
    "vocabulary": "data/vocabulary",
    "filename": "data/vocabulary/example",

    # Tracking rate of the eye tracker as fraction of the video recording rate
    # In this case 1.0 means that both the gaze data and video were recorded at the same speed.
    "tracker_rate": 1.0,

    # For evaluation purposes choose between [cross validation, temporal cross validation, past temporal cross validation, none]
    "evaluation": "cross validation",

    # Configuration for the feature extraction
    "features": {
        # Debug parameter
        "visualize": false,

        # Minimum number of keypoints to extract for an image to be considered
        "min_keypoints": 10,

        "output_directory": "",

        # Parameters of the extractor, here we use a SIFT feature extractor
        "extractor": {
            "type": "SIFT",
            "nfeatures": 0,
            "nOctaveLayers": 3,
            "contrastThreshold": 0.04,
            "edgeThreshold": 10,
            "sigma": 1.6
        }
    },

    # Parameters for the vocabulary tree construction
    "tree": {
        # Number of clusters per hierarchy
        "clusters": 10,

        # Number of layers in the hierarchy
        "height": 6,

        # The batch size for the mini-batch kmeans algorithm
        "batch_size": 5000000,

        # Minimum number of descriptors per layer and cluster
        "min_descriptors": 1000,

        # Maximum number of mini-batch kmeans iterations
        "max_iterations": 250,

        # Early stopping for mini-batch kmeans
        "max_no_improvements": 10,

        # Early stopping tolerance for mini-batch kmeans
        "tolerance": 1e-7,

        # Only use every 10th sample during tree generation
        "subsample": 10,

        # Perform everything in memory, to avoid slowdown through the hard drive
        "in_memory": true
    },

    # Parameters for the matching algorithm to find similar images
    "matching": {
        # Maximum number of matches we consider during fixation mapping
        "max_match_count": 65536,

        # Minimum number of matches we consider during fixation mapping
        "min_match_count": 10,

        # The ratio of two matches to be considered similar
        "filter_ratio": 0.75,

        # Unused parameter
        "match_ratio": 0.99,

        # Count of the matches returned per query vector
        "k-nearest": 2,

        # Norm to consider two match vectors as close
        "norm": "L2",

        # Perform everything in memory, to avoid slowdown through the hard drive
        "in_memory": true
    },


    "fixations": {
        # Final output directory for the generated saliency maps
        "output_directory": "data/saliency/",

        # The patch size to consider. See paper for details
        "patch_size": { "x": 128, "y": 128 },

        # Norm to consider two match vectors as close
        "norm": "L2",

        # Unused parameter
        "temporal_range": 1,

        # Unused parameter
        "temporal_offset": 1,

        # Maximum number of matches we consider during fixation mapping
        "max_match_count": 65536,

        # Minimum number of matches we consider during fixation mapping
        "min_match_count": 10,

        # Optimization of gaze after inital gathering.
        # This will share gaze with adjacent frames to generate a smoother output
        "optimize_gaze": 128,

        # How often we optimize the gaze
        "optimize_steps": 1,

        # Count of the matches returned per query vector
        "k-nearest": 2,

        # The ratio of two matches to be considered similar
        "filter_ratio": 0.75,

        # Unused parameter
        "match_ratio": 0.99,

        # Applied blur to the fixation map for the generation of the saliency map 
        "sigma": 14.0,

        # Size of the filter used for non-maximum supression for saliency map generation
        "suppression_size": -1,

        # Threshold used to supress fixation points 
        "supression_threshold": 0.0,

        # RANSAC projection threshold for image alignment
        "projection_threshold": 1.5,

        # Avoid using image patches and search with the full image instead
        "search_with_full_train_frame": false,

        # Restrict the training images to the foveal patches only
        "restrict_train_frame_to_foveal_patch": false,

        # Restrict the query images to the foveal patches only
        "restrict_query_frame_to_foveal_patch": true,

        # Perform everything in memory, to avoid slowdown through the hard drive
        "in_memory": true
    }
}
```

To generate saliency maps through our gaze mapping algorithm, three steps need to be run.
1. Computation of image features through SIFT. To run this command you have to call GazeMapper like this: \
   `GazeMapper --command desc --config "../config.json"`

2. Generation of a vocabulary tree. To generate the vocabulary tree from the image features, run GazeMapper through: \
   `GazeMapper --command tree --config "../config.json"`

3. Fixation map computation from the vocabulary tree
   `GazeMapper --command fixation --config "../config.json"`


## Results and Evaluation
- The algorithm was evaluated on the DGaze and Saliency in VR datasets, showing promising results in modeling the gaze of the current user compared to existing saliency predictors. For details see our paper.

## Contributing
- Contributions are welcome.

## License
- This project is licensed under [MIT License].

## Citation
If you use this gaze mapping algorithm or work based on it, please cite the original research paper:

```
@article{rolff2022gaze,
  title={Gaze Mapping for Immersive Virtual Environments Based on Image Retrieval},
  author={Rolff, Tim and Steinicke, Frank and Frintrop, Simone},
  journal={Frontiers in Virtual Reality},
  volume={3},
  pages={802318},
  year={2022},
  publisher={Frontiers}
}
```

Thank you for acknowledging our contribution.

For more details on the algorithm and its implementation, please refer to the full research paper. If you have any questions or need further assistance, feel free to reach out to us.