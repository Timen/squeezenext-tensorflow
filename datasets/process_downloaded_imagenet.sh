#!/usr/bin/env bash

# This file is a modified version of the script found at:
# https://github.com/tensorflow/models/blob/master/research/slim/datasets/download_and_convert_imagenet.sh
# The modifications include removal of imagenet data downloads and different paths used for extraction and
# output.


# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# usage:
#  ./process_downloaded_imagenet.sh [OUTDIR]
set -e


WORK_DIR=$PYTHONPATH
LABELS_FILE="${WORK_DIR}/datasets/imagenet_lsvrc_2015_synsets.txt"
OUTDIR="${1:-./}"

BBOX_DIR="${OUTDIR}bounding_boxes"
mkdir -p "${BBOX_DIR}"
cd "${OUTDIR}"

# See here for details: http://www.image-net.org/download-bboxes
BBOX_TAR_BALL="ILSVRC2012_bbox_train_v2.tar.gz"
echo "Uncompressing bounding box annotations ..."
tar xzf "${BBOX_TAR_BALL}" -C "${BBOX_DIR}"

LABELS_ANNOTATED="${BBOX_DIR}/*"
NUM_XML=$(ls -1 ${LABELS_ANNOTATED} | wc -l)
echo "Identified ${NUM_XML} bounding box annotations."

# Convert the XML files for bounding box annotations into a single CSV.
echo "Extracting bounding box information from XML."
BOUNDING_BOX_SCRIPT="${WORK_DIR}/datasets/process_bounding_boxes.py"
BOUNDING_BOX_FILE="${OUTDIR}/imagenet_2012_bounding_boxes.csv"
BOUNDING_BOX_DIR="${OUTDIR}bounding_boxes/"
"${BOUNDING_BOX_SCRIPT}" "${BOUNDING_BOX_DIR}" "${LABELS_FILE}" \
 | sort >"${BOUNDING_BOX_FILE}"
# Uncompress all images from the ImageNet 2012 validation dataset.
VALIDATION_TARBALL="ILSVRC2012_img_val.tar"
OUTPUT_PATH="${OUTDIR}validation/"
mkdir -p "${OUTPUT_PATH}"
tar xf "${VALIDATION_TARBALL}" -C "${OUTPUT_PATH}"

# Umcompress all images from the ImageNet 2012 train dataset.
TRAIN_TARBALL="ILSVRC2012_img_train.tar"
OUTPUT_PATH="${OUTDIR}train/"
mkdir -p "${OUTPUT_PATH}"
tar xf "${TRAIN_TARBALL}" -C "${OUTPUT_PATH}"

# Un-compress the individual tar-files within the train tar-file.
echo "Uncompressing individual train tar-balls in the training data."
#
while read SYNSET; do
  echo "Processing: ${SYNSET}"

  # Create a directory and delete anything there.
  mkdir -p "${OUTPUT_PATH}/${SYNSET}"
  rm -rf "${OUTPUT_PATH}/${SYNSET}/*"

  # Uncompress into the directory.
  tar xf "${TRAIN_TARBALL}" "${SYNSET}.tar"
  tar xf "${SYNSET}.tar" -C "${OUTPUT_PATH}/${SYNSET}/"
  rm -f "${SYNSET}.tar"

  echo "Finished processing: ${SYNSET}"
done < "${LABELS_FILE}"

# Note the locations of the train and validation data.
TRAIN_DIRECTORY="${OUTDIR}train/"
VALIDATION_DIRECTORY="${OUTDIR}validation/"


# Preprocess the validation data by moving the images into the appropriate
# sub-directory based on the label (synset) of the image.
echo "Organizing the validation data into sub-directories."
PREPROCESS_VAL_SCRIPT="${WORK_DIR}/datasets/preprocess_imagenet_validation_data.py"
VAL_LABELS_FILE="${WORK_DIR}/datasets/imagenet_2012_validation_synset_labels.txt"

"${PREPROCESS_VAL_SCRIPT}" "${VALIDATION_DIRECTORY}" "${VAL_LABELS_FILE}"

echo "Finished downloading and preprocessing the ImageNet data."

# Build the TFRecords version of the ImageNet data.
BUILD_SCRIPT="${WORK_DIR}/datasets/build_imagenet_data.py"
OUTPUT_DIRECTORY="${OUTDIR}/tf-records/"
IMAGENET_METADATA_FILE="${WORK_DIR}/datasets/imagenet_metadata.txt"

"${BUILD_SCRIPT}" \
  --train_directory="${TRAIN_DIRECTORY}" \
  --validation_directory="${VALIDATION_DIRECTORY}" \
  --output_directory="${OUTPUT_DIRECTORY}" \
  --imagenet_metadata_file="${IMAGENET_METADATA_FILE}" \
  --labels_file="${LABELS_FILE}" \
  --bounding_box_file="${BOUNDING_BOX_FILE}"