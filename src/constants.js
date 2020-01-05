export const ML_STATUSES = {
  INITIAL: "INITIAL",
  DOWNLOADING_MODELS: "DOWNLOADING_MODELS",
  WARMING_UP: "WARMING_UP",
  READY_FOR_DETECTION: "READY_FOR_DETECTION",
  DETECTING: "DETECTING",
  CLASSIFYING: "CLASSIFYING",
  COMPLETE: "COMPLETE",
}

export const DETECTION_MODEL_URL =
  "https://jk-fish-test.s3.us-east-2.amazonaws.com/fish_mobilenet2/model.json"

// export const CLASSIFICATION_MODEL_URL =
//   "https://jk-fish-test.s3.us-east-2.amazonaws.com/test_fish_classifier/model.json"
export const CLASSIFICATION_MODEL_URL =
  "https://jk-fish-test.s3.us-east-2.amazonaws.com/rockfish-12/model.json"
