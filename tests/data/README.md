# Test data

- `golfer.jpg` — 640×480 crop of "U.S. Army soldier swings a golf club after
  duty" by Staff Sgt. Vernon Young Jr., U.S. Air Force, a work of the U.S.
  federal government in the public domain (Wikimedia Commons, file
  `MPOTY_2014_U.S._Army_soldier_swings_a_golf_club_after_duty.jpg`).
- `golfer.rtmlib.json` — reference output of rtmlib (onnxruntime) for that
  image, used by `tests/pose_models.rs`:
  - `detections`: rtmlib's `YOLOX` decoder for raw outputs with the
    `yolox-tiny` model (person boxes, score threshold 0.3, NMS IoU 0.45), on
    the image as BGR.
  - `poses`: rtmlib's `RTMPose` with `rtmpose-s-halpe26` and
    `rtmpose-m-halpe26` for three boxes per frame (the detection, a looser
    box and one running off the image), on the image as RGB.
  - Frame `flipped` is the image mirrored left to right.

  Both sides decode the JPEG with zune-jpeg, so the comparison sees the same
  pixels.
