import numpy as np
from aslcv.landmarks import to_pixel_space
from aslcv.features import normalize_landmarks


def test_pixel_space_scales_x_by_width_y_by_height():
    lm = np.array([[0.5, 0.5, 0.1]] * 21)
    px = to_pixel_space(lm, width=640, height=480)
    assert np.allclose(px[0], [320, 240, 64])  # z uses width scale, as MediaPipe documents


def test_same_hand_gives_same_features_on_4x3_and_square_images():
    rng = np.random.default_rng(1)
    hand_px = rng.random((21, 3)) * 150 + 20          # a hand measured in pixels
    on_webcam = hand_px / [640, 480, 640]             # what MediaPipe reports on 640x480
    on_square = hand_px / [200, 200, 200]             # ...and on a 200x200 image
    a = normalize_landmarks(to_pixel_space(on_webcam, 640, 480))
    b = normalize_landmarks(to_pixel_space(on_square, 200, 200))
    assert np.allclose(a, b)


def test_mirror_for_display_flips_x_only():
    from aslcv.landmarks import Hand, mirror_hand
    lm = np.array([[0.2, 0.3, 0.1]] * 21)
    h = Hand(lm, lm * [640, 480, 640], "Right", 0.9)
    m = mirror_hand(h)
    assert np.allclose(m.landmarks[0], [0.8, 0.3, 0.1])
    assert m.handedness == "Right"           # label is untouched: it describes the real hand


def test_mediapipe_label_follows_the_unmirrored_image():
    """Convention check (journal 07): MediaPipe's label describes the hand as it appears in the
    image you give it. Flipping the image flips the label, so detection must run on the camera's
    real, unmirrored image for 'Right' to mean the signer's right hand."""
    import glob, cv2, pytest
    from aslcv.landmarks import HandDetector, MODEL_PATH
    files = sorted(glob.glob("data/raw/asl_alphabet_train/asl_alphabet_train/Y/*.jpg"))[100:140]
    if not files or not MODEL_PATH.exists():
        pytest.skip("needs the public dataset and the MediaPipe model")
    det = HandDetector(mode="image")
    pairs = []
    for f in files:
        img = cv2.imread(f)
        a, b = det.detect(img), det.detect(cv2.flip(img, 1))
        if a and b:
            pairs.append((a[0].handedness, b[0].handedness))
    det.close()
    assert len(pairs) >= 10
    assert sum(x != y for x, y in pairs) / len(pairs) > 0.9
