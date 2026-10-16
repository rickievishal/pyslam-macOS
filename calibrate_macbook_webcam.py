#!/usr/bin/env python3
"""Capture checkerboard views from a webcam and update a pySLAM camera settings file."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def replace_setting(text: str, key: str, value: float | int) -> str:
    prefix = f"{key}:"
    lines = [f"{prefix} {value}" if line.startswith(prefix) else line for line in text.splitlines()]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a monocular webcam using a checkerboard.")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--cols", type=int, default=9, help="Inner checkerboard corners across")
    parser.add_argument("--rows", type=int, default=6, help="Inner checkerboard corners down")
    parser.add_argument("--square-size", type=float, default=1.0, help="Physical square size in any unit")
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--settings", type=Path, default=Path("settings/MACBOOK_M4_WEBCAM.yaml"))
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check Camera permission and --camera-id.")

    pattern = (args.cols, args.rows)
    object_template = np.zeros((args.rows * args.cols, 3), np.float32)
    object_template[:, :2] = np.mgrid[0 : args.cols, 0 : args.rows].T.reshape(-1, 2)
    object_template *= args.square_size
    object_points, image_points = [], []
    image_size = None

    print("Show the checkerboard at varied angles and distances. Press SPACE to capture, q to finish.")
    while len(image_points) < args.samples:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Webcam frame read failed during calibration.")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern)
        preview = frame.copy()
        cv2.drawChessboardCorners(preview, pattern, corners, found)
        cv2.putText(preview, f"samples {len(image_points)}/{args.samples}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("MacBook webcam calibration", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and found:
            refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            object_points.append(object_template)
            image_points.append(refined)
            image_size = (gray.shape[1], gray.shape[0])
            print(f"Captured {len(image_points)} samples")

    cap.release()
    cv2.destroyAllWindows()
    if len(image_points) < 10 or image_size is None:
        raise RuntimeError("Need at least 10 valid checkerboard captures; settings were not changed.")

    rms, matrix, distortion, _, _ = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
    values = {
        "Camera.fx": matrix[0, 0], "Camera.fy": matrix[1, 1], "Camera.cx": matrix[0, 2], "Camera.cy": matrix[1, 2],
        "Camera.k1": distortion.ravel()[0], "Camera.k2": distortion.ravel()[1], "Camera.p1": distortion.ravel()[2], "Camera.p2": distortion.ravel()[3],
        "Camera.k3": distortion.ravel()[4] if distortion.size > 4 else 0.0, "Camera.width": image_size[0], "Camera.height": image_size[1],
    }
    text = args.settings.read_text()
    for key, value in values.items():
        text = replace_setting(text, key, float(value) if isinstance(value, np.floating) else value)
    args.settings.write_text(text)
    print(f"Saved calibration to {args.settings}. RMS reprojection error: {rms:.4f} pixels")


if __name__ == "__main__":
    main()
