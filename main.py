"""Thử nhận diện biển số bằng fast-alpr từ camera hoặc từ file ảnh.

Cài đặt (cần Python 3.10 trở lên, nên dùng venv riêng):
    pip install "fast-alpr[onnx]"

Cách chạy:
    python test_fast_alpr.py              # mở camera, bấm SPACE để chụp, Q để thoát
    python test_fast_alpr.py a.jpg b.jpg  # thử trên ảnh có sẵn

Ảnh gốc và ảnh đã vẽ khung được lưu trong thư mục thu_fast_alpr/.
"""
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "thu_fast_alpr"
DETECTOR_MODEL = "yolo-v9-t-384-license-plate-end2end"
OCR_MODEL = "cct-xs-v2-global-model"
CAMERA_INDEX = 0
WARMUP_FRAMES = 15      # bỏ vài khung đầu vì camera chưa lấy sáng xong
FALLBACK_SHOTS = 3      # số ảnh tự chụp khi không xem trước được
FALLBACK_DELAY = 5      # giây giữa các lần tự chụp


def stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def use_certifi_certs():
    """Python cài từ python.org trên macOS thường thiếu chứng chỉ gốc, nên tải mô hình báo
    CERTIFICATE_VERIFY_FAILED. Dùng bộ chứng chỉ của certifi nếu có."""
    if os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import certifi
    except ImportError:
        return
    os.environ["SSL_CERT_FILE"] = certifi.where()


def load_alpr():
    use_certifi_certs()
    try:
        from fast_alpr import ALPR
    except ImportError:
        sys.exit('Chưa cài fast-alpr. Dùng Python 3.10+ rồi chạy: pip install "fast-alpr[onnx]"')
    print("Đang tải mô hình (lần đầu có thể mất vài phút)...")
    try:
        return ALPR(detector_model=DETECTOR_MODEL, ocr_model=OCR_MODEL)
    except Exception as exc:
        if "CERTIFICATE_VERIFY_FAILED" in str(exc):
            sys.exit(
                "Không tải được mô hình vì thiếu chứng chỉ SSL (thường gặp với Python cài từ python.org trên macOS).\n"
                "Cách 1: chạy file 'Install Certificates.command' trong thư mục /Applications/Python 3.x/.\n"
                "Cách 2: pip install --upgrade certifi rồi chạy lại script này."
            )
        raise


def describe(result):
    """In kết quả; thuộc tính được đọc an toàn vì tên có thể khác giữa các phiên bản."""
    ocr = getattr(result, "ocr", None)
    text = getattr(ocr, "text", None)
    conf = getattr(ocr, "confidence", None)
    if text is not None:
        if isinstance(conf, (list, tuple)):
            conf = sum(conf) / len(conf) if conf else None
        conf_text = f" (tin cậy {conf:.2f})" if isinstance(conf, (int, float)) else ""
        print(f"  Biển số: {text}{conf_text}")
    print(f"  Chi tiết: {result}")


def draw(alpr, frame, results):
    """Trả về ảnh đã vẽ khung; ưu tiên hàm vẽ của thư viện, không được thì tự vẽ."""
    try:
        out = alpr.draw_predictions(frame.copy())
        image = getattr(out, "image", out)
        if isinstance(image, np.ndarray):
            return image
    except Exception:
        pass

    image = frame.copy()
    for r in results:
        box = getattr(getattr(r, "detection", None), "bounding_box", None)
        text = getattr(getattr(r, "ocr", None), "text", "") or ""
        try:
            x1, y1, x2, y2 = (int(getattr(box, k)) for k in ("x1", "y1", "x2", "y2"))
        except Exception:
            continue
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, text, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return image


def process(alpr, frame, name):
    """Chạy fast-alpr trên một ảnh, in kết quả và lưu ảnh gốc + ảnh có khung."""
    OUT_DIR.mkdir(exist_ok=True)
    started = time.time()
    results = alpr.predict(frame)
    elapsed = time.time() - started
    print(f"[{name}] {len(results)} biển số, {elapsed:.2f}s")
    for r in results:
        describe(r)
    if not results:
        print("  Không phát hiện biển số nào.")
    cv2.imwrite(str(OUT_DIR / f"{name}_goc.jpg"), frame)
    cv2.imwrite(str(OUT_DIR / f"{name}_ket_qua.jpg"), draw(alpr, frame, results))
    print(f"  Đã lưu ảnh vào {OUT_DIR}/{name}_*.jpg")


def run_images(alpr, paths):
    for p in paths:
        frame = cv2.imread(p)
        if frame is None:
            print(f"Không đọc được ảnh: {p}")
            continue
        process(alpr, frame, Path(p).stem)


def run_camera(alpr):
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        sys.exit(
            "Không mở được camera. Trên macOS, vào System Settings > Privacy & Security > Camera "
            "và cho phép Terminal hoặc PyCharm, rồi chạy lại."
        )
    for _ in range(WARMUP_FRAMES):
        cap.read()

    preview = True
    shots = 0
    last_shot = time.time()
    print("Camera đã mở. Bấm SPACE để chụp và nhận diện, Q để thoát.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Không đọc được hình ảnh từ camera.")
                break

            if preview:
                try:
                    cv2.imshow("Thu fast-alpr - SPACE: chup, Q: thoat", frame)
                    key = cv2.waitKey(1) & 0xFF
                except (cv2.error, AttributeError):
                    print(
                        "OpenCV của bạn không có cửa sổ xem trước (bản headless). "
                        f"Chuyển sang tự chụp {FALLBACK_SHOTS} ảnh, mỗi {FALLBACK_DELAY} giây; "
                        "hãy đưa biển số vào trước camera."
                    )
                    preview = False
                    last_shot = time.time()
                    continue
                if key == ord("q"):
                    break
                if key == ord(" "):
                    process(alpr, frame, stamp())
            elif time.time() - last_shot >= FALLBACK_DELAY:
                process(alpr, frame, stamp())
                shots += 1
                last_shot = time.time()
                if shots >= FALLBACK_SHOTS:
                    break
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except (cv2.error, AttributeError):
            pass


def main():
    alpr = load_alpr()
    if len(sys.argv) > 1:
        run_images(alpr, sys.argv[1:])
    else:
        run_camera(alpr)


if __name__ == "__main__":
    main()