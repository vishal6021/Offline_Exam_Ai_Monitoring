import cv2
import os
import csv
from ultralytics import YOLO

model = YOLO("best.pt")

image_path = ""
output_crops_dir = "cheaters"
os.makedirs(output_crops_dir, exist_ok=True)

def process_image(image_file, show_window=True):

    frame = cv2.imread(image_file)
    if frame is None:
        print("Image not found:", image_file)
        return

    results = model(frame, conf=0.2, iou=0.1)

    student_id_counter = 1
    student_data = []

    for result in results:
        boxes_sorted = []

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = model.names[int(box.cls[0])].lower()

            boxes_sorted.append({
                "coords": (x1, y1, x2, y2),
                "conf": conf,
                "label": label,
                "center_y": (y1 + y2) // 2,
                "center_x": (x1 + x2) // 2
            })

        boxes_sorted.sort(key=lambda b: (b["center_y"] // 100, b["center_x"]))

        for b in boxes_sorted:
            x1, y1, x2, y2 = b["coords"]
            conf = b["conf"]
            label = b["label"]

            student_id = f"STD{student_id_counter:03d}"
            student_id_counter += 1

            if "tidak" in label:          # Not cheating
                center = ((x1 + x2)//2, (y1 + y2)//2)
                cv2.circle(frame, center, 6, (0,255,0), -1)
                status = "Not Cheating"

            else:                          # Cheating
                crop = frame[y1:y2, x1:x2]
                if crop.size != 0:
                    cv2.imwrite(os.path.join(output_crops_dir, f"{student_id}.jpg"), crop)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(frame, f"{student_id} {conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                status = "Cheating"

            student_data.append([student_id, status, round(conf,3), f"{x1},{y1},{x2},{y2}"])

    output_image = "output_" + os.path.basename(image_file)
    cv2.imwrite(output_image, frame)

    if show_window:
        cv2.imshow("Result", cv2.resize(frame, (900,600)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    csv_name = "log_" + os.path.splitext(os.path.basename(image_file))[0] + ".csv"
    with open(csv_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID","Status","Conf","Box"])
        writer.writerows(student_data)

    print("Done:", output_image, "| Log saved:", csv_name)


if os.path.isdir(image_path):
    for file in os.listdir(image_path):
        if file.lower().endswith((".jpg",".jpeg",".png")):
            process_image(os.path.join(image_path, file))
else:
    process_image(image_path)
