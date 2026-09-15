from ultralytics import YOLO
import cv2
import os
import time

# Load YOLO model
model = YOLO("yolo11n.pt")

# Input video
input_video =  r"C:\Users\ayush\OneDrive\Desktop\test.mp4"

# Create output folder safely
os.makedirs("output", exist_ok=True)

# Output video
output_video = "output/detected_tracking.mp4"

# Open video
cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("❌ Unable to open video")
    exit()

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_video = cap.get(cv2.CAP_PROP_FPS)

if fps_video <= 0:
    fps_video = 30

# Video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps_video,
    (width, height)
)

prev_time = 0

print("🚀 Processing video...")
print("Press Q to stop.")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # YOLO detection + tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    # Draw boxes and tracking IDs
    annotated_frame = results[0].plot()

    # Count objects
    total_objects = 0
    person_count = 0

    if results[0].boxes is not None:

        total_objects = len(results[0].boxes)

        for box in results[0].boxes:

            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if class_name == "person":
                person_count += 1

    # Calculate FPS
    current_time = time.time()

    if prev_time != 0:
        fps = 1 / (current_time - prev_time)
    else:
        fps = 0

    prev_time = current_time

    # Information panel
    cv2.rectangle(
        annotated_frame,
        (10, 10),
        (430, 110),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        annotated_frame,
        "AI OBJECT DETECTION & TRACKING",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Objects: {total_objects}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Persons: {person_count}",
        (180, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # Show video
    cv2.imshow(
        "AI Object Detection & Tracking",
        annotated_frame
    )

    # Save processed frame
    out.write(annotated_frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("\n✅ Video processing completed!")
print(f"📁 Output saved at: {output_video}")