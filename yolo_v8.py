import cv2
import torch
import time
from ultralytics import YOLO  


model = YOLO('yolov8n.pt')  


SCALE_FACTOR = 0.01  
SPEED_THRESHOLD = 5.0 
PROLONGED_COLLISION_FRAMES = 30  
MIN_COLLISION_DISTANCE = 50  

def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def calculate_speed(box1, box2, time_diff):
    distance_px = ((box1[0] - box2[0]) ** 2 + (box1[1] - box2[1]) ** 2) ** 0.5
    speed_pxps = distance_px / time_diff if time_diff > 0 else 0  
    return speed_pxps

def detect_accident(video_path):
    cap = cv2.VideoCapture(video_path)
    accident_detected = False
    frame_counter = 0
    collision_threshold = 120  
    no_movement_counter = 0
    collision_count = 0
    last_boxes = []
    prolonged_collision_count = 0  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        results = model(frame)  
        boxes = results[0].boxes.xyxy.cpu().numpy()

        
        for i, box in enumerate(boxes):
            if len(box) >= 4:
                x1, y1, x2, y2 = box[:4]
                conf = box[4] if len(box) > 4 else None
                cls = int(box[5]) if len(box) > 5 else None

                label = f'Class {cls}: {conf:.2f}' if conf is not None else 'Unknown'
                color = (0, 255, 0)  
                if accident_detected:
                    color = (0, 0, 255)  

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                
                if len(last_boxes) > i:
                    speed = calculate_speed(last_boxes[i], box[:4], 1)  
                    cv2.putText(frame, f'Speed: {speed:.2f} px/s', (int(x1), int(y1) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                    
                    if speed < SPEED_THRESHOLD:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Sudden stop detected for object {i}!")

        collision_detected = False

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                box1 = boxes[i][:4]
                box2 = boxes[j][:4]
                
                iou = calculate_iou(box1, box2)
                if iou > 0.2:
                    collision_detected = True
                    collision_count += 1
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Collision detected between object {i} and object {j}!")

                    
                    if prolonged_collision_count < PROLONGED_COLLISION_FRAMES:
                        prolonged_collision_count += 1
                    if prolonged_collision_count >= PROLONGED_COLLISION_FRAMES:
                        accident_detected = True
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Accident Detected due to prolonged collision!")

                
                distance_between_boxes = ((box1[0] - box2[0]) ** 2 + (box1[1] - box2[1]) ** 2) ** 0.5
                if distance_between_boxes < MIN_COLLISION_DISTANCE:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Accident Detected due to close proximity between object {i} and object {j}!")

        if collision_detected:
            frame_counter += 1
            no_movement_counter = 0

            
            if frame_counter >= collision_threshold:
                accident_detected = True
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Accident Detected due to no movement for 2 minutes!")

        else:
            no_movement_counter += 1
            frame_counter = 0

        
            collision_percentage = len([1 for i in range(len(boxes)) for j in range(i + 1, len(boxes)) if calculate_iou(boxes[i][:4], boxes[j][:4]) > 0.2])
            if collision_percentage >= len(boxes) * 0.2:  # 20% of boxes are colliding
                accident_detected = True
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Accident Detected due to 20% collision of bounding boxes!")

        
        if accident_detected:
            cv2.putText(frame, "Accident Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        last_boxes = boxes.copy() 
        cv2.imshow('YOLOv8 Accident Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"Total collisions detected: {collision_count}")
    cap.release()
    cv2.destroyAllWindows()


video_path = 'Accident-1.mp4'
detect_accident(video_path)