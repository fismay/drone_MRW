import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import json
import numpy as np
from ultralytics import YOLO
import os  # для работы с директориями
from datetime import datetime  # для формирования имени файла

def nms_boxes(boxes, scores, iou_threshold):
    """
    Perform Non-Maximum Suppression (NMS) on bounding boxes.
    boxes: numpy array of shape (N, 4) where each box is [x1, y1, x2, y2]
    scores: numpy array of shape (N,)
    iou_threshold: float
    Returns indices of boxes to keep.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # descending

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

class DroneDetector(Node):
    def __init__(self):
        super().__init__('drone_detector_node')
        
        self.camera_topic = '/camera/image_raw'
        self.get_logger().info(f"Ожидаю данные с камеры в топике: {self.camera_topic}")
        self.output_image_topic = '/drone/detection_image'
        self.output_data_topic = '/drone/detection_data'
        
        self.subscription = self.create_subscription(Image, self.camera_topic, self.image_callback, 10)
        self.image_publisher = self.create_publisher(Image, self.output_image_topic, 10)
        self.data_publisher = self.create_publisher(String, self.output_data_topic, 10)
        
        self.bridge = CvBridge()
        
        self.get_logger().info("Загружаю модель YOLO...")
        self.model = YOLO("yolov8n.pt")
     
        self.target_classes = ['orange', 'teddy bear']
        self.display_names = {'orange': 'orange', 'teddy bear': 'cheburashka'}
        self.angles = [0, 90, 180, 270]

        self.save_dir = "detections"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            self.get_logger().info(f"Создана директория для сохранения: {self.save_dir}")
        
        self.saved_count = 0
        self.get_logger().info("node succesfully started")

    def rotate_image(self, image, angle):
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        rotated = cv2.warpAffine(image, M, (new_w, new_h))
        return rotated, M, (w, h)

    def transform_boxes_back(self, boxes, M, original_shape, rotated_shape):
        M_inv = cv2.invertAffineTransform(M)
        h, w = original_shape
        rh, rw = rotated_shape
        transformed = []
        for box in boxes:
            x1, y1, x2, y2 = box
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32).reshape(-1, 1, 2)
            corners_back = cv2.transform(corners, M_inv).reshape(-1, 2)
            x1_back = np.clip(np.min(corners_back[:, 0]), 0, w)
            y1_back = np.clip(np.min(corners_back[:, 1]), 0, h)
            x2_back = np.clip(np.max(corners_back[:, 0]), 0, w)
            y2_back = np.clip(np.max(corners_back[:, 1]), 0, h)
            transformed.append([x1_back, y1_back, x2_back, y2_back])
        return np.array(transformed)

    def save_detection_image(self, image, filename_prefix=None):
        if filename_prefix:
            filename = f"{filename_prefix}.jpg"
        else:
            self.saved_count += 1
            filename = f"{self.saved_count:06d}.jpg"
        filepath = os.path.join(self.save_dir, filename)
        cv2.imwrite(filepath, image)
        return filepath

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Ошибка конвертации: {e}')
            return

        original_image = cv_image.copy()
        all_detections = []   # список для сбора детекций со всех поворотов
        all_boxes = []        # для NMS
        all_confs = []
        all_cls = []

        for angle in self.angles:
            if angle == 0:
                rotated = original_image
                M = None
                rot_shape = original_image.shape[:2]
            else:
                rotated, M, orig_shape = self.rotate_image(original_image, angle)
                rot_shape = rotated.shape[:2]

            results = self.model(rotated, verbose=False)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]
                    if class_name in self.target_classes:
                        x1, y1, x2, y2 = map(float, box.xyxy[0])  # float для точности
                        conf = float(box.conf[0])
                        
                        display_name = self.display_names.get(class_name, class_name)
                        
                        if angle == 0:
                            # Для исходного угла просто сохраняем
                            all_boxes.append([x1, y1, x2, y2])
                            all_confs.append(conf)
                            all_cls.append(cls_id)
                            all_detections.append({
                                "class": display_name,  # Используем отображаемое имя
                                "original_class": class_name,  # Сохраняем оригинальное имя для логирования
                                "confidence": conf,
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                "center": [int((x1+x2)//2), int((y1+y2)//2)],
                                "angle": angle
                            })
                        else:
                            # Преобразуем координаты обратно
                            boxes_back = self.transform_boxes_back(
                                np.array([[x1, y1, x2, y2]]), M, original_image.shape[:2], rot_shape)
                            x1b, y1b, x2b, y2b = boxes_back[0]
                            all_boxes.append([x1b, y1b, x2b, y2b])
                            all_confs.append(conf)
                            all_cls.append(cls_id)
                            all_detections.append({
                                "class": display_name,  # Используем отображаемое имя
                                "original_class": class_name,  # Сохраняем оригинальное имя для логирования
                                "confidence": conf,
                                "bbox": [int(x1b), int(y1b), int(x2b), int(y2b)],
                                "center": [int((x1b+x2b)//2), int((y1b+y2b)//2)],
                                "angle": angle
                            })

        if all_boxes:
            all_boxes = np.array(all_boxes)
            all_confs = np.array(all_confs)
            all_cls = np.array(all_cls)
            keep_indices = nms_boxes(all_boxes, all_confs, iou_threshold=0.5)
        else:
            keep_indices = []
            
        if not keep_indices:
            out_msg = self.bridge.cv2_to_imgmsg(original_image, "bgr8")
            self.image_publisher.publish(out_msg)
            return
        final_detections = []
        detected_classes = set()  # Множество для отслеживания обнаруженных классов
        
        # Сначала собираем все детекции
        for idx in keep_indices:
            det = all_detections[idx]
            final_detections.append(det)
            detected_classes.add(det["class"])  # Добавляем класс в множество
        
        # Создаем копию изображения для отрисовки
        annotated_image = original_image.copy()
        
        # Определяем общее сообщение для изображения
        combined_message = None
        if 'cheburashka' in detected_classes and 'orange' in detected_classes:
            combined_message = "cheburashka with orange"
            self.get_logger().info(f"ОБНАРУЖЕНЫ ОБА ОБЪЕКТА: {combined_message}")
        
        # Рисуем на изображении
        for det in final_detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            class_name = det["class"]  # Это уже отображаемое имя
            original_class = det.get("original_class", class_name)  # Для логирования
            center_x, center_y = det["center"]
            
            # Рисуем рамки для каждого объекта
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.circle(annotated_image, (center_x, center_y), 5, (0, 255, 0), -1)
            
            # Если есть комбинированное сообщение, используем его вместо отдельных подписей
            if combined_message and len(final_detections) == 2:
                # Рисуем одно общее сообщение вверху изображения
                cv2.putText(annotated_image, combined_message, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                # Не рисуем отдельные подписи у каждого объекта
            else:
                # Рисуем отдельные подписи
                label = f"{class_name} {conf:.2f}"
                cv2.putText(annotated_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            

            self.get_logger().info(f"ОБНАРУЖЕН: {class_name} | Точность: {conf:.2f}")


        timestamp_sec = msg.header.stamp.sec
        timestamp_nsec = msg.header.stamp.nanosec
        
        filename_prefix = ''
        if combined_message:
            filename_prefix += f"both_"
        elif len(final_detections) == 1:
            filename_prefix += f"{final_detections[0]['class']}_"
        else:
            filename_prefix += f"{len(final_detections)}_objects_-"
        
        
        if timestamp_sec == 0 and timestamp_nsec == 0:
            now = datetime.now()
            filename_prefix += now.strftime("%Y%m%d_%H%M%S_%f")
        else:
            filename_prefix += f"{timestamp_sec}_{timestamp_nsec:09d}"
        self.save_detection_image(annotated_image, filename_prefix)

        # Публикация данных в JSON
        if final_detections:
            # Создаем данные для публикации
            publish_detections = []
            
            # Если обнаружены оба объекта, добавляем специальное сообщение
            if 'cheburashka' in detected_classes and 'orange' in detected_classes:
                combined_data = {
                    "combined_message": "cheburashka with orange",
                    "detections": []
                }
                for det in final_detections:
                    publish_det = {k: v for k, v in det.items() if k != 'original_class'}
                    combined_data["detections"].append(publish_det)
                data_msg = String()
                data_msg.data = json.dumps(combined_data)
            else:
                # Обычная публикация отдельных детекций
                for det in final_detections:
                    publish_det = {k: v for k, v in det.items() if k != 'original_class'}
                    publish_detections.append(publish_det)
                data_msg = String()
                data_msg.data = json.dumps(publish_detections)
            
            self.data_publisher.publish(data_msg)

        # Публикация изображения с нарисованными рамками
        out_msg = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
        self.image_publisher.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DroneDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
