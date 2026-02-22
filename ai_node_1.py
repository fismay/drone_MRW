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
        
        self.camera_topic = '/ov5647/image_raw'
        self.get_logger().info(f"Ожидаю данные с камеры в топике: {self.camera_topic}")

        self.output_image_topic = '/drone/detection_image'
        self.output_data_topic = '/drone/detection_data'
        
        self.subscription = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            10)
        
        self.image_publisher = self.create_publisher(Image, self.output_image_topic, 10)
        self.data_publisher = self.create_publisher(String, self.output_data_topic, 10)
        
        self.bridge = CvBridge()
        
        self.get_logger().info("Загружаю модель YOLO...")
        self.model = YOLO("yolov8n.pt")
        
        self.target_classes = ['orange', 'teddy bear']
        
        # Углы поворота для TTA (в градусах)
        self.angles = [0, 90, 180, 270]

        # --- Директория для сохранения изображений с детекциями ---
        self.save_dir = "detections"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            self.get_logger().info(f"Создана директория для сохранения: {self.save_dir}")

    def rotate_image(self, image, angle):
        """Поворачивает изображение на заданный угол и возвращает повёрнутое изображение и матрицу преобразования."""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Корректируем размер, чтобы изображение влезало полностью
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        rotated = cv2.warpAffine(image, M, (new_w, new_h))
        return rotated, M, (w, h)

    def transform_boxes_back(self, boxes, M, original_shape, rotated_shape):
        """Преобразует координаты bounding boxes из повёрнутого изображения обратно в исходное."""
        # Матрица для обратного преобразования
        M_inv = cv2.invertAffineTransform(M)
        h, w = original_shape
        rh, rw = rotated_shape
        
        transformed = []
        for box in boxes:
            x1, y1, x2, y2 = box
            # Угловые точки прямоугольника
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32).reshape(-1, 1, 2)
            # Применяем обратное аффинное преобразование
            corners_back = cv2.transform(corners, M_inv).reshape(-1, 2)
            # Находим минимальный ограничивающий прямоугольник
            x1_back = np.clip(np.min(corners_back[:, 0]), 0, w)
            y1_back = np.clip(np.min(corners_back[:, 1]), 0, h)
            x2_back = np.clip(np.max(corners_back[:, 0]), 0, w)
            y2_back = np.clip(np.max(corners_back[:, 1]), 0, h)
            transformed.append([x1_back, y1_back, x2_back, y2_back])
        return np.array(transformed)

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

            # Детекция на повёрнутом изображении
            results = self.model(rotated, verbose=False)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]
                    if class_name in self.target_classes:
                        x1, y1, x2, y2 = map(float, box.xyxy[0])  # float для точности
                        conf = float(box.conf[0])
                        
                        if angle == 0:
                            # Для исходного угла просто сохраняем
                            all_boxes.append([x1, y1, x2, y2])
                            all_confs.append(conf)
                            all_cls.append(cls_id)
                            all_detections.append({
                                "class": class_name,
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
                                "class": class_name,
                                "confidence": conf,
                                "bbox": [int(x1b), int(y1b), int(x2b), int(y2b)],
                                "center": [int((x1b+x2b)//2), int((y1b+y2b)//2)],
                                "angle": angle
                            })

        # Применяем NMS для удаления дубликатов
        if all_boxes:
            all_boxes = np.array(all_boxes)
            all_confs = np.array(all_confs)
            all_cls = np.array(all_cls)
            
            # Используем самодельный NMS
            # Можно выполнять NMS отдельно для каждого класса, но для простоты делаем общий NMS по всем классам
            keep_indices = nms_boxes(all_boxes, all_confs, iou_threshold=0.5)
        else:
            keep_indices = []

        # Формируем финальный список детекций после NMS и рисуем на исходном изображении
        final_detections = []
        for idx in keep_indices:
            det = all_detections[idx]
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            class_name = det["class"]
            center_x, center_y = det["center"]
            
            final_detections.append(det)
            
            # Рисуем на изображении
            cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            label = f"{class_name} {conf:.2f}"
            cv2.circle(original_image, (center_x, center_y), 5, (0, 255, 0), -1)
            cv2.putText(original_image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            self.get_logger().info(f"ОБНАРУЖЕН: {class_name} | Точность: {conf:.2f}")

        # --- Сохранение изображения с детекциями в файл ---
        # Используем время из заголовка сообщения для уникального имени
        timestamp_sec = msg.header.stamp.sec
        timestamp_nsec = msg.header.stamp.nanosec
        # Если время нулевое, используем текущее системное время
        if timestamp_sec == 0 and timestamp_nsec == 0:
            now = datetime.now()
            filename = now.strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
        else:
            filename = f"{timestamp_sec}_{timestamp_nsec:09d}.jpg"
        filepath = os.path.join(self.save_dir, filename)
        cv2.imwrite(filepath, original_image)
        self.get_logger().debug(f"Сохранено изображение: {filepath}")

        # Публикация данных в JSON
        if final_detections:
            data_msg = String()
            data_msg.data = json.dumps(final_detections)
            self.data_publisher.publish(data_msg)

        # Публикация изображения с нарисованными рамками
        out_msg = self.bridge.cv2_to_imgmsg(original_image, "bgr8")
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
