import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String  # Добавлен импорт для текстовых сообщений
from cv_bridge import CvBridge
import cv2
import json  # Добавлен импорт для работы с JSON
from ultralytics import YOLO

class DroneDetector(Node):
    def __init__(self):
        super().__init__('drone_detector_node')
        
        # 1. Объявляем параметр для входного топика камеры
        # Это позволит менять топик при запуске, не меняя сам код
        self.declare_parameter('input_camera_topic', '/camera/image_raw')
        self.camera_topic = self.get_parameter('input_camera_topic').get_parameter_value().string_value
        
        self.get_logger().info(f"Ожидаю данные с камеры в топике: {self.camera_topic}")

        # Настройки выходных топиков
        self.output_image_topic = '/drone/detection_image'
        self.output_data_topic = '/drone/detection_data' 
        
        # Подписка на топик камеры
        self.subscription = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            10)
        
        # Публишер для картинки
        self.image_publisher = self.create_publisher(Image, self.output_image_topic, 10)
        # Публишер для данных (координаты, классы)
        self.data_publisher = self.create_publisher(String, self.output_data_topic, 10)
        
        self.bridge = CvBridge()
        
        self.get_logger().info("Загружаю модель YOLO...")
        self.model = YOLO("yolov8n.pt")
        
        # Ищем плюшевых мишек и апельсины
        self.target_classes = ['orange', 'teddy bear']

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        except Exception as e:
            self.get_logger().error(f'Ошибка конвертации: {e}')
            return

        results = self.model(cv_image, verbose=False)
        
        # Список для сбора данных обо всех найденных объектах в текущем кадре
        detections_list = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                
                if class_name in self.target_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])  # Переводим в float для JSON
                    
                    # Логирование
                    self.get_logger().info(f"ОБНАРУЖЕН: {class_name} | Точность: {conf:.2f}")
                    
                    # Вычисляем центр объекта (полезно для управления дроном)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Добавляем данные объекта в список
                    detections_list.append({
                        "class": class_name,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                        "center": [center_x, center_y]
                    })
                    
                    # Рисуем рамку и текст
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    label = f"{class_name} {conf:.2f}"
                    # Рисуем точку в центре объекта
                    cv2.circle(cv_image, (center_x, center_y), 5, (0, 255, 0), -1)
                    cv2.putText(cv_image, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Публикация текстовых данных (если объекты найдены)
        if detections_list:
            data_msg = String()
            data_msg.data = json.dumps(detections_list) # Конвертируем список словарей в JSON-строку
            self.data_publisher.publish(data_msg)

        # Публикация картинки
        out_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
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
