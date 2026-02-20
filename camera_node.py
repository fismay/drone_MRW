import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher_node')
        
        # Топик, в который будем публиковать видео (тот самый, который слушает YOLO)
        self.camera_topic = '/camera/image_raw'
        
        # Создаем публишер
        self.publisher_ = self.create_publisher(Image, self.camera_topic, 10)
        
        # Инициализируем CvBridge для конвертации изображений OpenCV <-> ROS 2
        self.bridge = CvBridge()
        
        # Подключаемся к камере. 
        # '0' означает стандартную веб-камеру. Если у дрона другая камера, здесь может быть URL потока (например, 'udp://...')
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            self.get_logger().error('Не удалось открыть камеру!')
            return
            
        self.get_logger().info('Камера успешно подключена. Начинаю трансляцию...')
        
        # Создаем таймер, который будет вызывать функцию захвата кадра
        # 0.033 секунды ~ 30 кадров в секунду (FPS)
        timer_period = 0.033 
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        # Читаем кадр с камеры
        ret, frame = self.cap.read()
        
        if ret:
            try:
                # Конвертируем кадр из формата OpenCV (numpy array) в формат ROS 2 (sensor_msgs/Image)
                # Кодировка 'bgr8' стандартна для цветных изображений OpenCV
                ros_image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                
                # Публикуем сообщение в топик
                self.publisher_.publish(ros_image_msg)
                
            except Exception as e:
                self.get_logger().error(f'Ошибка конвертации или публикации кадра: {e}')
        else:
            self.get_logger().warning('Пропущен кадр с камеры')

def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraPublisher()
    
    try:
        # Крутим ноду, пока не прервут (Ctrl+C)
        rclpy.spin(camera_node)
    except KeyboardInterrupt:
        camera_node.get_logger().info('Остановка узла камеры...')
    finally:
        # Обязательно освобождаем камеру при закрытии программы
        camera_node.cap.release()
        camera_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()