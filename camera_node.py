import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher_node')
        
        self.camera_topic = '/camera/image_raw'
        self.publisher_ = self.create_publisher(Image, self.camera_topic, 10)
        self.bridge = CvBridge()
        
        # Строка конфигурации GStreamer для libcamera (стандарт для Raspberry Pi 5)
        # Здесь задается разрешение 640x480 и 30 FPS. Можно поменять под нужды YOLO.
        gstreamer_pipeline = (
            "libcamerasrc ! "
            "video/x-raw, width=640, height=480, framerate=30/1 ! "
            "videoconvert ! appsink"
        )
        
        self.get_logger().info('Попытка подключения к CSI-камере через libcamera...')
        
        # Подключаемся с использованием GStreamer
        self.cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
        
        # Запасной вариант (Fallback), если OpenCV собран без GStreamer
        if not self.cap.isOpened():
            self.get_logger().warning('GStreamer недоступен, пробую стандартный V4L2 (/dev/video0)...')
            # Обращаемся напрямую к устройству и указываем бэкенд V4L2
            self.cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
            
        if not self.cap.isOpened():
            self.get_logger().error('Не удалось открыть CSI-камеру! Проверьте шлейф и настройки.')
            return
            
        self.get_logger().info('Камера успешно подключена. Начинаю трансляцию...')
        
        timer_period = 0.033 
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        
        if ret:
            try:
                ros_image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                self.publisher_.publish(ros_image_msg)
            except Exception as e:
                self.get_logger().error(f'Ошибка конвертации или публикации кадра: {e}')
        else:
            self.get_logger().warning('Пропущен кадр с камеры')

def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraPublisher()
    
    try:
        rclpy.spin(camera_node)
    except KeyboardInterrupt:
        camera_node.get_logger().info('Остановка узла камеры...')
    finally:
        # Проверяем, существует ли объект cap, прежде чем его закрывать
        if hasattr(camera_node, 'cap') and camera_node.cap.isOpened():
            camera_node.cap.release()
        camera_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
