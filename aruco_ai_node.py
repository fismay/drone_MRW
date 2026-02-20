import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

import cv2
import cv2.aruco as aruco
import numpy as np
import math

class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector_node')

        # === ПАРАМЕТРЫ ===
        self.declare_parameter('camera_topic', '/camera/image_raw')
        camera_topic = self.get_parameter('camera_topic').value

        # === НАСТРОЙКИ ARUCO ===
        self.marker_size = 0.05  # 5 см
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        self.parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)

        self.camera_matrix = np.array([[600, 0, 320], 
                                       [0, 600, 240], 
                                       [0, 0, 1]], dtype=float)
        self.dist_coeffs = np.zeros((4,1)) 

        self.obj_points = np.array([
            [-self.marker_size / 2,  self.marker_size / 2, 0],
            [ self.marker_size / 2,  self.marker_size / 2, 0],
            [ self.marker_size / 2, -self.marker_size / 2, 0],
            [-self.marker_size / 2, -self.marker_size / 2, 0]
        ], dtype=np.float32)

        # === ROS 2 ИНФРАСТРУКТУРА ===
        self.bridge = CvBridge()
        
        # Подписчик на топик камеры
        self.subscription = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10)
        self.subscription  # keep reference

        # Паблишеры
        self.image_pub = self.create_publisher(Image, 'aruco/debug_image', 10)
        self.vector_pub = self.create_publisher(Point, 'aruco/vector_data', 10)

        self.get_logger().info(f"Нода Aruco Detector запущена. Подписка на топик: {camera_topic}")

    def image_callback(self, msg):
        # Конвертация ROS Image -> OpenCV
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Ошибка конвертации изображения: {e}")
            return

        h, w = frame.shape[:2]
        img_center_x = int(w / 2)
        img_center_y = int(h / 2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)

        # Рисуем прицел камеры
        cv2.drawMarker(frame, (img_center_x, img_center_y), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners)

            for i in range(len(ids)):
                success, rvec, tvec = cv2.solvePnP(self.obj_points, corners[i][0], self.camera_matrix, self.dist_coeffs)

                if success:
                    x, y, z = tvec[0][0], tvec[1][0], tvec[2][0]
                    distance_3d = math.sqrt(x**2 + y**2 + z**2)

                    c = corners[i][0]
                    marker_center_x = int((c[0][0] + c[1][0] + c[2][0] + c[3][0]) / 4)
                    marker_center_y = int((c[0][1] + c[1][1] + c[2][1] + c[3][1]) / 4)

                    # Вычисление вектора направления
                    dx = marker_center_x - img_center_x
                    dy = marker_center_y - img_center_y
                    offset_pixels = math.sqrt(dx**2 + dy**2)

                    # === ЛОГИРОВАНИЕ ROS 2 ===
                    self.get_logger().info(
                        f"ID: {ids[i][0]} | Dist 3D: {distance_3d:.2f}m | Vector: dx={dx}, dy={dy} | Offset={offset_pixels:.1f}px"
                    )

                    # === ПУБЛИКАЦИЯ В ТОПИК ===
                    msg_point = Point()
                    msg_point.x = float(dx)
                    msg_point.y = float(dy)
                    msg_point.z = float(distance_3d)
                    self.vector_pub.publish(msg_point)

                    # Визуализация
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.03)
                    cv2.arrowedLine(frame, (img_center_x, img_center_y), (marker_center_x, marker_center_y), (0, 0, 255), 2, tipLength=0.1)
                    
                    # Вывод текста
                    cv2.putText(frame, f"3D Dist: {distance_3d:.2f}m", (marker_center_x, marker_center_y - 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, f"Vector: dx={dx}, dy={dy}", (marker_center_x, marker_center_y - 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                    cv2.putText(frame, f"ID: {ids[i][0]}", (marker_center_x, marker_center_y - 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Публикуем обработанное изображение
        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.image_pub.publish(img_msg)

        # Локальное отображение (опционально)
        cv2.imshow('Aruco ROS2 Node', frame)
        cv2.waitKey(1)

    def destroy_node(self):
        # Освобождаем ресурсы OpenCV
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Остановка ноды пользователем...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
