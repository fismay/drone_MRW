import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import math

class FlightController(Node):
    def __init__(self):
        super().__init__('flight_controller')
        
        # Параметры (можно переопределить при запуске)
        self.declare_parameter('takeoff_height', 0.5)
        self.declare_parameter('forward_1', 1.5)
        self.declare_parameter('right_1', 0.8)
        self.declare_parameter('backward', 1.5)
        self.declare_parameter('right_2', 0.8)
        self.declare_parameter('forward_2', 1.5)
        self.declare_parameter('tolerance', 0.1)          # допуск достижения цели (м)
        self.declare_parameter('control_rate', 20.0)      # частота публикации setpoint (Гц)
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('setpoint_topic', '/setpoint_position/local')
        self.declare_parameter('landing_timeout', 5.0)    # время удержания посадки перед выходом (с)
        
        # Чтение параметров
        takeoff_height = self.get_parameter('takeoff_height').value
        forward_1 = self.get_parameter('forward_1').value
        right_1 = self.get_parameter('right_1').value
        backward = self.get_parameter('backward').value
        right_2 = self.get_parameter('right_2').value
        forward_2 = self.get_parameter('forward_2').value
        self.tolerance = self.get_parameter('tolerance').value
        control_rate = self.get_parameter('control_rate').value
        odom_topic = self.get_parameter('odom_topic').value
        setpoint_topic = self.get_parameter('setpoint_topic').value
        self.landing_timeout = self.get_parameter('landing_timeout').value
        
        # Список целей (x, y, z) в локальной системе координат (старт в (0,0,0))
        self.waypoints = [
            (0.0, 0.0, takeoff_height),                     # взлёт
            (forward_1, 0.0, takeoff_height),               # вперёд 1.5 м
            (forward_1, right_1, takeoff_height),           # направо 0.8 м
            (0.0, right_1, takeoff_height),                 # назад 1.5 м
            (0.0, right_1 + right_2, takeoff_height),       # направо 0.8 м
            (forward_2, right_1 + right_2, takeoff_height), # вперёд 1.5 м
            (forward_2, right_1 + right_2, 0.0)             # посадка
        ]
        
        self.current_wp_index = 0          # индекс текущей цели
        self.current_position = None        # последняя полученная позиция
        self.target_position = self.waypoints[0]  # текущая цель
        self.odom_received = False
        self.mission_complete = False
        
        # Подписка на одометрию
        self.odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            10
        )
        
        # Издатель для задания позиции (setpoint)
        self.setpoint_pub = self.create_publisher(
            PoseStamped,
            setpoint_topic,
            10
        )
        
        # Таймер для периодической отправки setpoint
        self.timer_period = 1.0 / control_rate
        self.timer = self.create_timer(self.timer_period, self.publish_setpoint)
        
        self.get_logger().info('Flight controller initialized. Waiting for odometry...')
    
    def odom_callback(self, msg):
        """Сохраняет текущую позицию и проверяет достижение цели."""
        self.current_position = msg.pose.pose.position
        if not self.odom_received:
            self.odom_received = True
            self.get_logger().info('Odometry received. Starting mission.')
        if not self.mission_complete:
            self.check_waypoint_reached()
    
    def check_waypoint_reached(self):
        """Проверяет, достигнута ли текущая цель, и переключает на следующую."""
        if self.current_position is None:
            return
        
        target = self.waypoints[self.current_wp_index]
        dx = self.current_position.x - target[0]
        dy = self.current_position.y - target[1]
        dz = self.current_position.z - target[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        if dist < self.tolerance:
            self.get_logger().info(
                f'Waypoint {self.current_wp_index} reached at '
                f'({self.current_position.x:.2f}, {self.current_position.y:.2f}, {self.current_position.z:.2f})'
            )
            
            if self.current_wp_index == len(self.waypoints) - 1:
                # Последняя точка (посадка) достигнута
                self.mission_complete = True
                self.get_logger().info('Mission complete. Landing...')
                # Через заданное время завершаем узел
                self.create_timer(self.landing_timeout, self.shutdown_node)
            else:
                self.current_wp_index += 1
                self.target_position = self.waypoints[self.current_wp_index]
                self.get_logger().info(f'Moving to waypoint {self.current_wp_index}: {self.target_position}')
    
    def publish_setpoint(self):
        """Публикует текущую целевую позицию."""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'  # или 'odom' – должно совпадать с системой координат одометрии
        pose_msg.pose.position.x = self.target_position[0]
        pose_msg.pose.position.y = self.target_position[1]
        pose_msg.pose.position.z = self.target_position[2]
        pose_msg.pose.orientation.w = 1.0  # без вращения
        self.setpoint_pub.publish(pose_msg)
    
    def shutdown_node(self):
        """Останавливает таймер и завершает узел."""
        self.get_logger().info('Shutting down flight controller.')
        self.timer.cancel()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = FlightController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
