import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
from geometry_msgs.msg import Twist
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String

class YOLOTrackingPublisher(Node):
    def __init__(self):
        super().__init__('yolo_tracking_publisher')
        self.publisher_ = self.create_publisher(CompressedImage, 'AMR_image', 10)
        self.bridge = CvBridge()
        self.model = YOLO('/home/rokey2/bag_c/test_ws/best_aml.pt')
        self.cap = cv2.VideoCapture(1)
        self.timer = self.create_timer(0.1, self.img_timer_callback)
        self.publishe_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.miss = self.create_publisher(String, 'tracking_id', 10)
        self.cnt = 0
        self.j = -1

    def img_timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to capture frame from webcam.")
            return
        # YOLO 모델 추적 실행
        results = self.model.track(source=frame, show=False, tracker='bytetrack.yaml')
        for result in results:
            for detection in result.boxes.data:
                if len(detection) >= 7:
                    x1, y1, x2, y2, id, confidence, class_id = detection[:7]
                    if int(class_id) == 0 and float(confidence) >= 0.8:
                        self.subscriber_ = self.create_subscription(String, 'tracking_id', self.id_callback, 10)
                        if int(id) == int(self.ordered_id):
                            self.x1, self.x2, self.y1, self.y2 = float(x1), float(x2), float(y1), float(y2)
                            self.calculate()

        # `c` 메시지 수신 시 처리
        if self.j == 0:
            self.cnt += 1
        else:
            self.cnt = 0
        if self.cnt > 10:
            m = String()
            m.data = 'c'
            self.ordered_id = 'c'
            self.cnt = 0
            self.miss.publish(m)

    def id_callback(self, data):
        self.ordered_id = data.data

    def calculate(self):
        pass

    def destroy_node(self):
        super().destroy_node()
        self.cap.release()


class WayBack(Node):
    def __init__(self):
        super().__init__('way_back')
        self.no = self.create_subscription(String, 'tracking_id', self.no_callback, 10)
        self.j = -0.01

    def no_callback(self, data):
        if data.data == 'c':
            self.addd = self.create_publisher(Twist, '/cmd_vel', 10)
            msg = Twist()
            msg.linear.x = self.j
            self.addd.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node1 = YOLOTrackingPublisher()
    node3 = WayBack()
    executor = SingleThreadedExecutor()
    executor.add_node(node1)
    executor.add_node(node3)

    def dynamic_node_callback(msg):
        if msg.data == 'c':
            node1.get_logger().info('Activating node1 and node3 for "c" message.')
            executor.add_node(node1)
            executor.add_node(node3)
        else:
            node1.get_logger().info('Deactivating other nodes.')

    trigger_subscriber = rclpy.create_node('trigger_listener')
    trigger_subscriber.create_subscription(String, 'tracking_id', dynamic_node_callback, 10)

    executor.add_node(trigger_subscriber)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node1.destroy_node()
        node3.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
