import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class CameraImagePublisher(Node):
    def __init__(self):
        super().__init__('camera_image_publisher')

        self.publisher_ = self.create_publisher(
            Image,
            'camera_image',
            10
        )
        # 100Hz 기준. 30fps 맞추려면 0.3??
        timer_period = 0.1
        # 일정 주기마다 publish_image
        self.timer = self.create_timer(timer_period, self.publish_image)
        # 카메라로 사진 찍기
        self.cap = cv2.VideoCapture('/dev/video0')
        #self.cap = cv2.VideoCapture(2)

        self.bridge = CvBridge()
        
        # 카메라 오류
        if not self.cap.isOpened():
            self.get_logger().error('camera_open_failed')
            raise RuntimeError('Camera_not_accessible')
        
    def publish_image(self):
        ret, frame = self.cap.read()
        if ret: # 카메라가 연결되어 사진이 있으면
            # cv2 -> ros2 형태로 이미지 변환
            msg = self.bridge.cv2_to_imgmsg(frame, encoding = 'bgr8')
            # 변환된 이미지 발행
            self.publisher_.publish(msg)
            # 이미지 발행 되었는지 알려줌
            self.get_logger().info('image published')
        else:
            self.get_logger().info('fail to capture image')

    def destroy_node(self):
        # 종료시킬 때 카메라 종료
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    image_publisher = CameraImagePublisher()

    try:
        rclpy.spin(image_publisher)
    except:
        pass
    finally:
        image_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()