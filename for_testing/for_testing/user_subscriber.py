import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class UserImage(Node):
    def __init__(self):
        super().__init__('user_subscriber')

        self.subscription = self.create_subscription(
            Image,
            'amr_image',
            self.image_callback,
            10
        )
        
        self.subscription
        self.bridge =CvBridge()

    def image_callback(self, data):
        frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')

        cv2.imshow('amr_image', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            rclpy.shutdown()
def main(args=None):
    rclpy.init(args=args)
    image_subscriber = UserImage()

    try:
        rclpy.spin(image_subscriber)
    except:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()