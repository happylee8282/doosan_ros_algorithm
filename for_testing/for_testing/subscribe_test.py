import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
import numpy as np

class UserImage(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription_rgb = self.create_subscription(
            CompressedImage,
            'AMR_image',
            self.listener_callback_rgb,
            10)
        self.subscription_rgb  # prevent unused variable warning

    def listener_callback_rgb(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Decode to color image
        cv2.imshow('RGB Image', image_np)
        cv2.waitKey(1)  # Display the image until a keypress


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