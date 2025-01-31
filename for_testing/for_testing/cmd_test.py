import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion
import threading
import sys
import select
import termios
import tty

class Test(Node):
    def __init__(self):
        super().__init__('fuck')
        self.subscriper = self.create_subscription(PoseWithCovarianceStamped, '/initialpose', 
                                                   self.listener_callback,
                                                   10)

    def listener_callback(self, data):
        print(type(data))

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = Test()

    try:
        rclpy.spin(image_subscriber)
    except:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
