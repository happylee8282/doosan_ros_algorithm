import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class subscribe_test1(Node):
    def __init__(self):
        super().__init__('subscribe_test1')
        self.publisher = self.create_subscription(Twist,'ro', self.listener_callback, 10)

    def listener_callback(self, msg):
        print(f'x : {msg.linear.x}')
        print(f'z : {msg.angular.z}')

def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = subscribe_test1()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

