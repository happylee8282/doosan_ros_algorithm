
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
#from ultralytics.engine.results import Boxes
#from collections import defaultdict


class RawImageSubscriber(Node):
    def __init__(self):
        super().__init__('raw_image_subscriber')

        self.subscription = self.create_subscription(
            Image,
            'camera_image',
            self.image_callback,
            10
        )
        self.subscription
        self.bridge = CvBridge()
        # 모델 설정
        #self.model = YOLO('/home/oh/vision2/test_ws/best.pt')
        self.model = YOLO('yolov8n.pt')

        
    def image_callback(self, data):
        self.get_logger().info('frame received')
        # opencv image를 ros2 에 맞게 변환. encoding - bgr8
        frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')

        # inference
        self.results = self.model.track(frame, show=True)

        print(self.results[0].numpy().boxes.id)# xyxy

        #self.get_logger().info(self.results[0])
            
        #self.annotated_frame = self.results[0].plot()

        #self.boxes = Boxes(self.results)
        
        # 데이터 뽑아오기
        #self.boxes = self.results[0].boxes
        
        #self.track_ids = self.results[0].boxes.id.int().cpu().tolist()

        # 이후 for문에서 오류

        #for self.box in self.boxes:
        #    self.x1, self.y1, self.x2, self.y2 = self.boxes[0].xyxy
            #self.x, self.y, self.w, self.h = self.boxes
        
            #self.get_logger().info(self.x1)        
        

        # testing - 이미지가 잘 변환되어 나오는지 확인
        #cv2.imshow('received image', frame)
        # cv2.imshow('annotaed image', self.annotated_frame)
        # q누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = RawImageSubscriber()

    try:
        rclpy.spin(image_subscriber)
    except:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

'''
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

class RawImageSubscriber(Node):
    def __init__(self):
        super().__init__('raw_image_subscriber')

        self.subscription = self.create_subscription(
            Image,
            'camera_image',
            self.image_callback,
            10
        )
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')  # YOLO 모델 로드

    def image_callback(self, data):
        self.get_logger().info('Frame received')
        # ROS2 이미지를 OpenCV 이미지로 변환
        frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')

        # YOLO 추론 실행
        results = self.model(frame)

        #
        self.get_logger().info(results[0].boxes)

        # 바운딩 박스 정보 추출
        boxes = results[0].boxes  # 바운딩 박스
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 바운딩 박스 좌표 (x1, y1, x2, y2)

            # 바운딩 박스의 크기 (너비, 높이)
            width = x2 - x1
            height = y2 - y1

            # 바운딩 박스의 중간 좌표 (cx, cy)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # 결과 출력
            self.get_logger().info(f'Box: ({x1}, {y1}, {x2}, {y2}), Width: {width}, Height: {height}, Center: ({cx}, {cy})')

            # 바운딩 박스 시각화 (OpenCV로 표시)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f'({int(cx)}, {int(cy)})', (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # `matplotlib`을 사용하여 이미지 표시
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # 축 숨기기
        plt.show(block=False)
        plt.pause(0.1)  # 이미지 갱신을 위한 잠시 대기

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = RawImageSubscriber()

    try:
        rclpy.spin(image_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

'''