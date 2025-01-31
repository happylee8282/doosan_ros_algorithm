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
        # 속도를 위해 압축된 이미지로 AMR에서 tracking한 이미지를 publish하는 topic
        self.publisher_ = self.create_publisher(CompressedImage, 'AMR_image', 10)
        # cv2 - ros2간 이미지 변환을 위한 함수
        self.bridge = CvBridge()
        # yolo pt file
        self.model = YOLO('/home/rokey2/bag_c/test_ws/best_aml.pt')
        # 촬영 시작
        self.cap = cv2.VideoCapture(0)
        # 사진을 처리
        self.timer = self.create_timer(0.1, self.img_timer_callback)  # Adjust timer frequency as needed

        # AMR에게 모터 제어 정보를 publish하는 topic
        self.publishe_cmd = self.create_publisher(Twist,'/cmd_vel', 10)
        # 추적 실패시 동작할 코드
        self.miss = self.create_publisher(String, 'tracking_id', 10)
        # 추적 실패시
        self.cnt = 0
        self.j = -1


        

    # 이미지 callback   
    def img_timer_callback(self):
        # frame - 우리가 원하는 사진
        ret, frame = self.cap.read()
        # 카메라가 연결이 안됬다면
        if not ret:
            self.get_logger().warn("Failed to capture frame from webcam.")
            return

        # Run tracking and get results
        results = self.model.track(source=frame, show=False, tracker='bytetrack.yaml')

        # results - tracking에 필요한 여러장의 사진
        # Iterate over results to extract data

        # 추적 대상을 놓쳤는지 확인. 추적 대상이 사진에 있으면 +를 할거라 for문 이후 j == 0이면 놓친상태, -면 추적 명령 아닌 상태
        try:
            if self.ordered_id != '0' and self.ordered_id != '-':
                print(f'ord : {self.ordered_id}')
                self.j = 0
            else:
                self.j = -1
        except:
            pass

        
        for result in results:
            # 사진에 있는 박스를 순차적으로 보기 위함
            i=0
            # 사진 한장에 있는 박스들의 데이터
            for detection in result.boxes.data:
                # 박스가 가지고 있는 데이터가 7개가 넘을 때
                if len(detection) >= 7:
                    # x1, y1, x2, y2, id(클래스와 무관하게 순서대로 1부터 매겨짐), 신뢰도, 클래스
                    x1, y1, x2, y2, id, confidence, class_id = detection[:7]
                    
                    # 첫번째 클래스만(이 경우에는 car), 신뢰도가 90% 넘은 경우만 박스침
                    if int(class_id) == 0 and float(confidence) >= 0.8:
                        # 강사님 코드 - 오류인듯(근데 뒤에 사용되는거 고치는거 깜빡해서 넣어둠)
                        track_id = result.boxes.id[i] if len(detection) > 7 else None

                        # 우리가 알려준 id라면, 처음에는 id가 입력이 안들어 오므로 try로 실행
                        try:
                            # 잘못된 코드
                            # track_ids = results[0].boxes.id.int().cpu().tolist()

                            # 우리가 원하는 객체라면
                            # PC로부터 id를 subscribe
                            self.subscriber_ = self.create_subscription(String,'tracking_id',self.id_callback, 10)
                            if int(id) == int(self.ordered_id):
                                # 박스의 각 좌표를 얻어냄
                                self.j+=1
                                self.x1, self.x2, self.y1, self.y2 = float(x1), float(x2), float(y1), float(y2)
                                # 추적 상태를 위한 변수
                                # j+=1
                        # id를 받지 못했으면 박스는 없음
                        except:
                            self.x1, self.x2, self.y1, self.y2 = 0.0, 0.0, 0.0, 0.0

                        # /cmd_vel publish
                        self.calculate() 

                        # 중심점 좌표
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)

                        # Draw bounding box and center point on the frame
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        label_text = f'Conf: {confidence:.2f} Class: {int(class_id)}'
                        if id is not None:
                            label_text = f'Track_ID: {int(id)}, ' + label_text

                        cv2.putText(frame, label_text, (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # 이 박스는 끝났고 다음 박스 확인
                i+=1

        # 추적 실패시
        if self.j == 0:
            self.cnt+=1
        else:
            self.cnt = 0

        if self.cnt > 10:
            m = String()
            m.data = '0'
            self.ordered_id = '0'
            self.cnt = 0
            self.miss.publish(m)

        # Compress the frame and convert it to a ROS 2 CompressedImage message
        _, compressed_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        compressed_img_msg = CompressedImage()
        compressed_img_msg.header.stamp = self.get_clock().now().to_msg()
        compressed_img_msg.format = "jpeg"
        compressed_img_msg.data = compressed_frame.tobytes()

        # Publish the compressed image
        self.publisher_.publish(compressed_img_msg)

    # id callback
    # AMR의 상태, 추적할 id를 받음
    def id_callback(self, data):
        self.ordered_id = data.data
        
    # 선속도 가속도 계산해서 /cmd_vel에 publish
    def calculate(self):    
        # 0이거나 - 이면 모터에 전달해줄 값 계산
        # 0이나 -는 waypoint도달, 복귀
        try:
            if self.ordered_id == '0':
                pass
            elif self.ordered_id == '-':
                pass
            else:

                # cmd callback
                # callback이어서 처음에 구동하면 바로 오류남. 이를 피하기 위한 try
                try:
                    width = (self.x2-self.x1)
                    height = (self.y2-self.y1)
                    center_x = (self.x1+self.x2)/2.0
                    center_y = (self.y1+self.y2)/2.0
                    area = width*height

                    # 앞에 차량이 끼어들어 박스가 작아진다면. 근데 실제 구동해보니 박스를 추정하기 때문에 굳이 필요 없음. 완벽하게 앞을 가릴때만 필요?
                    '''
                    ratio = width/height
                    threshold = 1.0
                    if ratio > threshold:
                        height = width/threshold
                        self.y2 = self.y1 + height
                        if self.y2 >= 480.0:
                            self.y2 = 480.0
                        center_y = (self.y1 + self.y2)/2
                    '''

                    # 화면의 x 중심점
                    x_center = 640.0/2.0
                    # 화면에서 추적 대상과 유지할 거리
                    y_threshold = 480.0/8.0

                    # 좌회전이 +z이므로 왼쪽에 있을 때 양수값
                    diff_x = x_center-center_x
                    # 전진이 +x이므로 멀리 있을 때 양수값
                    diff_y = center_y - y_threshold
                    # 화면 아래에서부터 거리 - 실제 거리에 비례
                    dist = center_y



                    # alpha(각속도), beta(선속도)는 상수(실험으로 수정)
                    # 회전은 y(거리)가 멀수록 diff_x에 둔감하게 반응. 
                    # 속력은 diff_y에만 반응. diff_y에 비례. 클수록 더욱 크기, 작을수록 더욱 작게. 

                    # 처음엔 중심점 거리를 보려 했으나 가까워지면 박스의 크기만 커지고 중심점은 차이가 크지 않아 박스 크기를 이용할 예정


                    alpha = 0.15
                    beta = 0.0000007

                    mo = Twist()

                    # 뒤에 이상하거 붙은건 소수점 둘째자리까지 표현하려는거 round가 잘 안되서 그럼
                    z = ((alpha * (diff_x/(dist+1)))//0.001)/1000# motor maximum = 2.84
                    # 박스가 아래에 붙으면 - 차량이 너무 가까우면
                    # 귀찮아서 상수로 둠. 사실 계산해야함
                    if self.y2 > 470:
                        x = -0.02
                    else:
                        # 박스가 너무 작은 경우 정지. 근데 필요 없을듯. 나중에 추적 실패에 처리하면 될듯
                        if area > 1.0:
                            x = ((beta * (640*480 - area - 162060.0))//0.001)/1000
                        else:
                            x = 0.0
                    #x = ((0.06 * (diff_y/beta)**3)//0.001)/1000  # motor maximum = 0.22

                    # Dynamixel의 한계가 있기 때문에 최대, 최소 설정
                    if z > 2.8:
                        z = 2.8
                    elif z < -2.8:
                        z = -2.8
                    if x > 0.22:
                        x = 0.22
                    elif x < -0.22:
                        x = -0.22

                    
                    # 선속도 - x, 각속도 - z
                    mo.linear.x = x
                    mo.angular.z = z
                    # publish
                    self.publishe_cmd.publish(mo)
                # 오류나면 넘어감. 처음 실행시 작동
                except:
                    pass
        
        except:
            pass


    # 노드 파쇄기
    def destroy_node(self):
        super().destroy_node()
        self.cap.release()


# 두번째 노드. tracking_id가 0이 들어오면 동작. 목적지까지 가는 부분을 넣으면 됨
class WaypointFollower(Node):
    def __init__(self):
        super().__init__('waypoint_follower')        
        
        self.no = self.create_subscription(String,'tracking_id',self.no_callback,10)
        self.miss = self.create_publisher(String, 'tracking_id', 10)
        self.i = 0.3
        self.j = 0

    def no_callback(self, data):
        if data.data == '0':
            
            # 아예 종료
            #super().destroy_node()
            self.addd = self.create_publisher(Twist,'/cmd_vel',10)
            if self.j < 20:
                msg = Twist()
                msg.angular.z = self.i
                self.addd.publish(msg)
            else:
                m = String()
                m.data = '-'
                self.miss.publish(m)
                self.j = 0
                

# 세번재 노드. tracking_id가 -가 들어오면 동작. 차고지로 복귀. 굳이 이렇게 안하고 두번째 노드에서 subscribe받은 문자를 기준으로 역순으로 진행해도 될듯
class WayBack(Node):
    def __init__(self):
        super().__init__('way_back')        
        
        self.no = self.create_subscription(String,'tracking_id',self.no_callback,10)
        self.j = 0.03

    def no_callback(self, data):
        if data.data == '-':
            
            # 아예 종료
            #super().destroy_node()
            self.addd = self.create_publisher(Twist,'/cmd_vel',10)
            msg = Twist()
            msg.linear.x = self.j
            self.addd.publish(msg)



def main(args=None):

    rclpy.init(args=args)
    node1 = YOLOTrackingPublisher()
    node2 = WaypointFollower()
    node3 = WayBack()
    # 멀티스레드 하니까 엄청 느려서 싱글스레드로 돌림.
    executor = SingleThreadedExecutor()
    executor.add_node(node1)
    executor.add_node(node2)
    executor.add_node(node3)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node1.destroy_node()
        node2.destroy_node()
        node3.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()
 


if __name__ == '__main__':
    main()
