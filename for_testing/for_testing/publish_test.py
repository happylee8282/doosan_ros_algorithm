import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
from geometry_msgs.msg import Twist
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String


from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion, PoseStamped
from nav2_msgs.action import FollowWaypoints
from rclpy.action import ActionClient
import time



class YOLOTrackingPublisher(Node):
    def __init__(self):
        super().__init__('yolo_tracking_publisher')
        # 속도를 위해 압축된 이미지로 AMR에서 tracking한 이미지를 publish하는 topica
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
            if self.ordered_id != 'a' and self.ordered_id != 'c' and self.ordered_id != 's':
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
            m.data = 'c'
            self.ordered_id = 'c'
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
            if self.ordered_id == 'a':
                pass
            elif self.ordered_id == 'c':
                pass
            elif self.ordered_id == 's':
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


                    alpha = 0.1
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

'''
# 두번째 노드. tracking_id가 0이 들어오면 동작. 목적지까지 가는 부분을 넣으면 됨
class WaypointFollower(Node):
    def __init__(self):
        super().__init__('waypoint_follower')        
        
        self.no = self.create_subscription(String,'tracking_id',self.no_callback,10)
        self.i = 0.01

    def no_callback(self, data):
        if data.data == '0':
            
            # 아예 종료
            #super().destroy_node()
            self.addd = self.create_publisher(Twist,'/cmd_vel',10)
            msg = Twist()
            msg.linear.x = self.i
            self.addd.publish(msg)
'''

class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')
        self.initialpose_publisher = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.action_client = ActionClient(self, FollowWaypoints, '/follow_waypoints')
        self.finde_publisher = self.create_publisher(String, '/tracking_id', 10)

        self.is_shutdown = False  # 종료 플래그
        self.c_count = 0  # "c" 메시지 발행 횟수
        self.timer = None  # 타이머 핸들

        self.subscription = self.create_subscription(
            String,
            '/tracking_id',
            self.trigger_callback,
            10
        )
        self.get_logger().info('Waiting for trigger message "a" on /trigger_topic...')
        self.p = 0

    def trigger_callback(self, msg):
        print('\n\n\n\n\n\n\n\n\n\n\n')
        if msg.data == 'a':
            self.get_logger().info('Received trigger message "a". Setting up navigation...')
            self.publish_initial_pose()
            time.sleep(2)  # 초기 위치 설정 후 대기
            self.get_logger().info('Waiting for navigation to be ready...')
            self.wait_for_navigation_ready()
            self.send_goal()

        else:
            self.cancel_goal()
            self.get_logger().warn(f'Ignored message: {msg.data}')

    def publish_initial_pose(self):
        self.get_logger().info('Publishing initial pose...')
        initial_pose = PoseWithCovarianceStamped()
        initial_pose.header.frame_id = 'map'
        initial_pose.header.stamp = self.get_clock().now().to_msg()

        initial_pose.pose.pose.position.x = 0.026354703738017037
        initial_pose.pose.pose.position.y = -0.023555173715895102
        initial_pose.pose.pose.orientation = Quaternion(
            x=0.0,
            y=0.0,
            z=-0.07846075992136109,
            w=0.9969172027568601
        )
        initial_pose.pose.covariance = [0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891909122467]
        self.initialpose_publisher.publish(initial_pose)
        self.get_logger().info('Initial pose published successfully.')

    def wait_for_navigation_ready(self):
        self.get_logger().info('Waiting for FollowWaypoints action server...')
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server not available. Is navigation running?')
            return
        self.get_logger().info('FollowWaypoints action server is ready.')


    def send_goal(self):
        waypoints = []

        # 웨이포인트 정의
        waypoint1 = PoseStamped()
        waypoint1.header.frame_id = "map"
        waypoint1.pose.position.x = 0.20045082073635237
        waypoint1.pose.position.y = -0.11526315559203755
        waypoint1.pose.orientation.z = -0.5501947083050299
        waypoint1.pose.orientation.w = 0.835036396184707
        waypoints.append(waypoint1)

        waypoint2 = PoseStamped()
        waypoint2.header.frame_id = "map"
        waypoint2.pose.position.x = 0.2885849106796945
        waypoint2.pose.position.y = -0.7729320617606924
        waypoint2.pose.orientation.z = -0.998553042937628
        waypoint2.pose.orientation.w = 0.053775649136049555
        waypoints.append(waypoint2)


        waypoint3 = PoseStamped()
        waypoint3.header.frame_id = "map"
        waypoint3.pose.position.x = -0.43766810834163267
        waypoint3.pose.position.y = -0.6090071044161587
        waypoint3.pose.orientation.z = 0.9647924028389457
        waypoint3.pose.orientation.w = 0.2630125841556892
        waypoints.append(waypoint3)

        waypoint4 = PoseStamped()
        waypoint4.header.frame_id = "map"
        waypoint4.pose.position.x = -1.024580178661337
        waypoint4.pose.position.y = -0.21175594593394653
        waypoint4.pose.orientation.z = 0.9813011653381158
        waypoint4.pose.orientation.w = 0.19247862973861757
        waypoints.append(waypoint4)

        waypoint5 = PoseStamped()
        waypoint5.header.frame_id = "map"
        waypoint5.pose.position.x = -1.4703407287627268
        waypoint5.pose.position.y = -0.292804214493149
        waypoint5.pose.orientation.z = -0.6196984326209984
        waypoint5.pose.orientation.w = 0.784840017205467
        waypoints.append(waypoint5)

        waypoint6 = PoseStamped()
        waypoint6.header.frame_id = "map"
        waypoint6.pose.position.x = -1.5726163715771126
        waypoint6.pose.position.y = -0.3743943690241698
        waypoint6.pose.orientation.z = -0.7126831075538593
        waypoint6.pose.orientation.w = 0.7014861283071635
        waypoints.append(waypoint6)


        goal_msg = FollowWaypoints.Goal()
        goal_msg.poses = waypoints

        self.get_logger().info('Sending goal with waypoints...')
        self._send_goal_future = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        try:
            self.goal_handle = future.result()
            if not self.goal_handle.accepted:
                self.get_logger().error('Goal rejected by Action server.')
                return
            self.get_logger().info('Goal accepted by Action server.')
            self._get_result_future = self.goal_handle.get_result_async()
            self._get_result_future.add_done_callback(self.get_result_callback)
        except Exception as e:
            self.get_logger().error(f'Failed to send goal: {e}')

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Moving towards Waypoint {feedback.current_waypoint}')

    def cancel_goal(self):
        try:
            if self.goal_handle is not None:
                self.get_logger().info('Attempting to cancel the goal...')
                cancel_future = self.goal_handle.cancel_goal_async()
                cancel_future.add_done_callback(self.cancel_done_callback)
            else:
                self.get_logger().info('No active goal to cancel.')
        except:
            pass

    # def cancel_done_callback(self, future):
    #     cancel_response = future.result()
    #     if cancel_response.accepted:
    #         self.get_logger().info('Goal cancellation accepted. Exiting program...')
    #         self.destroy_node()
    #         rclpy.shutdown()
    #         sys.exit(0)  # Exit the program after successful cancellation
    #     else:
    #         self.get_logger().info('Goal cancellation failed or no active goal to cancel.')
    
    def cancel_done_callback(self, future):
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self.get_logger().info('Goal cancellation accepted. Exiting program...')
            self.destroy_node()
        else:
            self.get_logger().info('Goal cancellation failed or no active goal to cancel.')

    def get_result_callback(self, future):
        try:
            result = future.result().result
            missed_waypoints = result.missed_waypoints
            if missed_waypoints:
                self.get_logger().warn(f'Missed waypoints: {missed_waypoints}')
            else:
                self.get_logger().info('All waypoints completed successfully!')
                self.cancel_goal()
                msg = String()
                msg.data = 'c'
                self.finde_publisher.publish(msg)

                self.safe_shutdown()
                
        except Exception as e:
            self.get_logger().error(f'Error getting result: {e}')

        finally:
            self.start_finde_publishing()

    def start_finde_publishing(self):
        """/finde 메시지를 5초 동안 5번만 발행"""
        self.get_logger().info('Starting periodic publishing of "c" to /finde...')
        self.c_count = 0
        self.timer = self.create_timer(0.1, self.publish_finde_message)  # 1초 간격으로 발행

    def publish_finde_message(self):
        """5번 메시지를 발행한 후 타이머 중지"""
        if self.c_count < 5:
            msg = String()
            msg.data = 'c'
            self.finde_publisher.publish(msg)
            self.get_logger().info(f'Published "c" to /finde topic. Count: {self.c_count + 1}')
            self.c_count += 1
        else:
            self.get_logger().info('Finished publishing "c" messages.')
            self.timer.cancel()  # 타이머 중지
            self.safe_shutdown()

    def safe_shutdown(self):
        if not self.is_shutdown:
            self.get_logger().info('Shutting down node...')
            if self.timer:
                self.timer.cancel()  # 타이머 중지
                #self.destroy_node()
            self.is_shutdown = True

# 세번재 노드. tracking_id가 -가 들어오면 동작. 차고지로 복귀. 굳이 이렇게 안하고 두번째 노드에서 subscribe받은 문자를 기준으로 역순으로 진행해도 될듯
class WayBack(Node):
    def __init__(self):
        super().__init__('way_back')        
        
        self.no = self.create_subscription(String,'tracking_id',self.no_callback,10)
        self.i = 0
        self.j = 0.1

    def no_callback(self, data):
        if data.data == 'c':
            
            # 아예 종료
            #super().destroy_node()
            self.addd = self.create_publisher(Twist,'/cmd_vel',10)
            msg = Twist()
            msg.angular.z = self.j
            self.addd.publish(msg)

class Stop(Node):
    def __init__(self):
        super().__init__('stop')        
        
        self.no = self.create_subscription(String,'tracking_id',self.no_callback,10)

    def no_callback(self, data):
        if data.data == 's':
            
            # 아예 종료
            #super().destroy_node()
            self.addd = self.create_publisher(Twist,'/cmd_vel',10)
            msg = Twist()
            msg.angular.z = 0.0
            msg.linear.x = 0.0
            self.addd.publish(msg)


def main(args=None):

    rclpy.init(args=args)
    node1 = YOLOTrackingPublisher()
    node2 = NavigationNode()
    node3 = WayBack()
    node4 = Stop()
    # 멀티스레드 하니까 엄청 느려서 싱글스레드로 돌림.
    executor = SingleThreadedExecutor()
    executor.add_node(node1)
    executor.add_node(node2)
    executor.add_node(node3)
    executor.add_node(node4)

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
