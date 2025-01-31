from setuptools import find_packages, setup

package_name = 'for_testing'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='oh',
    maintainer_email='oh@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # cli 명령어 = 디렉토리.파일이름:main
            'image_publisher = for_testing.image_publisher:main',
            'image_subscriber = for_testing.image_subscriber:main',
            'cmd_test = for_testing.cmd_test:main',
            'user_subscriber = for_testing.user_subscriber:main',
            'initial_pose = for_testing.test_init_pose:main',
            'send_waypoint = for_testing.send_waypoint:main',
            'send_goal_stop = for_testing.send_goal_stop:main',
            'publish_test = for_testing.publish_test:main',
            'subscribe_test = for_testing.subscribe_test:main',
            'publish_test1 = for_testing.publish_test1:main',
            'subscribe_test1 = for_testing.subscribe_test1:main',
            'yolo_publisher = for_testing.yolo_publisher:main',
            'yolo_subscriber = for_testing.yolo_subscriber:main',
            'yolo_tracking = for_testing.yolo_tracking:main',
            'june_tracking = for_testing.june_code:main',
        ],
    },
)
