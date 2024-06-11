import sys
import numpy as np
from xarm_msgs.srv import PlanPose
from xarm_msgs.srv import PlanExec
from xarm_msgs.srv import PlanJoint
from geometry_msgs.msg import Pose
import rclpy
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener



class FunctionLib(Node):

    def __init__(self):
        super().__init__('xarm_planning_client')
        self.plan_cli = self.create_client(PlanPose, 'xarm_pose_plan')
        self.exec_cli = self.create_client(PlanExec, 'xarm_exec_plan')
        self.gripper_cli = self.create_client(PlanJoint, 'xarm_gripper_joint_plan')
        self.gripper_exec_cli = self.create_client(PlanExec, 'xarm_gripper_exec_plan')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        
    def open_gripper(self):
        self._move_gripper(0.0)
    	
    def close_gripper(self):
        self._move_gripper(0.85)
    	
    def _move_gripper(self,rad):
        plan = PlanJoint.Request()
        target = [rad] * 6
        plan.target = target
        plan_future = self.gripper_cli.call_async(plan)
        rclpy.spin_until_future_complete(self, plan_future)
        while not self.exec_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        exec_req = PlanExec.Request()
        exec_req.wait = True
        exec_future = self.gripper_exec_cli.call_async(exec_req)
        rclpy.spin_until_future_complete(self, exec_future)
        return exec_future.result()
    
    def move_arm(self,x,y,z,roll,pitch,yaw):
        target_pose = Pose()

        target_pose.position.x = x
        target_pose.position.y = y
        target_pose.position.z = z

        quat = self.rpy_to_quaternion(roll, pitch, yaw)

        target_pose.orientation.x = quat[0]
        target_pose.orientation.y = quat[1]
        target_pose.orientation.z = quat[2]
        target_pose.orientation.w = quat[3]

        self._move_arm_pose(target_pose)

    def _move_arm_pose(self,pose):
        while not self.plan_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        plan_req = PlanPose.Request()
        plan_req.target = pose
        plan_future = self.plan_cli.call_async(plan_req)
        rclpy.spin_until_future_complete(self, plan_future)
        res = plan_future.result()
        if res.success:
            self.get_logger().info('Planning success executing...')
            while not self.exec_cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('service not available, waiting again...')
            exec_req = PlanExec.Request()
            exec_req.wait = True
            exec_future = self.exec_cli.call_async(exec_req)
            rclpy.spin_until_future_complete(self, exec_future)
            return exec_future.result()
        else:
            self.get_logger().info('Planning Failed!')
            return False
        
    def get_current_end_effector_pose(self):
        pose = self._get_pose('link_tcp')
        trans = pose.transform.translation
        rot = pose.transform.rotation
        pose = np.zeros(6)
        pose[0] = trans.x
        pose[1] = trans.y
        pose[2] = trans.z
        rpy = self.quaternion_to_rpy(rot.x,rot.y,rot.z,rot.w)
        pose[3] = rpy[0]
        pose[4] = rpy[1]
        pose[5] = rpy[2]
        return pose

    def _get_pose(self,target):
        to_frame_rel = target
        from_frame_rel = 'world'
        t = None
        while True:
            try:
                t = self.tf_buffer.lookup_transform(to_frame_rel,from_frame_rel,rclpy.time.Time())
                return t
            except TransformException as ex:
                # self.get_logger().info(f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
                pass
            rclpy.spin_once(self)

        
    def rpy_to_quaternion(self,roll, pitch, yaw):
        """
        Convert roll, pitch, yaw to a quaternion.
        
        Parameters:
        roll (float): Rotation around the x-axis in radians
        pitch (float): Rotation around the y-axis in radians
        yaw (float): Rotation around the z-axis in radians
        
        Returns:
        tuple: A tuple representing the quaternion (x, y, z, w)
        """
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return (x, y, z, w)
    
    def quaternion_to_rpy(self, x, y, z, w):
        """
        Convert a quaternion to roll, pitch, yaw.
        
        Parameters:
        x (float): x component of the quaternion
        y (float): y component of the quaternion
        z (float): z component of the quaternion
        w (float): w component of the quaternion
        
        Returns:
        tuple: A tuple representing the roll, pitch, and yaw (in radians)
        """
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi / 2  # use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return (roll, pitch, yaw)