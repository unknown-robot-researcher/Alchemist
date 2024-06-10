import rospy, moveit_commander,roslaunch, math
import moveit_msgs.msg, geometry_msgs.msg
import time, actionlib,sys,tf, tf2_ros, copy
import numpy as np
from numpy import linalg as LA
from tf import TransformListener
from std_msgs.msg import String, Bool
import geometry_msgs.msg, control_msgs
from actionlib import SimpleActionClient, GoalStatus
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from std_srvs.srv import Empty
from moveit_commander.conversions import pose_to_list
from moveit_msgs.msg import CollisionObject, AttachedCollisionObject
from shape_msgs.msg import SolidPrimitive  

class FunctionLib:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)

        self.robot = moveit_commander.RobotCommander()

        self.scene = moveit_commander.PlanningSceneInterface()

        group_name = "manipulator"
        self.group = moveit_commander.MoveGroupCommander(group_name)

        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
		                                       moveit_msgs.msg.DisplayTrajectory,
		                                       queue_size=20)

        self.gripper_publisher = rospy.Publisher('/icl_gripper/gripper_cmd/goal',
		                                       control_msgs.msg.GripperCommandActionGoal,
		                                       queue_size=20)

        self.plan=None
        self.delta=5
        self.object_dict={}
        self.object_marker_dict={}
        self.tf_listener = tf.TransformListener()
        self.constraints = moveit_msgs.msg.Constraints()
        self.object_dimensions = {}
        
        self.parse_dimensions("/home/ulasberkkarli/natural_robot/LLM/Lib/ur5/env_prompts/basic.txt")

        rospy.sleep(2)

        self.setup_the_scene()

    def get_object_dimensions(self,object_name):
        if object_name is not None:
            object_name = object_name.lower()
            obj_space = object_name.replace("_"," ")
            tokens = obj_space.split()
            tokens_first = tokens.pop(0)
            tokens.append(tokens_first)
            new_name=" ".join(tokens)
            if self.object_dimensions.has_key(object_name):
                return [self.object_dimensions[object_name]["radius"], self.object_dimensions[object_name]["height"]]
            elif self.object_dimensions.has_key(obj_space):
                return [self.object_dimensions[obj_space]["radius"], self.object_dimensions[obj_space]["height"]]
            elif self.object_dimensions.has_key(new_name):
                return [self.object_dimensions[new_name]["radius"], self.object_dimensions[new_name]["height"]]
            else:
                print(object_name, "not found in the dictionary, please enter a valid object name")
                rospy.logwarn(object_name, "not found in the dictionary, please enter a valid object name")
                exit()
        else:
            print("object name is None when retriving the object's dimension, please enter a valid object name")
            rospy.logwarn("object name is None when retriving the object's dimension, please enter a valid object name")
            exit()


    def parse_dimensions(self, filename):
        with open(filename, 'r') as file:
            for line in file:
                # Split the line into object name and dimensions
                parts = line.strip().split(":")
                if len(parts) != 2:
                    continue

                object_name = parts[0].strip().lower()
                dimensions = parts[1].split("and")

                # Extract radius and height from the dimensions
                radius = None
                height = None
                for dim in dimensions:
                    dim_parts = dim.split("=")
                    if len(dim_parts) != 2:
                        continue

                    dim_name = dim_parts[0].strip().lower()
                    dim_value = dim_parts[1].strip().split()[0]  # Get the numeric part of the value
                    content = "None"

                    if dim_name == "radius":
                        radius = float(dim_value)
                    elif dim_name == "height":
                        height = float(dim_value)
                    elif dim_name == "content":
                        content = dim_parts[1].lower()

                # Add dimensions to the object_dimensions dictionary
                if object_name and radius is not None and height is not None:
                    self.object_dimensions[object_name] = {'radius': radius/100, 'height': height/100, 'content': content}
            # print(self.object_dimensions)
            
    def move_to_home_position(self):
        self.group.set_max_velocity_scaling_factor(0.5)
        self.group.set_max_acceleration_scaling_factor(0.5)
        self.open_gripper()
        joint_goal=[-0.4587190786944788, -1.599515740071432, 1.7295589447021484, -0.1305630842791956, 1.8112632036209106, -0.0008500258075159195]
        self.group.go(joint_goal,wait=True)
        self.group.stop()

    def get_current_end_effector_pose(self):
        pose=self.group.get_current_pose().pose
        roll,pitch,yaw=self.quaternion_to_euler(pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w)
        return [pose.position.x,pose.position.y,pose.position.z,roll,pitch,yaw]
    
    def check_end_effector_reached_desired_target(self,target):
        current_ee_location=self.get_current_end_effector_pose()
        diff = np.linalg.norm(np.array(target[0:3])-np.array(current_ee_location[0:3]))
        # print(diff)
        return diff<0.02

    def close_gripper(self,name):
        name = name.lower()
        rospy.sleep(1)
        self.attach_object_to_gripper(name)
        grip = control_msgs.msg.GripperCommandActionGoal()
        grip.goal.command.position = 0.8
        grip.goal.command.max_effort = 100
        self.gripper_publisher.publish(grip)
        rospy.sleep(1)
        ##lift up
        ee_pose = self.get_current_end_effector_pose()
        self.go(ee_pose[0], ee_pose[1], ee_pose[2]+0.1, ee_pose[3], ee_pose[4], ee_pose[5])

        return True

    def open_gripper(self):
        rospy.sleep(1)
        grip = control_msgs.msg.GripperCommandActionGoal()
        grip.goal.command.position = 0.0
        grip.goal.command.max_effort = 100
        self.gripper_publisher.publish(grip)
        self.detach_object_from_gripper()

        return True
    
    def is_marker_visible(self,marker_number):
        start_time = time.time()
        timeout = 15
        while time.time() - start_time <= timeout and not rospy.is_shutdown(): 
            try:
                self.tf_listener.waitForTransform('/world', '/ar_marker_' + str(marker_number), rospy.Time(0), rospy.Duration(3.0))
                trans, rot = self.tf_listener.lookupTransform('/world', '/ar_marker_' + str(marker_number), rospy.Time(0))
                eul = self.quaternion_to_euler(rot[0], rot[1], rot[2], rot[3])
                return True
            except (tf2_ros.TransformException, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logwarn("Marker %d not found. Waiting for marker to appear... Error: %s", marker_number, str(e))
                rospy.sleep(1.0)
        rospy.logwarn("Timeout occurred while waiting for marker %d to appear", marker_number)
        return False

    def get_marker_location(self, marker_number):
        start_time = time.time()
        timeout = 15
        marker_size = 0.05
        while time.time() - start_time <= timeout and not rospy.is_shutdown(): 
            try:
                self.tf_listener.waitForTransform('/world', '/ar_marker_' + str(marker_number), rospy.Time(0), rospy.Duration(3.0))
                trans, rot = self.tf_listener.lookupTransform('/world', '/ar_marker_' + str(marker_number), rospy.Time(0))
                eul = self.quaternion_to_euler(rot[0], rot[1], rot[2], rot[3])

                if trans[2] < -0.11:
                    trans[2]= -0.11

                w_T_m = np.zeros([4,4])
                w_T_m[0:3,0:3] = self.quaternion_rotation_matrix(rot)
                w_T_m[0:3,3] = trans
                w_T_m[3,3] = 1

                m_T_o = np.zeros([4,4])
                m_T_o[0:3,0:3] = np.identity(3)
                m_T_o[0,3] = marker_size 
                m_T_o[3,3] = 1
                
                w_T_o = np.matmul(w_T_m, m_T_o)

                if eul[0] > 0:
                    eul[0] -= 180
                else:
                    eul[0] += 180

                z_offset = -0.00
                return [w_T_o[0,3],w_T_o[1,3],w_T_o[2,3] + z_offset, eul[0], eul[1], eul[2]]
            except (tf2_ros.TransformException, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logwarn("Marker %d not found. Waiting for marker to appear... Error: %s", marker_number, str(e))
                rospy.sleep(1.0)
        rospy.logwarn("Timeout occurred while waiting for marker %d to appear", marker_number)
        exit()

    def get_grasp_orientation(self, top=False):
        if top:
            return [180, 0, 0]
        else:
            return [180, 0, np.random.uniform(-90.0, 90.0)]
        

    def get_pour_orientation(self):
        return [180, 0, np.random.uniform(-90.0, 90.0)]

    def move_arm(self, x, y, z, roll, pitch ,yaw):
        waypoints=[]
        ori=self.euler_to_quaternion(roll,pitch,yaw)

        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.position.x = x
        pose_goal.position.y = y
        pose_goal.position.z = z

        pose_goal.orientation.w = ori[0]
        pose_goal.orientation.x = ori[1]
        pose_goal.orientation.y = ori[2]
        pose_goal.orientation.z = ori[3]

        current_position=self.group.get_current_pose().pose.position

        dx = (x-current_position.x)/self.delta
        dy = (y-current_position.y)/self.delta
        dz = (z-current_position.z)/self.delta
        pose=geometry_msgs.msg.Pose()

        for step in range(1,self.delta-1):
            print(current_position.z+dz*step)
            pose=geometry_msgs.msg.Pose()
            pose.position.x = (current_position.x+dx*step)
            pose.position.y = (current_position.y+dy*step)
            pose.position.z = (current_position.z+dz*step)
            pose.orientation.w = ori[0]
            pose.orientation.x = ori[1]
            pose.orientation.y = ori[2]
            pose.orientation.z = ori[3]

            waypoints.append(pose)

        waypoints.append(pose_goal)

        (plan, fraction) = self.group.compute_cartesian_path(
            waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
        )

        self.plan=plan

    def go(self, x, y, z, roll, pitch ,yaw, pour=False, orientation_constraint=True, position_constraint=True, joint_constraint=True, velocity=0.8, acceleration=0.5):
            ori=self.euler_to_quaternion(roll,pitch,yaw)

            pose_goal = geometry_msgs.msg.Pose()
            pose_goal.position.x = x
            pose_goal.position.y = y
            pose_goal.position.z = z
            z_constraint = -0.055

            if position_constraint:
                if(z < z_constraint):
                    rospy.logwarn("z constraint!")
                    pose_goal.position.z = z_constraint
            
            pose_goal.orientation.w = ori[0]
            pose_goal.orientation.x = ori[1]
            pose_goal.orientation.y = ori[2]
            pose_goal.orientation.z = ori[3]

            self.group.set_max_velocity_scaling_factor(velocity)
            self.group.set_max_acceleration_scaling_factor(acceleration)

            self.group.set_pose_target(pose_goal)

            self.enable_constraints(pose_goal, ori_flag=orientation_constraint, joint_flag=joint_constraint)

            flag = self.group.go(wait=True)
            count = 0
            while not flag:
                if not pour:
                    print("Motion planning failed, adding 1cm to z...")
                    pose_goal.position.z += 0.005
                else:
                    print("replanning for pouring...")
                    return flag
                self.group.set_pose_target(pose_goal)
                flag = self.group.go(wait=True)
                if count > 15:
                    print("Motion planning failed, the marker position might be unachievable. Please try to change marker position and orientation and try again.")
                    rospy.logwarn("Motion planning failed, the marker position might be unachievable. Please try to change marker position and orientation and try again.")
                    self.move_to_home_position()
                    exit()
                count+=1

            self.group.stop()
            self.group.clear_pose_targets()

            # Clear the constraints after planning and execution
            self.group.clear_path_constraints()

            return flag

    def enable_constraints(self, pose_goal, ori_flag, joint_flag):
        if ori_flag or joint_flag:
            self.constraints.name = "constraint"

            # Create a constraint for the wrist orientation
            ori_constraint = moveit_msgs.msg.OrientationConstraint()
            ori_constraint.header.frame_id = self.group.get_pose_reference_frame()
            ori_constraint.link_name = self.group.get_end_effector_link()  # Assuming end effector link is used for orientation control
            ori_constraint.orientation = pose_goal.orientation
            ori_constraint.absolute_x_axis_tolerance = 0.08  # Define your desired tolerances here
            ori_constraint.absolute_y_axis_tolerance = 0.08
            ori_constraint.absolute_z_axis_tolerance = 3.14  # Large tolerance to allow any wrist orientation
            ori_constraint.weight = 1.0

            # Create a joint constraint to restrict wrist 1's motion (prevent elbow-down moiton planning)
            joint_constraint = moveit_msgs.msg.JointConstraint()
            # the bound to be achieved is [position - tolerance_below, position + tolerance_above]
            joint_constraint.position = 0
            joint_constraint.tolerance_above = 1.57
            joint_constraint.tolerance_below = 1.57
            joint_constraint.weight = 1

            joint_constraint.joint_name = "wrist_1_joint"

            # Apply the constraints
            if ori_flag:
                self.constraints.orientation_constraints.append(ori_constraint)
            if joint_flag:
                self.constraints.joint_constraints.append(joint_constraint)
                
            self.group.set_path_constraints(self.constraints)

    def display_trajectory(self):
        if self.plan==None:
            pass
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory.append(self.plan)
        self.display_trajectory_publisher.publish(display_trajectory)

    def execute_plan(self):
        if self.plan==None:
            pass
        self.group.execute(self.plan, wait=True)

    def add_cylinder_to_workspace(self,name,marker_number=0,x=0.0,y=0.0,z=0.0,roll=0.0,pitch=0.0,yaw=0.0):
        if not marker_number == 0:
            x,y,z,roll,pitch,yaw = self.get_marker_location(marker_number)
        name = name.lower()
        if not self.object_dimensions.has_key(name):
            space = name.replace("_"," ")
            tokens = space.split()
            tokens_first = tokens.pop(0)
            tokens.append(tokens_first)
            new_name=" ".join(tokens)
            name = new_name
        x=float(x)
        y=float(y)
        z=float(z)
        height=self.get_object_dimensions(name)[1] + 0.01   # adding 2cm buffer
        radius=self.get_object_dimensions(name)[0]
        if z < -0.06:
            z=z+height/2.0
        self.object_dict[name]=[x,y,z,radius,height]
        pose = geometry_msgs.msg.PoseStamped()
        pose.header.frame_id = "world"

        eul=self.euler_to_quaternion(roll,pitch,yaw)

        w_T_m = np.zeros([4,4])
        w_T_m[0:3,0:3] = self.quaternion_rotation_matrix([eul[1],eul[2],eul[3],eul[0]])#self.euler_to_rotation_matrix(roll,pitch,yaw)
        w_T_m[0,3] = x
        w_T_m[1,3] = y
        w_T_m[2,3] = z
        w_T_m[3,3] = 1

        m_T_o = np.zeros([4,4])
        m_T_o[0:3,0:3] = np.identity(3)
        tokens = name.split()
        # if "cylinder" in tokens:
        #     # print(name, "add offset for cylinders")
        #     m_T_o[0,3] = radius + 0.01

        m_T_o[0,3] = radius
        m_T_o[3,3] = 1
                
        w_T_o = np.matmul(w_T_m, m_T_o)
        
        pose.pose.orientation.w = eul[0]
        pose.pose.orientation.x = eul[1]
        pose.pose.orientation.y = eul[2]
        pose.pose.orientation.z = eul[3]

        #pose.pose.orientation.w = 1.0
        pose.pose.position.x = w_T_o[0,3] #add cylinder from an offset of marker
        pose.pose.position.y = w_T_o[1,3] #add cylinder from an offset of marker
        pose.pose.position.z = z

        self.scene.add_cylinder(name, pose, height, radius)

    def add_box_to_workspace(self,name,x,y,z,sx,sy,sz):
        self.object_dict[name]=[x,y,z,sx,sy,sz]
        pose = geometry_msgs.msg.PoseStamped()
        pose.header.frame_id = "world"
        pose.pose.orientation.w = 1.0
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z

        self.scene.add_box(name, pose, size=(sx,sy,sz))

    def attach_object_to_gripper(self, name):
        name = name.lower()
        obj_space = name.replace("_", " ")
        tokens = obj_space.split()
        tokens_first = tokens.pop(0)
        tokens.append(tokens_first)
        new_name = " ".join(tokens)
        print(name)
        
        # Try to find the object using the original name
        try:
            obj = self.scene.get_objects([name])
            obj_pose = self.scene.get_object_poses([name])[name]

        except:
            # If not found, try with obj_space
            try:
                obj = self.scene.get_objects([obj_space])
                name = obj_space
                obj_pose = self.scene.get_object_poses([name])[name]
        
            except:
                # If still not found, try with new_name
                try:
                    obj = self.scene.get_objects([new_name])
                    name = new_name
                    obj_pose = self.scene.get_object_poses([name])[name]

                except:
                    # If object not found, print an error message and exit
                    print("Object not in scene, please add it into the workspace before grasping it!")
                    rospy.logwarn("Object not in scene, please add it into the workspace before grasping it!")
                    exit()

        ee_pose = self.get_current_end_effector_pose()
        if abs(ee_pose[2] - obj_pose.position.z) > 0.01:
            self.go(ee_pose[0], ee_pose[1], obj_pose.position.z, ee_pose[3], ee_pose[4], ee_pose[5])

        if not self.check_end_effector_reached_desired_target([obj_pose.position.x, obj_pose.position.y, obj_pose.position.z, obj_pose.orientation.x, obj_pose.orientation.y, obj_pose.orientation.z, obj_pose.orientation.w]):
            print("Motion planning failed, could not reach and grasp the object... Please move the object to a better position and try again")
            self.move_to_home_position()
            exit()

        eef_link = self.group.get_end_effector_link()
        grasping_group = "gripper"
        touch_links = self.robot.get_link_names(group=grasping_group)

        aco = AttachedCollisionObject()
        aco.object = obj[name]
        aco.link_name = eef_link
        aco.touch_links = touch_links
        self.scene.attach_object(aco)

    def detach_object_from_gripper(self):
        obj_dict=self.scene.get_attached_objects()
        if len(obj_dict)>0:
            obj = obj_dict[obj_dict.keys()[0]]
            self.scene.remove_attached_object()
            rospy.sleep(1)

            pose = self.get_object_location(obj.object.id)
            offset = pose[2] - obj.object.primitives[0].dimensions[0]/2.0

            if(offset > -0.05):
                self.scene.remove_world_object(name=obj.object.id)


    def setup_the_scene(self):
        self.add_box_to_workspace("workbench_1", x=0.44, y=-0.37, z=-0.475, sx=1.52, sy=0.61, sz=0.715)
        self.add_box_to_workspace("workbench_2", x=-0.03, y=-1.13, z=-0.475, sx=0.585, sy=0.91, sz=0.715)
        self.add_box_to_workspace("inventory_shelf", x=0.88, y=0.26, z=-0.47, sx=1.01, sy=0.63, sz=0.715)
        self.add_box_to_workspace("robot_base", x=0.0, y=0.24, z=-0.42, sx=0.75, sy=0.60, sz=0.825)
        self.add_box_to_workspace("autoclaving_table", x=-0.70, y=-0.30, z=-0.47, sx=0.61, sy=1.52, sz=0.715)
        self.add_box_to_workspace("wall", x=0.5, y=0.65, z=0.1, sx=2.0, sy=0.2, sz=2.0)

    def get_object_location(self,object_name):
        object_name = object_name.lower()
        obj_space = object_name.replace("_"," ")
        tokens = obj_space.split()
        tokens_first = tokens.pop(0)
        tokens.append(tokens_first)
        new_name=" ".join(tokens)
        if self.object_dimensions.has_key(object_name):
            obj_pose = self.scene.get_object_poses([object_name])[object_name]
            eul = self.quaternion_to_euler(obj_pose.orientation.x,obj_pose.orientation.y,obj_pose.orientation.z,obj_pose.orientation.w)
            obj = [obj_pose.position.x,obj_pose.position.y,obj_pose.position.z,eul[0],eul[1],eul[2]]
            return obj
        elif self.object_dimensions.has_key(obj_space):
            obj_pose = self.scene.get_object_poses([obj_space])[obj_space]
            eul = self.quaternion_to_euler(obj_pose.orientation.x,obj_pose.orientation.y,obj_pose.orientation.z,obj_pose.orientation.w)
            obj = [obj_pose.position.x,obj_pose.position.y,obj_pose.position.z,eul[0],eul[1],eul[2]]
            return obj
        elif self.object_dimensions.has_key(new_name):
            obj_pose = self.scene.get_object_poses([new_name])[new_name]
            eul = self.quaternion_to_euler(obj_pose.orientation.x,obj_pose.orientation.y,obj_pose.orientation.z,obj_pose.orientation.w)
            obj = [obj_pose.position.x,obj_pose.position.y,obj_pose.position.z,eul[0],eul[1],eul[2]]
            return obj
        else:
            return None        

    def marker_orientation(self, marker_number):
        if(marker_number<5):
            return [180, 0, -180]
        elif(marker_number >= 5 and marker_number < 8):
            return [180, 0, 0]
        elif(marker_number >= 8 and marker_number < 11):
            return [180, 0, -90]
        else:
            return [-180, 0, 90]

    def pour(self, container_name, tilt = 2.1):
        container_name = container_name.lower()
        obj_space = container_name.replace("_"," ")
        tokens = obj_space.split()
        tokens_first = tokens.pop(0)
        tokens.append(tokens_first)
        new_name=" ".join(tokens)
        if tilt <= 1.57 or tilt >=3.14:
            print("please set the tilt to between 1.57 and 3.14 to perform a pouring action!")
            rospy.logwarn("please set the tilt to between 1.57 and 3.14 to perform a pouring action!")
        else:
            rospy.sleep(2)
            
            container = self.scene.get_object_poses([container_name])
            
            if(container == {}):
                container = self.scene.get_object_poses([obj_space])
                container_name = obj_space
            if(container == {}):
                container = self.scene.get_object_poses([new_name])
                container_name = new_name


            try:
                eulc = self.quaternion_to_euler(container[container_name].orientation.x, container[container_name].orientation.y, container[container_name].orientation.z, container[container_name].orientation.w)
                eul = [180, 0, 0]
                container_pose = [container[container_name].position.x, container[container_name].position.y, container[container_name].position.z, eul[0], eul[1], eul[2]] 
                pre_pour_pose = copy.deepcopy(container_pose)
                before_pour = copy.deepcopy(container_pose)
            except:
                print("container not in scene, please add it into the workspace before pouring into it!")
                rospy.logwarn("container not in scene, please add it into the workspace before pouring into it!")

                exit()
            try:
                objects = self.scene.get_attached_objects()
                obj = objects.values()[0]
                obj_height = obj.object.primitives[0].dimensions[0]
                obj_radius = obj.object.primitives[0].dimensions[1]
            except:
                print("object not attached to gripper, please add it into the workspace and make sure the gripper has grasped it before pouring the content!")
                rospy.logwarn("object not attached to gripper, please add it into the workspace and make sure the gripper has grasped it before pouring the content!")
                exit()


            container_height = self.get_object_height(container_name)
            container_radius = self.get_object_radius(container_name)
            x_offset = np.cos(container_pose[5] * np.pi/180) * (container_radius + obj_height/2.0 * np.cos(tilt - 1.57) )
            y_offset = np.sin(container_pose[5] * np.pi/180) * (container_radius + obj_height/2.0 * np.cos(tilt - 1.57) )
            z_offset = container_pose[2] + container_height/2.0 + obj_height/2.0*np.sin(tilt - 1.57) + obj_radius
            pre_pour_pose[0] -= x_offset
            pre_pour_pose[1] -= y_offset
            pre_pour_pose[2] += z_offset

            if pre_pour_pose[1] > 0:
                if pre_pour_pose[5] < 90:   
                    pre_pour_pose[5] += 90
                else:
                    pre_pour_pose[5] -= 270

            elif pre_pour_pose[1] <= 0:
                if pre_pour_pose[5] > -90:   
                    pre_pour_pose[5] -= 90
                else:
                    pre_pour_pose[5] += 270
            
            theta=0
            xo = container_radius/2.0 + obj_height/2.0 * np.cos(tilt - 1.57)
            # success = self.go(container_pose[0]-xo,container_pose[1],container_pose[2]+ container_height/2.0 + obj_height/2.0*np.sin(tilt - 1.57) + obj_radius*2 + 0.1, 180.0,0.0,-90.0)
            success = self.go(container_pose[0]-xo,container_pose[1],container_pose[2]+ container_height/2.0 + obj_radius*2 + 0.1, 180.0,0.0,-90.0)
            # success = self.go(container_pose[0]-xo,container_pose[1],container_pose[2]+ container_height/2.0 + obj_height/2.0*np.sin(tilt - 1.57) + obj_radius*2 + 0.05, 180.0,0.0,-90.0)
            success = self.go(container_pose[0]-xo,container_pose[1],container_pose[2]+ container_height/2.0 + obj_radius*2 +0.05, 180.0,0.0,-90.0)
            while not success:
                w_T_c = np.zeros([4,4])
                w_T_c[0:3,0:3] = self.quaternion_rotation_matrix([container[container_name].orientation.x, container[container_name].orientation.y, 
                                                                container[container_name].orientation.z, container[container_name].orientation.w])
                w_T_c[0,3] = container_pose[0]
                w_T_c[1,3] = container_pose[1]
                w_T_c[2,3] = container_pose[2]
                w_T_c[3,3] = 1

                c_T_p = np.zeros([4,4])
                c_T_p[0:3,0:3] = self.euler_to_rotation_matrix(0,0,np.deg2rad(-90+theta))
                c_T_p[0,3] = (container_radius + obj_height/2.0) * np.cos(np.deg2rad(theta))
                c_T_p[1,3] = (container_radius + obj_height/2.0) * np.sin(np.deg2rad(theta))
                c_T_p[2,3] = container_height/2.0 + obj_height/2.0 + 0.01
                c_T_p[3,3] = 1
                        
                w_T_p = np.matmul(w_T_c, c_T_p)

                eule=self.rotation_matrix_to_euler(w_T_p[0:3,0:3])
               
                print([w_T_p[0,3],w_T_p[1,3],-w_T_p[2,3],np.rad2deg(eule[0]),np.rad2deg(eule[1]),np.rad2deg(eule[2])])
                print([container_pose[0],container_pose[1],container_pose[2]+ container_height/2.0 + obj_height/2.0 + 0.05, 180.0,0.0,-90.0])

                success = self.go(w_T_p[0,3],w_T_p[1,3],-w_T_p[2,3],np.rad2deg(eule[0]),np.rad2deg(eule[1]),np.rad2deg(eule[2]),pour=True)
                theta+=30
                print(theta)
                if theta > 360:
                    break
            
            print("moved to above prepour pose")

            rospy.sleep(1)
            if success:
                curr_pose = self.group.get_current_joint_values()
                pour_pose = copy.deepcopy(curr_pose)
                pour_pose[-1] = -tilt
                
                self.group.set_max_velocity_scaling_factor(0.3)
                self.group.set_max_acceleration_scaling_factor(0.5)
                result = self.group.go(pour_pose,wait=True)
                self.group.stop()
                rospy.sleep(1)
                if not result:
                    print("Failed pouring, trying again...")
                    result = self.group.go(pour_pose,wait=True)
                    self.group.stop()
                    rospy.sleep(1)
                    print("Failed pouring, aborted")

                else:
                    self.group.set_max_velocity_scaling_factor(0.8)
                    self.group.set_max_acceleration_scaling_factor(0.6)
                    self.group.go(curr_pose,wait=True)
                    self.group.stop()
                    rospy.sleep(1)
                    success = self.go(container_pose[0]-xo,container_pose[1],container_pose[2]+ container_height/2.0 + obj_height/2.0*np.sin(tilt - 1.57) + obj_radius*2 + 0.1, 180.0,0.0,-90.0)

                    print("Completed pouring!")
            else:
                print("Not possible to move near the container. Please move the object to a better position and try again")
                rospy.logwarn("Not possible to move near the container. Please move the object to a better position and try again")
                

    def get_object_name_by_contents(self, target_contents):
        for container_name, container_info in self.object_dimensions.items():
            if target_contents.lower() in container_info['content'] or container_info['content'] in target_contents.lower():
                return container_name
        return None  # Return None if no matching container is found

    def get_object_height(self, object_name):
        try:
            object_name = object_name.lower()
            obj = self.scene.get_objects([object_name])
            if obj is not None:
                return obj[object_name].primitives[0].dimensions[0]

        except:
            print("\n{" + object_name + "} not in scene, please add it into the workspace before trying to get it's height!\n")
            rospy.logwarn("\n{" + object_name + "} not in scene, please add it into the workspace before trying to get it's height!\n")
            
            exit()

    def get_object_radius(self, object_name):
        object_name = object_name.lower()
        obj = self.scene.get_objects([object_name])
        return obj[object_name].primitives[0].dimensions[1]
    
    def matrix_to_quaternion(self, rotation_matrix):
        trace = np.trace(rotation_matrix)
        
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / S
            qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / S
            qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / S
        elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
            S = np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2
            qw = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / S
            qx = 0.25 * S
            qy = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / S
            qz = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / S
        elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
            S = np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2
            qw = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / S
            qx = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / S
            qy = 0.25 * S
            qz = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / S
        else:
            S = np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2
            qw = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / S
            qx = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / S
            qy = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / S
            qz = 0.25 * S
            
        return qx, qy, qz, qw
    
    def quaternion_rotation_matrix(self,Q):
        """
        source: https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix

        Covert a quaternion into a full three-dimensional rotation matrix.
    
        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
    
        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix. 
                This rotation matrix converts a point in the local reference 
                frame to a point in the global reference frame.
        """
        # Extract the values from Q
        q0 = Q[3]
        q1 = Q[0]
        q2 = Q[1]
        q3 = Q[2]
        
        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
        
        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)
        
        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1
        
        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])
        
        return rot_matrix
    
    def euler_to_quaternion(self,roll, pitch, yaw):
        # Convert Euler angles (roll, pitch, yaw) to quaternions
        roll=math.radians(roll)
        pitch=math.radians(pitch)
        yaw=math.radians(yaw)

        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = sy * cp * sr + cy * sp * cr
        z = sy * cp * cr - cy * sp * sr

        return [w, x, y, z]
    
    def quaternion_to_euler(self, x, y, z, w):
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)

        t2 = 2.0 * (w * y - z * x)
        t2 = 1.0 if t2 > 1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)

        return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]
    
    def euler_to_rotation_matrix(self,roll, pitch, yaw):
        """
        Convert Euler angles to a 3x3 rotation matrix.

        Args:
            roll (float): Rotation angle around the X-axis (in radians).
            pitch (float): Rotation angle around the Y-axis (in radians).
            yaw (float): Rotation angle around the Z-axis (in radians).

        Returns:
            np.ndarray: 3x3 rotation matrix representing the given Euler angles.
        """
        # Calculate trigonometric values to avoid redundant calculations
        cos_r, sin_r = np.cos(roll), np.sin(roll)
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)

        # Calculate elements of the rotation matrix
        R_x = np.array([[1, 0, 0],
                        [0, cos_r, -sin_r],
                        [0, sin_r, cos_r]])

        R_y = np.array([[cos_p, 0, sin_p],
                        [0, 1, 0],
                        [-sin_p, 0, cos_p]])

        R_z = np.array([[cos_y, -sin_y, 0],
                        [sin_y, cos_y, 0],
                        [0, 0, 1]])

        # Combine the rotation matrices in the specified order (XYZ)
        rotation_matrix = np.dot(R_z, np.dot(R_y, R_x))

        return rotation_matrix
    
    def rotation_matrix_to_euler(self,matrix):
        """
        Convert a 3x3 rotation matrix to Euler angles (in radians) using XYZ convention.

        Args:
            matrix (np.ndarray): 3x3 rotation matrix.

        Returns:
            tuple: Tuple containing three Euler angles (roll, pitch, yaw) in radians.
        """
        # Extract individual rotation matrix elements
        r11, r12, r13 = matrix[0, 0], matrix[0, 1], matrix[0, 2]
        r21, r22, r23 = matrix[1, 0], matrix[1, 1], matrix[1, 2]
        r31, r32, r33 = matrix[2, 0], matrix[2, 1], matrix[2, 2]

        # Calculate pitch (around Y-axis)
        pitch = np.arcsin(-r31)

        # Check for singularity at +/- 90 degrees pitch
        if np.abs(r31) == 1:
            # Gimbal lock detected
            yaw = 0
            if r31 == -1:
                roll = np.arctan2(r12, r13)
            else:
                roll = -np.arctan2(-r12, -r13)
        else:
            # Calculate roll and yaw using trigonometric relationships
            roll = np.arctan2(r32 / np.cos(pitch), r33 / np.cos(pitch))
            yaw = np.arctan2(r21 / np.cos(pitch), r11 / np.cos(pitch))

        return roll, pitch, yaw

    def __del__(self):
        moveit_commander.roscpp_shutdown()

