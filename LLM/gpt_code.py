import rospy
from FunctionLib import FunctionLib

# Initialize ROS node
rospy.init_node("gpt")

# Initialize function library
lib = FunctionLib()

# Helper function to move to a new position based on an offset and print a message
def move_and_print(offset_x, offset_y, offset_z, message):
    # Get the current end effector pose
    current_pose = lib.get_current_end_effector_pose()
    
    # Apply the offsets
    new_x = current_pose[0] + offset_x
    new_y = current_pose[1] + offset_y
    new_z = current_pose[2] + offset_z

    # Move to the new position
    lib.go(new_x, new_y, new_z, current_pose[3], current_pose[4], current_pose[5])

    # Check if end effector reached the target
    target = [new_x, new_y, new_z]
    if lib.check_end_effector_reached_desired_target(target):
        print(message)

# Move 5 cm in each direction and print "arrived"
move_and_print(0.05, 0, 0, "Arrived at +X direction")
move_and_print(-0.05, 0, 0, "Back to original X position")

move_and_print(0, 0.05, 0, "Arrived at +Y direction")
move_and_print(0, -0.05, 0, "Back to original Y position")

move_and_print(0, 0, 0.05, "Arrived at +Z direction")
move_and_print(0, 0, -0.05, "Back to original Z position")

# Move the robot back to the home position
lib.move_to_home_position()
