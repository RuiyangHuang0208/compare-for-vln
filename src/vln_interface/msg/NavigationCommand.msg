# Unified command produced by every VLN/VLA adapter.
uint8 STOP=0
uint8 WAYPOINT=1
uint8 TRAJECTORY=2
uint8 VELOCITY=3

std_msgs/Header header
uint8 command_type

# WAYPOINT uses one or more ordered target poses.
geometry_msgs/PoseStamped[] waypoints

# TRAJECTORY uses an ordered path in header.frame_id.
nav_msgs/Path trajectory

# VELOCITY maps linear.x, linear.y, angular.z to vx, vy, yaw_rate.
geometry_msgs/Twist velocity
