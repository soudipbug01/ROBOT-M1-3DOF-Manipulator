import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==========================================
# 1. ROBOT PARAMETERS (Standard D-H)
# ==========================================
D1 = 0.15  # Base height (m)
A2 = 0.25  # Upper arm length (m)
A3 = 0.20  # Forearm length (m)

# ==========================================
# 2. URDF GENERATOR (For GitHub / BSERC)
# ==========================================
URDF_CONTENT = f"""<?xml version="1.0"?>
<robot name="robot_m1_3dof">
  <link name="base_link"><visual><origin xyz="0 0 {D1/2}"/><geometry><cylinder radius="0.06" length="{D1}"/></geometry></visual></link>
  <link name="shoulder_hub"><visual><origin rpy="1.5708 0 0"/><geometry><cylinder radius="0.04" length="0.06"/></geometry></visual></link>
  <joint name="joint_1" type="revolute"><parent link="base_link"/><child link="shoulder_hub"/><origin xyz="0 0 {D1}"/><axis xyz="0 0 1"/><limit lower="-3.1416" upper="3.1416" effort="50" velocity="3.0"/></joint>
  <link name="link_2"><visual><origin xyz="{A2/2} 0 0"/><geometry><box size="{A2} 0.035 0.035"/></geometry></visual></link>
  <joint name="joint_2" type="revolute"><parent link="shoulder_hub"/><child link="link_2"/><origin xyz="0 0 0"/><axis xyz="0 -1 0"/><limit lower="-1.5708" upper="2.356" effort="50" velocity="3.0"/></joint>
  <link name="link_3"><visual><origin xyz="{A3/2} 0 0"/><geometry><box size="{A3} 0.028 0.028"/></geometry></visual></link>
  <joint name="joint_3" type="revolute"><parent link="link_2"/><child link="link_3"/><origin xyz="{A2} 0 0"/><axis xyz="0 -1 0"/><limit lower="-2.6" upper="2.6" effort="50" velocity="3.0"/></joint>
  <link name="end_effector"><visual><geometry><sphere radius="0.022"/></geometry></visual></link>
  <joint name="ee_fixed" type="fixed"><parent link="link_3"/><child link="end_effector"/><origin xyz="{A3} 0 0"/></joint>
</robot>
"""
with open("robot_m1.urdf", "w") as f:
    f.write(URDF_CONTENT)

# ==========================================
# 3. DENAVIT-HARTENBERG FORWARD KINEMATICS
# ==========================================
def dh_matrix(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,        sa,       ca,      d],
        [0,         0,        0,      1]
    ])

def forward_kinematics_all_joints(q):
    t1, t2, t3 = q
    T0 = np.eye(4)
    T1 = dh_matrix(t1, D1, 0.0, np.pi / 2)
    T2 = T1 @ dh_matrix(t2, 0.0, A2, 0.0)
    T3 = T2 @ dh_matrix(t3, 0.0, A3, 0.0)
    pts = np.array([T0[:3, 3], T1[:3, 3], T2[:3, 3], T3[:3, 3]])
    return pts, T3

# ==========================================
# 4. ANALYTICAL INVERSE KINEMATICS
# ==========================================
def inverse_kinematics(target):
    x, y, z = target
    theta1 = math.atan2(y, x)
    r = math.hypot(x, y)
    s = z - D1
    cos_t3 = (r**2 + s**2 - A2**2 - A3**2) / (2 * A2 * A3)
    if abs(cos_t3) > 1.0:
        raise ValueError(f"Target {target} is out of workspace!")
    theta3 = math.atan2(-math.sqrt(max(0.0, 1.0 - cos_t3**2)), cos_t3)
    theta2 = math.atan2(s, r) - math.atan2(A3 * math.sin(theta3), A2 + A3 * math.cos(theta3))
    return np.array([theta1, theta2, theta3])

# ==========================================
# 5. CUBIC POLYNOMIAL TRAJECTORY PLANNER
# ==========================================
def cubic_trajectory(q_start, q_end, duration, fps=30):
    steps = int(duration * fps)
    t = np.linspace(0, duration, steps)
    tau = t / duration
    s = 3 * tau**2 - 2 * tau**3
    s_dot = (6 * tau - 6 * tau**2) / duration
    q_traj = q_start + np.outer(s, (q_end - q_start))
    qd_traj = np.outer(s_dot, (q_end - q_start))
    return t, q_traj, qd_traj

# ==========================================
# 6. WAYPOINTS & KINEMATIC VERIFICATION
# ==========================================
waypoints = [
    ("Home",           [0.25,  0.00, 0.35], 1.5, False),
    ("Pick Approach",  [0.28, -0.18, 0.18], 2.0, False),
    ("Pick Grasp",     [0.28, -0.18, 0.05], 1.5, True),
    ("Lift Payload",   [0.28, -0.18, 0.22], 1.5, True),
    ("Place Approach", [0.28,  0.18, 0.22], 2.5, True),
    ("Place Release",  [0.28,  0.18, 0.05], 1.5, False),
    ("Retract",        [0.28,  0.18, 0.22], 1.5, False),
    ("Return Home",    [0.25,  0.00, 0.35], 2.0, False),
]

print("\n--- D-H FORWARD & INVERSE KINEMATICS VERIFICATION ---")
q_waypoints = []
for name, pos, _, _ in waypoints:
    q = inverse_kinematics(pos)
    pts, _ = forward_kinematics_all_joints(q)
    err_mm = np.linalg.norm(np.array(pos) - pts[-1]) * 1000.0
    q_waypoints.append(q)
    print(f"{name:<15} | Target: {pos} | IK (deg): {np.round(np.degrees(q), 2)} | Error: {err_mm:.6f} mm")

full_t, full_q, full_qd, grasp_flags, phase_labels = [], [], [], [], []
t_offset = 0.0
for i in range(len(waypoints) - 1):
    t_seg, q_seg, qd_seg = cubic_trajectory(q_waypoints[i], q_waypoints[i+1], waypoints[i+1][2])
    full_t.append(t_seg + t_offset)
    full_q.append(q_seg)
    full_qd.append(qd_seg)
    grasp_flags.extend([waypoints[i][3]] * len(t_seg))
    phase_labels.extend([f"{waypoints[i][0]} -> {waypoints[i+1][0]}"] * len(t_seg))
    t_offset += waypoints[i+1][2]

full_t = np.concatenate(full_t)
full_q = np.vstack(full_q)
full_qd = np.vstack(full_qd)

# Save Trajectory Plot for Technical Report
fig_plot, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
for j in range(3):
    ax1.plot(full_t, np.degrees(full_q[:, j]), label=f"Joint {j+1}")
    ax2.plot(full_t, np.degrees(full_qd[:, j]), label=f"Joint {j+1}")
ax1.set_ylabel("Joint Angle (deg)")
ax1.set_title("ROBOT-M1: Cubic Polynomial Trajectory Profiles (Pick & Place)")
ax1.grid(True); ax1.legend()
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Joint Velocity (deg/s)")
ax2.grid(True); ax2.legend()
plt.tight_layout()
plt.savefig("trajectory_profiles.png", dpi=200)
plt.close(fig_plot)
print("\nSaved 'trajectory_profiles.png' and 'robot_m1.urdf'.")

# ==========================================
# 7. 3D AUTONOMOUS PICK & PLACE ANIMATION
# ==========================================
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-0.1, 0.45)
ax.set_ylim(-0.30, 0.30)
ax.set_zlim(0.0, 0.45)
ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
ax.view_init(elev=25, azim=35)

# Static Pick & Place pads
ax.scatter([0.28], [-0.18], [0.0], color='green', s=120, marker='s', label='Pick Zone')
ax.scatter([0.28], [0.18], [0.0], color='blue', s=120, marker='s', label='Place Zone')

arm_line, = ax.plot([], [], [], 'o-', lw=6, markersize=8, color='#1f77b4', label='3-DOF Manipulator')
trail_line, = ax.plot([], [], [], '--', lw=1.5, color='gray', alpha=0.6, label='End-Effector Path')
payload_dot, = ax.plot([], [], [], 's', markersize=14, color='#d97706', label='Payload')
status_text = ax.set_title("ROBOT-M1 Autonomous Simulation")
ax.legend(loc='upper left')

trail_x, trail_y, trail_z = [], [], []
payload_pos = np.array([0.28, -0.18, 0.03])

def update(frame):
    global payload_pos
    pts, _ = forward_kinematics_all_joints(full_q[frame])
    arm_line.set_data(pts[:, 0], pts[:, 1])
    arm_line.set_3d_properties(pts[:, 2])

    ee = pts[-1]
    trail_x.append(ee[0]); trail_y.append(ee[1]); trail_z.append(ee[2])
    trail_line.set_data(trail_x, trail_y)
    trail_line.set_3d_properties(trail_z)

    if grasp_flags[frame]:
        payload_pos = ee
    payload_dot.set_data([payload_pos[0]], [payload_pos[1]])
    payload_dot.set_3d_properties([payload_pos[2]])

    status_text.set_text(f"ROBOT-M1 Pick & Place | {phase_labels[frame]} | t = {full_t[frame]:.1f}s")
    return arm_line, trail_line, payload_dot, status_text

ani = FuncAnimation(fig, update, frames=len(full_t), interval=33, repeat=True)
plt.show()