import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# 시뮬레이션 환경 설정
class Simulation:
    def __init__(self, rail_length=300):
        self.rail_length = rail_length
        self.robot_position = 0
        self.pollination_detected = False  # 꽃 감지 여부

    def move_robot(self):
        # 로봇이 꽃을 감지했는지 확인
        if not self.pollination_detected:
            # 로봇을 한 칸 이동 (라운드 로빈)
            self.robot_position = (self.robot_position + 1) % self.rail_length

    def perform_pollination(self):
        # 꽃 감지 로직 (예시로 단순하게 처리)
        self.pollination_detected = np.random.rand() < 0.1  # 10% 확률로 꽃 감지

# 원통형 바퀴 생성 함수
def draw_cylinder(ax, position, height=0.1, radius=0.05, color='r'):
    z = np.linspace(-height / 2, height / 2, 10)
    theta = np.linspace(0, 2 * np.pi, 30)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + position
    y_grid = radius * np.sin(theta_grid) + 0.5  # y 위치 조정
    return ax.plot_surface(x_grid, y_grid, z_grid, color=color)

# 애니메이션 업데이트 함수
def update(frame):
    sim.perform_pollination()  # 꽃 감지 수행
    sim.move_robot()          # 로봇을 한 칸 이동

    # 로봇 위치를 업데이트
    robot_body.set_data([sim.robot_position, sim.robot_position + 1], [0.25, 0.25])  # 로봇 본체
    robot_body.set_3d_properties([0, 0])  # z 데이터

    # 바퀴 위치 업데이트 (각 레일에 2개의 바퀴)
    wheel1_position = sim.robot_position - 1  # 앞 왼쪽 바퀴
    wheel2_position = sim.robot_position + 1  # 앞 오른쪽 바퀴
    wheel3_position = sim.robot_position - 1  # 뒤 왼쪽 바퀴
    wheel4_position = sim.robot_position + 1  # 뒤 오른쪽 바퀴

    # 바퀴를 원통형으로 그리기
    ax.collections.clear()  # 모든 그래픽 객체를 초기화
    draw_cylinder(ax, wheel1_position)  # 앞 왼쪽 바퀴
    draw_cylinder(ax, wheel2_position)  # 앞 오른쪽 바퀴
    draw_cylinder(ax, wheel3_position)  # 뒤 왼쪽 바퀴
    draw_cylinder(ax, wheel4_position)  # 뒤 오른쪽 바퀴

    # 꽃 감지 상태에 따라 로봇 멈추기
    if sim.pollination_detected:
        robot_body.set_markerfacecolor('orange')  # 로봇 본체 색상 변경
        status_text.set_text(f"Position: {sim.robot_position}, Pollination Detected!")
    else:
        robot_body.set_markerfacecolor('blue')  # 로봇 본체 색상 변경
        status_text.set_text(f"Position: {sim.robot_position}, No Pollination Detected.")

    return robot_body, status_text

# 시뮬레이션 객체 생성
sim = Simulation()

# 애니메이션 설정
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, sim.rail_length)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

# 레일 그리기 (두 개의 평행 레일)
ax.plot([0, sim.rail_length], [0.5, 0.5], [0, 0], color='black', lw=3)  # 첫 번째 레일
ax.plot([0, sim.rail_length], [0, 0], [0, 0], color='black', lw=3)  # 두 번째 레일

# 로봇 본체와 바퀴 마커 생성 (상자형 로봇)
robot_body, = ax.plot([], [], [], 'bs', markersize=10)  # 로봇 본체
status_text = ax.text(0, 0, 0.5, '', fontsize=12)  # 상태 텍스트

# 축 제거
ax.axis('off')

# 애니메이션 생성
ani = animation.FuncAnimation(fig, update, frames=100, interval=100)

plt.show()

