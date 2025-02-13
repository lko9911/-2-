from vpython import *

# 레일 생성
rail_length = 10
rail = cylinder(pos=vector(0, 0, 0), axis=vector(rail_length, 0, 0), radius=0.05, color=color.gray(0.5))

# 로봇 생성
robot = box(pos=vector(0, 0.1, 0), size=vector(0.5, 0.2, 0.3), color=color.blue)

# 바퀴 생성
wheel1 = cylinder(pos=vector(-0.25, 0, -0.15), axis=vector(0, 0, 0.2), radius=0.05, color=color.black)
wheel2 = cylinder(pos=vector(0.25, 0, -0.15), axis=vector(0, 0, 0.2), radius=0.05, color=color.black)

flower = sphere(pos=vector(7, 0, 0), radius=0.1, color=color.yellow)

# 시뮬레이션 루프 수정
while True:
    rate(30)  # FPS 설정
    if robot.pos.x < 7:  # 꽃을 감지할 때
        robot.pos.x += 0.1  # 계속 이동
    else:
        # 꽃 감지 시 멈춤
        robot.pos.x = robot.pos.x  # 멈춤

