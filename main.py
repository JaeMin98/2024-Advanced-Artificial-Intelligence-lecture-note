import os
import subprocess

scripts_and_dirs = [
    (os.path.join('DQN_LunarLander','d_dqn_train.py'), os.path.join('DQN_LunarLander')),
    (os.path.join('Double_DQN_LunarLander','d_dqn_train.py'), os.path.join('Double_DQN_LunarLander')),
    (os.path.join('DQN_LunarLander','d_dqn_train.py'), os.path.join('DQN_LunarLander')),
]

# 상대 경로를 절대 경로로 변환
absolute_scripts_and_dirs = [(os.path.abspath(script), os.path.abspath(directory)) for script, directory in scripts_and_dirs]

for script_path, working_dir in absolute_scripts_and_dirs:
    os.chdir(working_dir)  # 작업 디렉토리 변경
    subprocess.run(["python", script_path])  # 스크립트 실행
