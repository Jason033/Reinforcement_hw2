from flask import Flask, render_template, request, jsonify
import random

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    """
    接收前端網格資訊並執行「價值迭代」，最後回傳:
      - policy: dict, 格子 -> 最優動作
      - qValues: dict, 格子 -> 以最優動作計算得到的 Q-value
      - path: list, 依照最優策略由起點到終點的路徑(以index表現)
      - reachable: bool, 是否成功抵達終點
    """
    data = request.get_json()
    gridSize = data['gridSize']
    startPos = data['startPos']
    goalPos = data['goalPos']
    obstacles = data['obstacles']

    # 將障礙物轉為 (row, col) 集合
    obstacle_set = set((o['row'], o['col']) for o in obstacles)

    # 基本設定
    actions = ['U', 'D', 'L', 'R']  # 上、下、左、右
    gamma = 0.9                     # 折扣因子

    startState = (startPos['row'], startPos['col'])
    goalState = (goalPos['row'], goalPos['col'])

    # 建立所有有效的狀態(不含障礙物)
    valid_states = []
    for r in range(gridSize):
        for c in range(gridSize):
            if (r, c) not in obstacle_set:
                valid_states.append((r, c))

    # 判斷是否為有效移動
    def is_valid(r, c):
        return 0 <= r < gridSize and 0 <= c < gridSize and (r, c) not in obstacle_set

    # 根據動作，回傳 (nextState, immediate_reward)
    def step(state, action):
        # 若已在終點，回傳 (同樣狀態, 獎勵0)，可視為吸收狀態
        if state == goalState:
            return state, 0.0
        (r, c) = state
        if action == 'U': r -= 1
        elif action == 'D': r += 1
        elif action == 'L': c -= 1
        elif action == 'R': c += 1

        # 撞牆或障礙物：留在原地，獎勵 -1
        if not is_valid(r, c):
            return (state), -1.0
        # 抵達終點：+1
        if (r, c) == goalState:
            return (r, c), 1.0
        # 一般移動：0
        return (r, c), 0.0

    #--- 1) 初始化狀態價值函數 V(s) ---
    V = {}
    for s in valid_states:
        V[s] = 0.0   # 初始先給 0
    # goal 狀態可先固定為 0（終點的價值可視為 0）

    #--- 2) 進行價值迭代 ---
    threshold = 1e-3
    max_iterations = 1000

    for _ in range(max_iterations):
        delta = 0.0
        new_V = {}
        for s in valid_states:
            # 若是終點，價值維持 0 即可
            if s == goalState:
                new_V[s] = 0.0
                continue

            # 對所有動作計算 R + gamma * V(nextState)
            best_action_value = float('-inf')
            for a in actions:
                nxt, reward = step(s, a)
                candidate_value = reward + gamma * V[nxt]
                if candidate_value > best_action_value:
                    best_action_value = candidate_value

            new_V[s] = best_action_value
            # 追蹤最大變動量，若小於 threshold 則表示收斂
            delta = max(delta, abs(new_V[s] - V[s]))

        V = new_V
        if delta < threshold:
            # 收斂就可提早跳出
            break

    #--- 3) 根據最終 V(s) 反推政策 policy(s) ---
    #     同時也將對應之最優動作 Q-value 儲存
    policy = {}
    qValues = {}
    for s in valid_states:
        if s == goalState:
            # goal 狀態不需要動作，其 Q-value 設為0
            qValues[s] = 0.0
            continue

        best_a = None
        best_value = float('-inf')
        # 從 V(s) 中找出使得價值最大的動作
        for a in actions:
            nxt, reward = step(s, a)
            val = reward + gamma * V[nxt]
            if val > best_value:
                best_value = val
                best_a = a

        policy[s] = best_a
        qValues[s] = best_value

    #--- 4) 將 state (r,c) 轉換為 index，讓前端好對應 ---
    idx_map = {}
    idx_counter = 1
    for r in range(gridSize):
        for c in range(gridSize):
            idx_map[(r, c)] = idx_counter
            idx_counter += 1

    # 將 policy, qValues 用 index 做 key
    policy_idx = {}
    qValues_idx = {}
    for s in valid_states:
        s_idx = idx_map[s]
        if s == goalState:
            # goal 狀態
            qValues_idx[s_idx] = 0.0
        else:
            policy_idx[s_idx] = policy[s]
            qValues_idx[s_idx] = qValues[s]

    #--- 5) 從起點按照 policy 找出路徑 ---
    path = []
    visited = set()
    current = startState
    reachable = False

    def valid_for_path(r, c):
        return 0 <= r < gridSize and 0 <= c < gridSize and (r, c) not in obstacle_set

    while True:
        path.append(idx_map[current])
        if current == goalState:
            reachable = True
            break
        if current in visited:
            # 代表進入了循環，無法到終點
            break
        visited.add(current)

        c_idx = idx_map[current]
        if c_idx not in policy_idx:
            # 無動作可走
            break

        a = policy_idx[c_idx]
        (r, c) = current
        if a == 'U': r -= 1
        elif a == 'D': r += 1
        elif a == 'L': c -= 1
        elif a == 'R': c += 1

        if not valid_for_path(r, c):
            break
        current = (r, c)

    print(f"價值迭代完成，reachable = {reachable}")

    #--- 6) 封裝回傳 ---
    resp = {
        "policy": policy_idx,
        "qValues": qValues_idx,
        "path": path,
        "reachable": reachable
    }
    return jsonify(resp)

if __name__ == '__main__':
    app.run(debug=True)
