import numpy as np

def slicing_window(data, window_size, horizon):
    """
    Creates input and output sequences for long-term forecasting.
    """
    inputs = []
    outputs = []
    for i in range(len(data) - window_size - horizon + 1):
        inputs.append(data[i:i+window_size])
        outputs.append(data[i+window_size:i+window_size+horizon])
    return np.array(inputs), np.array(outputs)
    
# 예시 데이터
data = np.random.randn(500)

# 인풋 시계열 데이터와 레이블 생성
window_size = 100
horizon = 10
x, y = create_longterm_input_output(data, window_size, horizon)
