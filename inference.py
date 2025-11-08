import torch
import time

model.eval()
model.cuda()

dummy_input = torch.randn(1, 3, 640, 640).cuda()  # 替换为你的真实图像大小
warmup = 10
repeat = 50

# 预热
for _ in range(warmup):
    with torch.no_grad():
        _ = model(dummy_input)

# 正式计时
torch.cuda.synchronize()
start = time.time()
for _ in range(repeat):
    with torch.no_grad():
        _ = model(dummy_input)
torch.cuda.synchronize()
end = time.time()

avg_time = (end - start) / repeat
print(f"Average inference time per image: {avg_time * 1000:.2f} ms, FPS: {1 / avg_time:.2f}")
