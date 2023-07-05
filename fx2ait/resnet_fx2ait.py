import time
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn

from fx2ait.lower.lower import AitLowerer
from fx2ait.lower.lower_settings import LowerSettings

model = torchvision.models.resnet18()
model.eval()
model.half().cuda()
inputs = [torch.randn(1, 3, 224, 224).half().cuda()]
ref_output = model(*inputs)
lower = AitLowerer.create(
    LowerSettings(workdir="/tmp", name="test_ait_lower", min_acc_module_size=0)
)
lowered = lower(model, inputs)
for _ in range(10):
    lower_output = lowered(*inputs)
t0 = time.time()
for _ in range(1000):
    lower_output = lowered(*inputs)
t1 = time.time()
print(t1-t0)

model = torch.compile(model)
for _ in range(10):
    output = model(*inputs)
t0 = time.time()
for _ in range(1000):
    output = model(*inputs)
t1 = time.time()
print(t1-t0)

torch.testing.assert_close(ref_output, lower_output, check_dtype=False)
