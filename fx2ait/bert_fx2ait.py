import time
import torch
import transformers
import logging
import copy
from fx2ait.lower.lower import AitLowerer
from fx2ait.lower.lower_settings import LowerSettings
torch._logging.set_logs(dynamo=logging.INFO,inductor=logging.INFO)

batch = 1
def ait_backend(gm, example_inputs):
    lower = AitLowerer.create(
        LowerSettings(workdir="/tmp", name="test_ait_lower", min_acc_module_size=0)
    )
    lowered = lower(gm, example_inputs)
    return lowered

device = torch.device("cuda:0")
model = transformers.BertForMaskedLM.from_pretrained("bert-base-uncased")
config = model.config
config.num_hidden_layers = 1
model = transformers.BertForMaskedLM(config)

model.eval()
model.cuda()
model.half()
ait_model = copy.deepcopy(model)

ins = {'input_ids': torch.randint(0, 10, size=(batch, 512)).to(device), 'attention_mask': torch.ones(batch, 512, dtype=torch.int64).to(device)}

# ait_backend = "inductor"
ait_model = torch.compile(ait_model, backend=ait_backend)
ref_model = torch.compile(model, backend="inductor")
for _ in range(10):
    res = ait_model(**ins)

t0 = time.time()
for _ in range(1000):
    res = ait_model(**ins)
t1 = time.time()
print(t1-t0)

for _ in range(10):
    ref_res = ref_model(**ins)

t0 = time.time()
for _ in range(1000):
    ref_res = ref_model(**ins)
t1 = time.time()
print(t1-t0)
torch.testing.assert_close(res, ref_res, rtol=1e-2, atol=1e-2)
print(torch.max(res.logits), torch.max(ref_res.logits))
