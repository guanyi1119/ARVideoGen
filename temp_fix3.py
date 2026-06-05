import re

with open('methods/reward_forcing/videoalign/wan_inference.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace disable_flash_attn2=False with dynamic check
old_line = 'disable_flash_attn2=False,'
new_line = 'disable_flash_attn2=(os.environ.get("DEVICE_TYPE", "cuda") == "npu"),'
content = content.replace(old_line, new_line)

with open('methods/reward_forcing/videoalign/wan_inference.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done')
