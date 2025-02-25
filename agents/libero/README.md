# Magma: Multimodal Agentic Models

Magma: A multimodal agentic foundation for multimodal understanding and agentic tasks.


#### LIBERO Setup
Clone and install LIBERO and other requirements:
```
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -r libero_quick_start/requirements.txt
cd LIBERO
pip install -e .
```

#### Quick Evaluation
The following code demonstrates how to run Magma on a single LIBERO task and evaluate its performance:
```
import numpy as np
from libero.libero import benchmark
from libero_env_utils import get_libero_env, get_libero_dummy_action, get_libero_obs, get_max_steps, save_rollout_video
from libero_magma_utils import get_magma_model, get_magma_prompt, get_magma_action

# Set up benchmark and task
benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_goal" # or libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]()
task_id = 1
task = task_suite.get_task(task_id)

# Initialize environment
env, task_description = get_libero_env(task, resolution=256)
print(f"Task {task_id} description: {task_description}")

# Load MAGMA model
model_name = "microsoft/magma-8b-libero-goal"  # or your local path
processor, magma = get_magma_model(model_name)
prompt = get_magma_prompt(task_description)

# Run evaluation
num_steps_wait = 10
max_steps = get_max_steps(task_suite_name)

env.seed(0)
obs = env.reset()
init_states = task_suite.get_task_init_states(task_id) 
obs = env.set_init_state(init_states[0])

step = 0
replay_images = []
while step < max_steps + num_steps_wait:
    if step < num_steps_wait:
        obs, _, done, _ = env.step(get_libero_dummy_action())
        step += 1
        continue
    
    img = get_libero_obs(obs, resize_size=256)
    replay_images.append(img)
    action = get_magma_action(magma, processor, img, prompt, task_suite_name)
    obs, _, done, _ = env.step(action.tolist())
    step += 1

env.close()
save_rollout_video(replay_images, success=done, task_description=task_description)
```
**Notes:** The above script only tests one episode on a single task and visualizes MAGMA's trajectory with saved video. For comprehensive evaluation on each task suite, please use `eval_magma_libero.py`.
```
python eval_magma_libero.py \
  --model_name microsoft/magma-8b-libero-object \
  --task_suite_name libero_object \

python eval_magma_libero.py \
  --model_name microsoft/magma-8b-libero-spatial \
  --task_suite_name libero_spatial \

python eval_magma_libero.py \
  --model_name microsoft/magma-8b-libero-goal \
  --task_suite_name libero_goal \
```

## Bias, Risks, and Limitations

Please note that this model is not specifically designed or evaluated for all downstream purposes. Developers should consider common limitations of language models as they select use cases, and evaluate and mitigate for accuracy, safety, and fairness before using within a specific downstream use case, particularly for high-risk scenarios. Developers should be aware of and adhere to applicable laws or regulations (including privacy, trade compliance laws, etc.) that are relevant to their use case.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
