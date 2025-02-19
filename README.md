<div align="center">
<h2>Magma: A Foundation Model for Multimodal AI Agents</h2>

[Jianwei Yang](https://jwyang.github.io/)<sup>*</sup><sup>1</sup><sup>†</sup>&nbsp;
[Reuben Tan](https://cs-people.bu.edu/rxtan/)<sup>1</sup><sup>†</sup>&nbsp;
[Qianhui Wu](https://qianhuiwu.github.io/)<sup>1</sup><sup>†</sup>&nbsp;
[Ruijie Zheng](https://ruijiezheng.com/)<sup>2</sup><sup>‡</sup>&nbsp;
[Baolin Peng](https://scholar.google.com/citations?user=u1CNjgwAAAAJ&hl=en&oi=ao)<sup>1</sup><sup>‡</sup>&nbsp;
[Yongyuan Liang](https://cheryyunl.github.io)<sup>2</sup><sup>‡</sup>

[Yu Gu](https://users.umiacs.umd.edu/~hal/)<sup>1</sup>&nbsp;
[Mu Cai](https://pages.cs.wisc.edu/~mucai/)<sup>3</sup>&nbsp;
[Seonghyeon Ye](https://seonghyeonye.github.io/)<sup>4</sup>&nbsp;
[Joel Jang](https://joeljang.github.io/)<sup>5</sup>&nbsp;
[Yuquan Deng](https://scholar.google.com/citations?user=LTC0Q6YAAAAJ&hl=en)<sup>5</sup>&nbsp;
[Lar Liden](https://sites.google.com/site/larsliden)<sup>1</sup>&nbsp;
[Jianfeng Gao](https://www.microsoft.com/en-us/research/people/jfgao/)<sup>1</sup><sup>▽</sup>

<sup>1</sup> Microsoft Research; <sup>2</sup> University of Maryland; <sup>3</sup> University of Wisconsin-Madison  
<sup>4</sup> KAIST; <sup>5</sup> University of Washington

<sup>*</sup> Project lead  <sup>†</sup> First authors  <sup>‡</sup> Second authors  <sup>▽</sup> Leadership  

\[[arXiv Paper](https://www.arxiv.org/pdf/2502.13130)\] &nbsp; \[[Project Page](https://effective-robot-9p5jy3n.pages.github.io/)\] &nbsp; \[[Model Coming Soon!](https://github.com/microsoft/Magma)\] &nbsp; 

</div>

<div align="center">
<img src="assets/images/magma_teaser.png?raw=true" width="100%">
</div>

## :fire: News
* **[2025.02.18]**  Our flagship Project Magma at MSR is released on [arXiv](https://www.arxiv.org/pdf/2502.13130)!

## What is Magma?

<div align="center">
<img src="assets/images/magma_intro_fig.png?raw=true" width="50%">
</div>

**Magma is a foundation model for multimodal AI agents**. As the basis for agentic models, it should possesse strong capabilities for both perceiving the multimodal world AND precisely taking goal-driven actions (see above figure). With this in mind, we are striving for the following goals:

* **Verbal and spatial-temporal intelligence:** Magma is supposed to have both strong verbal and spatial-temporal intelligence to understand images and videos, ground its actions on the observations, and further translate the external goal into action plan and executions.
* **Digial and physical world:** Magma should not be limited to either the digital world (e.g., web navigation) or the physical world (e.g., robotics manipulation), but rather be able to work across both worlds, just like humans ourselves.

It a large-scale vision-language-action model that can generate text, visual plans, and robotic actions based on the input text and images. Magma is trained on a large-scale dataset, and it achieves state-of-the-art performance on various multimodal tasks, including generic image and video understanding, UI navigation, and robotics manipulation. Magma is designed to be a general-purpose model for multimodal AI agents, and it can be easily adapted to different tasks and domains. It is also designed to be efficient and scalable, making it suitable for real-world applications.

## :sparkles: Highlights
* **Multimodal Understanding:** Magma has strong verbal and spatial intelligence to understand images and videos, making it suitable for a wide range of multimodal tasks.
* **General Agentic Capabilities:** Magma can generate visual plans and robotic actions, making it suitable for robotics and agentic tasks.
* **State-of-the-art Performance:** Magma achieves state-of-the-art performance on various multimodal tasks, including image captioning, video captioning, UI navigation, and robotics manipulation.
* **Efficient and Scalable:** Magma is designed to be efficient and scalable, making it suitable for real-world applications.

## How we pretrain Magma?

<div align="center">
<img src="assets/images/magma_pt_v3.png?raw=true" width="100%">
</div>

## :bookmark_tabs: Todos
We will be releasing all the following contents:
- [ ] Demo code
- [ ] Model checkpoint
- [ ] Comprehensive user guide
- [ ] Pretraining and downstream finetuning code
- [ ] Pretraining data and downstream finetuning data
- [ ] Evaluation code

## Model Usage

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

This model is intended for broad research use in English. The model take images and text as inputs, and produces the textual outputs for the following uses:

* **Image/Video-Conditoned Text Generation:** The model can generate text (e.g., descriptions, answers) based on the input text and image.

* **Visual Planning Capabilities:** The model can also produce the visual trace as the future planning to accomplish a task (e.g., move object from one place to another).

* **Agentic Capabilities:** The model can also generate UI grounding (e.g., click ``search'' button) and robotics manipulations (e.g., 7 DoF for the robot gripper).

Our model is designed only for research purpose and aimed at knowledge-sharing and accelerating research in multimodal AI, in particularly the mutimodal agentic AI.

### Downstream Use

The model can be further finetuned for different downstream tasks, such as:

* **Image Captioning and QA:** We can further finetune this model for image captioning and QA tasks under the pipeline of multimodal LLMs. Based on our experiments, the model can achieve competitive performance yet better spatial understanding and reasoning on these tasks.

* **Video Captioning and QA:** We can further finetune this model for video captioning and QA tasks under the pipeline of multimodal LLMs. Based on our experiments, the model can achieve competitive performance yet better temporal understanding and reasoning on these tasks.

* **UI Navigation:** We can finetune this model for specific UI navigation tasks, such as web navigation or mobile navigation. The model can achieve superior performance on these tasks.

* **Robotics Manipulation:** Our model can be further finetuned for robotics tasks given its general agentic capabilities as a vision-language-action model. After finetuning, our model significantly outperms the state-of-the-art models such as OpenVLA on robotics manipulation tasks.

## Bias, Risks, and Limitations

Please note that this model is not specifically designed or evaluated for all downstream purposes. Developers should consider common limitations of language models as they select use cases, and evaluate and mitigate for accuracy, safety, and fairness before using within a specific downstream use case, particularly for high-risk scenarios. Developers should be aware of and adhere to applicable laws or regulations (including privacy, trade compliance laws, etc.) that are relevant to their use case.


## Citation
If you use this model in your research, please consider citing:

```bibtex
@misc{yang2025magmafoundationmodelmultimodal,
      title={Magma: A Foundation Model for Multimodal AI Agents}, 
      author={Jianwei Yang and Reuben Tan and Qianhui Wu and Ruijie Zheng and Baolin Peng and Yongyuan Liang and Yu Gu and Mu Cai and Seonghyeon Ye and Joel Jang and Yuquan Deng and Lars Liden and Jianfeng Gao},
      year={2025},
      eprint={2502.13130},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.13130}, 
}
```

## Acknowledgements

Our work is supported by Microsoft Research. We thank all the contributors for their efforts in building this project. 

Our work is built on top of some amazing open-source projects, including [Transformers](https://github.com/huggingface/transformers), [LLaVA](https://github.com/haotian-liu/LLaVA), [OpenVLA](https://github.com/openvla/openvla), [SeeClick](https://github.com/njucckevin/SeeClick), [Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web), and also a number of awesome open-source datasets, including [Ego4d](https://ego4d-data.org/), [Epic-Kitchen](https://epic-kitchens.github.io/2025), [Something-Somethingv2](https://www.qualcomm.com/developer/artificial-intelligence/datasets), [Open-X-Embodiment](https://robotics-transformer-x.github.io/), and a number of evaluation benchmarks, including [SimplerEnv](https://github.com/simpler-env/SimplerEnv), [Libero](https://github.com/Lifelong-Robot-Learning/LIBERO).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

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
