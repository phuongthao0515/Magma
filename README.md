# Magma: Multimodal Agentic Models

Magma: A multimodal agentic foundation for multimodal understanding and agentic tasks.

## Introduction

Magma is a multimodal agentic AI model that can generate text based on the input text and image. The model is designed for research purposes and aimed at knowledge-sharing and accelerating research in multimodal AI, in particular the multimodal agentic AI. The main innovation of this model lies on the introduction of two technical innovations: Set-of-Mark and Trace-of-Mark, and the leverage of a large-amount of unlabeled video data to learn the spatial-temporal grounding and planning. Please refer to our paper for more technical details. The model is developed by Microsoft and is funded by Microsoft Research. 

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

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

<!-- {{ downstream_use | default("[More Information Needed]", true)}} -->

<!-- ### Out-of-Scope Use -->

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

<!-- {{ out_of_scope_use | default("[More Information Needed]", true)}} -->

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
