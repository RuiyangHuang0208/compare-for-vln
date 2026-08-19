# InternVLA-N1 Model Series

![License](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-9cf?style=flat)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)

---

## Model Description
InternVLA-N1 is a state-of-the-art navigation foundation model built on a **multi-system design**. Within this framework, it introduces a **dual-system approach** that joint trains the **System 2** for high-level reasoning and **System 1** for low-level action and control. This asynchronous architecture enables smooth, efficient, and robust instruction-following navigation in both simulated and real-world environments.


---

### 🔗 Resources

[![Code](https://img.shields.io/badge/GitHub-InternNav-181717?logo=github)](https://github.com/InternRobotics/InternNav)
[![Technical Report — InternVLA-N1](https://img.shields.io/badge/Technical_Report-InternVLA--N1-BB2649?logo=adobeacrobatreader&logoColor=white)](https://internrobotics.github.io/internvla-n1.github.io/static/pdfs/InternVLA_N1.pdf)
[![DualVLN Paper — arXiv](https://img.shields.io/badge/arXiv-DualVLN-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.08186)
[![Project Page — InternVLA-N1](https://img.shields.io/badge/Project_Page-InternVLA--N1-4285F4?logo=google-chrome&logoColor=white)](https://internrobotics.github.io/internvla-n1.github.io/)
[![Project Page — DualVLN](https://img.shields.io/badge/Project_Page-DualVLN-4285F4?logo=google-chrome&logoColor=white)](https://internrobotics.github.io/internvla-n1-dualvln.github.io/)
[![Dataset](https://img.shields.io/badge/Dataset-InternData--N1-FF6F00?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/InternRobotics/InternData-N1)

---

## Key Features

- 🧩 **Modular Multi-System Support**  
  Combines **System 2** (reasoning/planning) with **System 1** (action/control) in an asynchronous framework, delivering the first **Dual-System Vision-Language Navigation (VLN) Foundation Model**.

- 🚀 **Zero-Shot Sim2Real Generalization**  
  Trained exclusively on simulation data (**InternData-N1**) while generalizing effectively to real-world deployments.

- 🏆 **State-of-the-Art Performance**  
  Achieves leading results on multiple VLN benchmarks, including **VLN-CE R2R/RxR** and **VLN-PE**.

- ⚡ **Asynchronous Inference**  
  Enables smooth execution and dynamic obstacle avoidance during navigation.


---

## Model Variants

| Model Variant | Description | Key Characteristics |
|--------------|-------------|----------------------|
| [**InternVLA-N1 (S2)**](https://huggingface.co/InternRobotics/InternVLA-N1-System2) | Finetuned Qwen2.5-VL model for pixel-goal grounding | Strong System 2 module; compatible with decoupled System 1 controllers or joint optimization pipelines |
| [**InternVLA-N1 (Dual System) _w/ NavDP\*_**](https://huggingface.co/InternRobotics/InternVLA-N1-w-NavDP) | Jointly tuned System 1 (NavDP*) and InternVLA-N1 (S2) | Optimized end-to-end performance; uses RGB-D observations |
| [**InternVLA-N1 (Dual System) _DualVLN_**](https://huggingface.co/InternRobotics/InternVLA-N1-DualVLN) | Latest dual-system architecture | Optimized end-to-end performance and faster convergence; uses RGB observations |




> The previously released version is now called [InternVLA-N1-wo-dagger](https://huggingface.co/InternRobotics/InternVLA-N1-wo-dagger). The lastest official release is recommended for best performance.

---

## Usage
For inference, evaluation, and the Gradio demo, please refer to the [InternNav repository](https://github.com/InternRobotics/InternNav).

---

## Citation
If you find our work helpful, please consider starring this repository 🌟 and citing:

```bibtex
@misc{internvla-n1,
    title = {{InternVLA-N1: An} Open Dual-System Navigation Foundation Model with Learned Latent Plans},
    author = {InternVLA-N1 Team},
    year = {2025},
    booktitle={arXiv},
}
@misc{internnav2025,
    title = {{InternNav: InternRobotics'} open platform for building generalized navigation foundation models},
    author = {InternNav Contributors},
    howpublished={\url{https://github.com/InternRobotics/InternNav}},
    year = {2025}
}
@misc{wei2025groundslowfastdualsystem,
      title={Ground Slow, Move Fast: A Dual-System Foundation Model for Generalizable Vision-and-Language Navigation}, 
      author={Meng Wei and Chenyang Wan and Jiaqi Peng and Xiqian Yu and Yuqiang Yang and Delin Feng and Wenzhe Cai and Chenming Zhu and Tai Wang and Jiangmiao Pang and Xihui Liu},
      year={2025},
      eprint={2512.08186},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2512.08186}, 
}


