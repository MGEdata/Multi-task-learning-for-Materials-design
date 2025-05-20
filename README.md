# Design of superalloys with multiple properties via multi-task learning
The development of new materials requires the collaborative design of multiple properties, requiring an analysis of the interactions amongst material composition, processing methods, and individual properties. Traditional data-driven materials design approaches typically rely on single-task models that operate independently, often neglecting the shared insights across related tasks. To overcome this limitation, we propose a collaborative design framework that employs multi-task learning for the development of novel Co-based superalloys. In this framework, six thermodynamic and microstructural property tasks share a common encoder, which effectively captures the underlying influence of alloy compositions across different properties. Each task then utilizes its own dedicated decoder to ensure precise predictions. As a result, the average normalized error for the predictions of the six properties is reduced by 37.5% compared to conventional single-task learning methods.

This package is released under MIT License, please see the LICENSE file for details.

This code and data is a companion to the paper, "Design of superalloys with multiple properties via multi-task learning."

**Features**
----------------------
- A multi-property collaborative design framework for alloys integrating multi-task learning (MTL) and inverse design.
- Shared knowledge extraction: A common encoder captures latent relationships across six thermodynamic and microstructural properties, reducing average prediction error by 37.5% compared to single-task learning.
- Task-specific optimization: Dedicated decoders for each property ensure precise predictions while leveraging shared insights from the encoder.
- Latent variable exploration: High-dimensional variables from the common encoder guide optimal alloy screening and inverse design.
- Multi-property balancing: Simultaneously optimizes six key criteria, including density, freezing range, γ′ size, γ′ solvus temperature, and oxidation resistance.
- Negative transfer mitigation: Identifies optimal task combinations for joint training to minimize conflicting gradients in MTL.
- First MTL application in alloy design: Demonstrates broad potential for collaborative materials development beyond superalloys.

**Citing**
----------------------
If you use this work (data or code), please cite the following work as appropriate:
```
Wang W,  et al. Design of superalloys with multiple properties via multi-task learning [J]. Acta Materialia.
```
