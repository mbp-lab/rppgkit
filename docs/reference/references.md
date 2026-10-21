# References

This page lists the datasets, methods, algorithms and tools that rppgkit implements or
uses. Other pages cite these entries by author and year and link to them. Docstrings
and configuration comments give only the short citation.

## Datasets

Bobbia, S., Macwan, R., Benezeth, Y., Mansouri, A., & Dubois, J. (2019). Unsupervised
skin tissue segmentation for remote photoplethysmography. *Pattern Recognition
Letters*, 124, 82–90. <https://doi.org/10.1016/j.patrec.2017.10.017>
{ #bobbia2019 }

Used for: UBFC-rPPG (`datasets/ubfc_rppg.py`, `configs/dataset/ubfc_rppg.yaml`).

## Methods

Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). Algorithmic principles
of remote PPG. *IEEE Transactions on Biomedical Engineering*, 64(7), 1479–1491.
<https://doi.org/10.1109/TBME.2016.2609282>
{ #wang2017 }

Used for: POS (`methods/pos.py`, `configs/method/pos.yaml`).

Yu, Z., Li, X., & Zhao, G. (2019). Remote photoplethysmograph signal measurement from
facial videos using spatio-temporal networks. *British Machine Vision Conference
(BMVC)*. <https://arxiv.org/abs/1905.02419>
{ #yu2019 }

Used for: PhysNet and its negative Pearson loss (`methods/physnet.py`, `losses.py`,
`configs/method/physnet.yaml`).

## Signal processing

Tarvainen, M. P., Ranta-aho, P. O., & Karjalainen, P. A. (2002). An advanced detrending
method with application to HRV analysis. *IEEE Transactions on Biomedical
Engineering*, 49(2), 172–175. <https://doi.org/10.1109/10.979357>
{ #tarvainen2002 }

Used for: smoothness-priors detrending (`signal_processing.py`, `detrend_lambda` in
`configs/evaluation/default.yaml`).

Welch, P. (1967). The use of fast Fourier transform for the estimation of power
spectra: A method based on time averaging over short, modified periodograms. *IEEE
Transactions on Audio and Electroacoustics*, 15(2), 70–73.
<https://doi.org/10.1109/TAU.1967.1161901>
{ #welch1967 }

Used for: the `welch` heart-rate estimator (`hr_estimator: welch` in
`configs/evaluation/default.yaml`).

## Face detection

Qi, D., Tan, W., Yao, Q., & Liu, J. (2021). YOLO5Face: Why reinventing a face detector.
*arXiv:2105.12931*. <https://arxiv.org/abs/2105.12931>
{ #qi2021 }

Used for: the YOLO5Face detector (`roi/face/yolo5face/`, `configs/roi/yolo5face*.yaml`).

Yang, S., Luo, P., Loy, C. C., & Tang, X. (2016). WIDER FACE: A face detection
benchmark. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*,
5525–5533. <https://doi.org/10.1109/CVPR.2016.596>
{ #yang2016 }

Used for: the training data of the bundled YOLO5Face weights (`Y5sF_WFRGB.pth`).

Viola, P., & Jones, M. (2001). Rapid object detection using a boosted cascade of simple
features. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, I-511–I-518.
<https://doi.org/10.1109/CVPR.2001.990517>
{ #viola2001 }

Lienhart, R., & Maydt, J. (2002). An extended set of Haar-like features for rapid object
detection. *International Conference on Image Processing (ICIP)*, I-900–I-903.
<https://doi.org/10.1109/ICIP.2002.1038171>
{ #lienhart2002 }

Used for: the OpenCV Haar cascade detector (`roi/face/haar/`, `configs/roi/haar72.yaml`).

## Tools

Liu, X., Narayanswamy, G., Paruchuri, A., Zhang, X., Tang, J., Zhang, Y., Sengupta, R.,
Patel, S., Wang, Y., & McDuff, D. (2023). rPPG-Toolbox: Deep remote PPG toolbox.
*Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks
Track*. <https://arxiv.org/abs/2210.00716>
{ #liu2023 }

Used for: the reference that rppgkit's defaults are checked against; see
[Comparison with rPPG-Toolbox](rppg-toolbox.md).

Yadan, O. (2019). Hydra: A framework for elegantly configuring complex applications.
GitHub. <https://github.com/facebookresearch/hydra>
{ #yadan2019 }

Used for: configuration of every script (`configs/`).
