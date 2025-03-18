# BKPSSA
Code of Hyperspectral Images Reconstructoion Based on Blur-Kernel-Prior and Spatial-Spectral Attention

Aiming at the problem of spatial detail loss and spectral feature degradation in hyperspectral images (HSIs) characterized as blur often caused by various noises during image acquisition and methods of removing blur noise designed on HSIs are insufficient, we propose a HSIs reconstruction network based on a Blur-Kernel-Prior (BKP) method and Spectral-Spatial Attention(SSA) strategy for noise removal and reconstruction of HSIs. Specifically, a grouping strategy is designed to segment the HSIs into spectral dimension sub-images, and the BKP module, based on UNet, learns the spatially adaptive blur kernel to extract and remove blurred features from each sub-image while preserving spatial features with spatial resolution. Subsequently, the SSA block is employed to extract shallow features, details, and edge information using a hybrid 2D-3D convolution from the sub-images and followed by deep features extraction using a deep ResNet and multi-head attention(MSA) on the merged image to maximize the preservation of spectral dimension information. The $L_1$ loss function, combined with spectral dimension loss and peak signal-to-noise ratio loss, is utilized to constrain and ensure reconstruction accuracy. Experiments on both synthetic and real datasets demonstrate that our method exhibits excellent performance in reconstructing HSIs affected by blurred noise, outperforming existing methods in terms of quantitative quality and recovery of spectral dimension information.

2025.3.18 Kernel Net has released.

Train and Test datasets can be download at :[Cave](https://cave.cs.columbia.edu/repository/Multispectral)
[Pavia University](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_Centre_and_University)
[Chikusei](https://paperswithcode.com/dataset/chikusei-dataset)
[XiongAn](http://www.hrs-cas.com/a/share/shujuchanpin/2019/0501/1049.html)
