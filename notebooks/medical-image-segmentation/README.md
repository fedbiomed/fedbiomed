This folder contains two implementations of the medical segmentation notebook 
using IXI data. Both are based on the UNet network, but one notebook implements
unet with custom code while the other uses the unet library. The purpose of
this duplication is to show the tradeoff between adding dependencies on the
node side (i.e. requiring data providing centers to install the unet library)
and writing complex custom code. 
