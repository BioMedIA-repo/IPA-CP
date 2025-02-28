from models.unet_3D import unet_3D


def net_factory_3d(net_type="unet_3D", in_chns=1, class_num=2):
    if net_type == "mynet":
        net = unet_3D(n_classes=class_num, in_channels=in_chns)
    else:
        net = None
    return net

