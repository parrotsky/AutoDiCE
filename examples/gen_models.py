import torch
import torchvision.models as models
net = models.vit_b_16(weights='IMAGENET1K_V1').half().float()
net.eval()

torch.manual_seed(0)
x = torch.rand(1, 3, 224, 224)
a = net(x)

    # export torchscript
mod = torch.jit.trace(net, x, check_trace=False)
mod.save("vit_b_16.pt")
   # torchscript to pnnx
#    import os
#   os.system("pnnx test_vit_b_32.pt inputshape=[1,3,224,224]")

    # ncnn inference
#    import test_vit_b_32_ncnn
#    b = test_vit_b_32_ncnn.test_inference()
