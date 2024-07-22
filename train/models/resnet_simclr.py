import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, freeze_extractor):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d),
                            "resnet50": models.resnet50(pretrained=False, norm_layer=nn.InstanceNorm2d)}

        self.resnet = self._get_basemodel(base_model)
        dim_mlp = self.resnet.fc.in_features

        # self.features = nn.Sequential(*list(resnet.children())[:-1])

        # projection MLP
        # self.l1 = nn.Linear(num_ftrs, num_ftrs)
        # self.l2 = nn.Linear(num_ftrs, out_dim)
        self.resnet.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                            nn.ReLU(), nn.Linear(dim_mlp, out_dim))

        self.freeze_extractor = freeze_extractor
        if self.freeze_extractor:
            for name, param in self.resnet.named_parameters():
                print(name)
                if "fc" not in name:
                    param.requires_grad = False
                print( param.requires_grad)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        # h = self.features(x)
        # h = h.squeeze()

        # x = self.l1(h)
        # x = F.relu(x)
        # x = self.l2(x)
        # return h, x
        return self.resnet(x)