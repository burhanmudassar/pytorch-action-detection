from torch.nn.modules.container import Sequential
from lib.modeling.components.modules import TemporalAggregator
import torch
import numpy as np
from collections import OrderedDict

from lib.utils.flops_counter import add_flops_counting_methods
from lib.utils.flops_counter import flops_to_string
from lib.utils.flops_counter import get_model_parameters_number

class Summary():
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.summary = OrderedDict()
        self.model.apply(self.register_hook)

    def register_hook(self, module):
        self.hooks.append(module.register_forward_hook(self.hook))

    def hook(self, module, input, output):
        if any([isinstance(module, TemporalAggregator),\
                isinstance(module, Sequential)]):
            return
        

        class_name = str(module.__class__).split(".")[-1].split("'")[0]
        module_idx = len(self.summary)

        m_key = "%s-%i" % (class_name, module_idx + 1)
        self.summary[m_key] = OrderedDict()
        if isinstance(input, (list, tuple)):
            self.summary[m_key]["input_shape"] = [
                list(it.size()) for it in input[0]
            ]
        else:
            self.summary[m_key]["input_shape"] = list(input[0].size())
        if isinstance(output, (list, tuple)):
            if isinstance(output[0], (list, tuple)):
                self.summary[m_key]["output_shape"] = [
                    list(o_.size()) for o in output for o_ in o
                ]
            else:
                self.summary[m_key]["output_shape"] = [
                    list(o.size()) for o in output
                ]
        else:
            self.summary[m_key]["output_shape"] = list(output.size())
        params = 0
        if hasattr(module, "weight") and hasattr(module.weight, "size"):
            params += torch.prod(torch.LongTensor(list(module.weight.size())))
            self.summary[m_key]["trainable"] = module.weight.requires_grad
        if hasattr(module, "bias") and hasattr(module.bias, "size"):
            params += torch.prod(torch.LongTensor(list(module.bias.size())))
        self.summary[m_key]["nb_params"] = params

        # if (
        #      not isinstance(module, nn.Sequential)
        #      and not isinstance(module, nn.ModuleList)
        #      and not (module == model)
        # ):

    def print_summary(self):
        print("----------------------------------------------------------------")
        line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
        print(line_new)
        print("================================================================")
        total_params = 0
        total_output = 0
        trainable_params = 0
        for layer in self.summary:
            # input_shape, output_shape, trainable, nb_params
            line_new = "{:>20} {:>25} {:>25} {:>15}".format(
                layer,
                str(self.summary[layer]["input_shape"]),
                str(self.summary[layer]["output_shape"]),
                "{0:,}".format(self.summary[layer]["nb_params"]),
            )
            total_params += self.summary[layer]["nb_params"]
            total_output += np.sum([np.prod(a) for a in self.summary[layer]["output_shape"]])
            if "trainable" in self.summary[layer]:
                if self.summary[layer]["trainable"] == True:
                    trainable_params += self.summary[layer]["nb_params"]
            print(line_new)

        total_input_size = abs(np.prod(self.input_size) * self.batch_size * 4. / (1024 ** 2.))
        total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
        total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
        total_size = total_params_size + total_output_size + total_input_size

        print("================================================================")
        print("Total params: {0:,}".format(total_params))
        print("Trainable params: {0:,}".format(trainable_params))
        print("Non-trainable params: {0:,}".format(total_params - trainable_params))
        print("----------------------------------------------------------------")
        print("Input size (MB): %0.2f" % total_input_size)
        print("Forward/backward pass size (MB): %0.2f" % total_output_size)
        print("Params size (MB): %0.2f" % total_params_size)
        print("Estimated Total Size (MB): %0.2f" % total_size)
        print("----------------------------------------------------------------")

    def do_param_count(self, input_size=(1, 3, 3, 300, 300)):
        self.input_size = input_size
        self.batch_size = input_size[0]

        batch = torch.FloatTensor(*input_size)
        if torch.cuda.is_available():
            batch = batch.cuda()
        # with torch.no_grad():
        self.model.train()
        out = self.model(batch)

        self.print_summary()

    def do_flop_count(self, input_size=(1, 3, 3, 300, 300)):
        self.input_size = input_size
        self.batch_size = input_size[0]

        batch = torch.FloatTensor(*input_size)
        if torch.cuda.is_available():
            batch = batch.cuda()
        net = add_flops_counting_methods(self.model)
        net.eval().start_flops_count()

        out = self.model(batch)

        print('Flops:  {}'.format(flops_to_string(net.compute_average_flops_cost())))
        print('Params: ' + get_model_parameters_number(net))
        
        
        
if __name__ == '__main__':
    # Test Case for running Summarize Object
    from torchvision.models import resnet50

    model = resnet50(pretrained=True)
    if torch.cuda.is_available():
        model = model.cuda()
    s_ = Summary(model=model)

    s_.do_param_count(input_size=(1, 3, 224, 224))
    s_.do_flop_count(input_size=(1, 3, 224, 224))
