import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import shap
import matplotlib.pyplot as plt

run_test = 0

def plt_heat(shared_features):
    global run_test

    intermediate_output_np = shared_features.cpu().detach().numpy()
    # 可视化激活热图
    plt.imshow(intermediate_output_np, cmap='viridis', aspect='auto')
    plt.title('Activation Heatmap of Shared Layer')
    plt.xlabel('Neuron Index')
    plt.ylabel('Sample Index')
    plt.colorbar()
    plt.savefig(r"E:\desktop\save\heat_%s.jpg" % run_test)
    plt.close()



class AbsArchitecture(nn.Module):
    r"""An abstract class for MTL architectures.

    Args:
        task_name (list): A list of strings for all tasks.
        encoder_class (class): A neural network class.
        decoders (dict): A dictionary of name-decoder pairs of type (:class:`str`, :class:`torch.nn.Module`).
        rep_grad (bool): If ``True``, the gradient of the representation for each task can be computed.
        multi_input (bool): Is ``True`` if each task has its own input data, otherwise is ``False``. 
        device (torch.device): The device where model and data will be allocated. 
        kwargs (dict): A dictionary of hyperparameters of architectures.

    """

    def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
        super(AbsArchitecture, self).__init__()

        self.task_name = task_name
        self.task_num = len(task_name)
        self.encoder = encoder_class
        self.decoders = decoders
        self.rep_grad = rep_grad
        self.multi_input = multi_input
        self.device = device
        self.kwargs = kwargs

        if self.rep_grad:
            self.rep_tasks = {}
            self.rep = {}

    def forward(self, inputs, task_name=None):
        r"""

        Args:
            inputs (torch.Tensor): The input data.
            task_name (str, default=None): The task name corresponding to ``inputs`` if ``multi_input`` is ``True``.

        Returns:
            dict: A dictionary of name-prediction pairs of type (:class:`str`, :class:`torch.Tensor`).
        """
        # out = {}
        # first_outputs = list()
        # same_rep = True if not isinstance(inputs, list) and not self.multi_input else False
        # for tn, task in enumerate(self.task_name):
        #     if task_name is not None and task != task_name:
        #         continue
        #     ss_rep = inputs[tn] if isinstance(inputs, list) else inputs
        #     ss_rep = self._prepare_rep(ss_rep, task, same_rep)
        #     encoder_out = self.encoder[task](ss_rep.float())
        #     out[task] = encoder_out
        #     first_outputs.append(encoder_out)
        # out2={}
        # for tn, task in enumerate(self.task_name):
        #     if task_name == task:
        #         # ss_rep = s_rep[tn] if isinstance(s_rep, list) else s_rep
        #         ss_rep = first_outputs[0] if isinstance(first_outputs, list) else first_outputs
        #         ss_rep = self._prepare_rep(ss_rep, task, same_rep)
        #         out2[task] = self.decoders[task](ss_rep)

        first_outputs = list()
        same_rep = True if not isinstance(inputs[0], list) and not self.multi_input else False
        out = {}
        for tn, task in enumerate(self.task_name):
            if task_name is not None and task != task_name:
                continue
            ss_rep = inputs[0][tn] if isinstance(inputs[0], list) else inputs[0]
            ss_rep = self._prepare_rep(ss_rep, task, same_rep)
            # print("*"*10)
            # print(inputs[0])
            # print(ss_rep.float())
            # print("*-*" * 10)
            shared_features = self.encoder(ss_rep.float())
            out[task] = shared_features
            first_outputs.append(shared_features)
        out2 = {}
        for tn, task in enumerate(self.task_name):
            if task_name == task:
                # ss_rep = s_rep[tn] if isinstance(s_rep, list) else s_rep
                ss_rep = first_outputs[0] if isinstance(first_outputs, list) else first_outputs
                ss_rep = self._prepare_rep(ss_rep, task, same_rep)
                # print(task_name)
                if inputs[1].numel() != 0:
                    # print(ss_rep.shape)
                    # print(inputs[1].shape)
                    out2[task] = self.decoders[task](torch.cat((ss_rep, inputs[1]), dim=1).to(torch.float32))
                else:
                    out2[task] = self.decoders[task](ss_rep)
        # print("out2")
        # print(out2)
        return out2, out#, out

    def get_share_params(self):
        r"""Return the shared parameters of the model.
        """
        return self.encoder.parameters()

    def zero_grad_share_params(self):
        r"""Set gradients of the shared parameters to zero.
        """
        self.encoder.zero_grad()

    def _prepare_rep(self, rep, task, same_rep=None):
        if self.rep_grad:
            if not same_rep:
                self.rep[task] = rep
            else:
                self.rep = rep
            self.rep_tasks[task] = rep.detach().clone()
            self.rep_tasks[task].requires_grad = True
            return self.rep_tasks[task]
        else:
            return rep
