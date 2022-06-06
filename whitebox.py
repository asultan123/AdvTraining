# from networks.ensemble_resnet import (
#     avg_ensemble_2_vgg8,
#     avg_ensemble_3_resnet18,
#     avg_ensemble_3_resnet18_fc,
#     cat_1_resnet34,
#     cat_1_resnet50,
#     ensemble_1_resnet18,
#     ensemble_1_resnet34,
#     ensemble_1_resnet50,
# )
import pickle
from config.dataset_config import getData
from art.classifiers import PyTorchClassifier
from art.attacks.saliency_map import SaliencyMapMethod
from art.attacks import (
    BasicIterativeMethod,
    FastGradientMethod,
    ProjectedGradientDescent,
)
import cw
from sklearn.metrics import confusion_matrix
# import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pickle as pkl
import os
import argparse
import torch
# from functools import partial
import os
import random
from preactresnet import PreActResNet18
# from torchvision import datasets, transforms


torch.cuda.current_device()
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

parser = argparse.ArgumentParser(description="PyTorch CNN Testing")
parser.add_argument(
    "--method",
    type=str,
    default="adv_train",
    help="methods:[baseline adp deg pdd pdd_deg adv_train]",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="CIFAR100",
    help="datasets:[Tiny_Image FashionMNIST CIFAR100]",
)
parser.add_argument(
    "--attack", type=str, default="PGD", help="attack methods:[FGSM PGD BIM JSMA CW]"
)
parser.add_argument("--norm", type=str, default="Linf", help="norm:[L2, Linf]")
parser.add_argument("--model-path", type=str,
                    default=None, help="model path")
parser.add_argument("--model-base-path", type=str,
                    default="./cifar100_model", help="model base path")
parser.add_argument("--eval-path", type=str,
                    default=None, help="model path")
parser.add_argument("--bs", type=int, default=128, help="batch size")
parser.add_argument("--conf", action="store_true")
parser.add_argument("--save_dir", type=str, default="bs",
                    help="save confusion matrix")
parser.add_argument('--data-dir', default='./cifar100-data', type=str)

parser.add_argument('--show-progress', action='store_true')
parser.set_defaults(show_progress=False)
opt = parser.parse_args()

if opt.show_progress:
    import progress


class LogNLLLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(LogNLLLoss, self).__init__()
        assert reduction == "mean"
        self.reduction = reduction

    def forward(self, x, targets):
        log_x = torch.log(x)
        log_nll_loss = nn.NLLLoss()(log_x, targets)
        return log_nll_loss


if opt.dataset == "FashionMNIST":
    num_classes, train_data, test_data = getData("FashionMNIST")
    input_shape = (opt.bs, 3, 32, 32)
if opt.dataset == "CIFAR100":
    num_classes, train_data, test_data = getData("CIFAR100")
    input_shape = (opt.bs, 3, 32, 32)
if opt.dataset == "CIFAR10":
    num_classes, train_data, test_data = getData("CIFAR10")
    input_shape = (opt.bs, 3, 32, 32)
if opt.dataset == "Tiny_Image":
    num_classes, train_data, test_data = getData("Tiny_Image")
    input_shape = (opt.bs, 3, 64, 64)


trainloader = torch.utils.data.DataLoader(
    train_data, batch_size=opt.bs, shuffle=True, num_workers=4, pin_memory=True
)

testloader = torch.utils.data.DataLoader(
    test_data, batch_size=opt.bs, shuffle=False, num_workers=4, pin_memory=True
)

if opt.method == "adv_train":
    model = PreActResNet18(num_classes)
    model_str = "/adv_train_preActResNet18"
else:
    raise NotImplementedError

if opt.model_path is None:
    model_path = opt.model_base_path + "/" + opt.dataset + model_str + "/best_model.pth"
else:
    model_path = opt.model_path

if opt.method != "adv_train":
    model.load_state_dict(torch.load(model_path)["net"])
else:
    model.load_state_dict(torch.load(model_path))



model.cuda()
model.eval()

criterion = LogNLLLoss()

classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    optimizer=None,
    clip_values=(0.0, 1.0),
    input_shape=input_shape,
    nb_classes=num_classes,
)


torch.manual_seed(2302)
np.random.seed(2302)
random.seed(2302)


def test_clean(testloader, model):
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if True:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)
        correct += predicted.eq(targets.data).cpu().sum()
        total += targets.size(0)

        if opt.show_progress:
            progress.progress_bar(
                batch_idx,
                len(testloader),
                "clean_acc: %.3f%% (%d/%d)"
                "" % (100.0 * float(correct) / total, correct, total),
            )
    return 100 * float(correct) / total


def test_robust(opt, model, classifier, attack_method, c, norm=None):
    if opt.attack == "FGSM":
        adv_crafter = FastGradientMethod(
            classifier,
            norm=norm,
            eps=c,
            targeted=False,
            num_random_init=0,
            batch_size=opt.bs,
        )
    if opt.attack == "PGD":
        adv_crafter = ProjectedGradientDescent(
            classifier,
            norm=norm,
            eps=c,
            eps_step=c / 10,
            max_iter=10,
            targeted=False,
            num_random_init=0,
            batch_size=opt.bs,
        )
    if opt.attack == "BIM":
        adv_crafter = ProjectedGradientDescent(
            classifier,
            norm=norm,
            eps=c,
            eps_step=c / 10.0,
            max_iter=10,
            targeted=False,
            num_random_init=0,
            batch_size=opt.bs,
        )
    if opt.attack == "JSMA":
        adv_crafter = SaliencyMapMethod(
            classifier, theta=0.1, gamma=c, batch_size=opt.bs
        )
    if opt.attack == "CW":
        adv_crafter = cw.L2Adversary(
            targeted=False,
            confidence=0.01,
            c_range=(c, 1e10),
            max_steps=1000,
            abort_early=False,
            search_steps=5,
            box=(0.0, 1.0),
            optimizer_lr=0.01,
        )

    exclusive_correct = 0
    total = 0
    total_sum = 0
    common_id = []
    y_pred = []
    y_true = []

    test_n = 0
    inclusive_correct = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        output = classifier.predict(inputs.cpu().numpy(), batch_size=opt.bs)
        output = torch.tensor(output)
        output = output.cuda()
        init_pred = output.max(1, keepdim=False)[1]
        common_id = np.where(init_pred.cpu().numpy() ==
                             targets.cpu().numpy())[0]

        if opt.attack == "CW":
            x_test_adv = adv_crafter(model, inputs, targets, to_numpy=True)
            perturbed_output = classifier.predict(x_test_adv)
        elif opt.attack == "CleverPGD":
            x_test_adv = adv_crafter.generate(x=inputs)
            perturbed_output = model(x_test_adv)
        else:
            x_test_adv = adv_crafter.generate(x=inputs.cpu().numpy())
            perturbed_output = classifier.predict(x_test_adv)

        perturbed_output = torch.tensor(perturbed_output)
        perturbed_output = perturbed_output.cuda()
        final_pred = perturbed_output.max(1, keepdim=False)[1]
        total_sum += targets.size(0)
        total += len(common_id)
        exclusive_correct += final_pred[common_id].eq(
            targets[common_id].data).cpu().sum()
        exclusive_attack_acc = 100.0 * float(exclusive_correct) / total

        inclusive_correct += (perturbed_output.max(1)[1] == targets).sum().item()
        test_n += targets.size(0)
        inclusive_attack_acc = 100.0 * float(inclusive_correct) / test_n

        if opt.conf:
            y_pred.extend(final_pred.data.cpu().numpy())  # Save Prediction
            y_true.extend(targets.data.cpu().numpy())  # Save Truth

        if opt.show_progress:
            progress.progress_bar(
                batch_idx,
                len(testloader),
                "Attack Strength:%.3f, robust accuracy (exclusive): %.3f%% (%d/%d), robust accuracy (inclusive): %.3f%% (%d/%d)"
                "" % (c, exclusive_attack_acc, exclusive_correct, total, inclusive_attack_acc, inclusive_correct, test_n),
            )

    if opt.conf:
        print("==> Generating confusion matrix..")
        cf_matrix = confusion_matrix(y_true, y_pred)
        print(cf_matrix)
        conf_mat_file = os.path.join("models/" + opt.dataset + model_str, "conf_mat.pkl")
        with open(conf_mat_file, "wb") as f:
            pkl.dump(cf_matrix, f)
        print("==> Saved confusion matrix in {}..".format(conf_mat_file))

    return {'eps': c, 'inclusive': inclusive_attack_acc, 'exclusive': exclusive_attack_acc}

if __name__ == "__main__":
    print("Clean Accuracy:")
    result_dict = {}
    correct_acc = test_clean(testloader, model)
    result_dict['clean'] = correct_acc
    print("Attack: {}".format(opt.attack))
    if opt.attack == "FGSM" or "BIM" or "PGD":
        if opt.norm == "Linf":
            # [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
            epsilons = [0.01, 0.02, 8/255]
            norm = np.inf
        if opt.norm == "L2":
            epsilons = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8]
            norm = 2
    if opt.attack == "JSMA":
        epsilons = [0.05, 0.1, 0.2, 0.4]
        norm = None
    if opt.attack == "CW":
        epsilons = [0.001, 0.01, 0.1]
        norm = None
    result_dict['attacked'] = []
    for eps in epsilons:
        print(f"Evaluating error: {eps}")
        attack_result = test_robust(opt, model, classifier, opt.attack, eps, norm=norm)
        result_dict['attacked'].append(attack_result)
    if opt.eval_path:
        with open(opt.eval_path, "wb") as file:
            pickle.dump(result_dict, file)
    print(result_dict)