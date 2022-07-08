import torch


class Args:
    n_epochs = 10
    batch_size_train = 128
    batch_size_test = 128
    learning_rate = 0.001  # resnet
    log_interval = 10
    random_seed = 1
    USE_GPU = torch.cuda.is_available()
    show_examples = False
    train_data_path = "./data_sem/train"
    test_data_path = "./data_sem/test"
    translit = False

    def __init__(self):
        self.model_save_path = self._model_save_path()

    def _model_save_path(self):
        if "sem" in self.train_data_path:
            return self._path("sem")
        if "common" in self.train_data_path:
            return self._path("common")
        if Args.translit:
            return self._path('translit')
        else:
            raise NameError

    @staticmethod
    def _path(name):
        return './save_models/' + name + '_model.pth', './save_models/' + name + '_optimizer.pth'
