
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
 
from src.custom_layers import * 

class SmallWeightTrainer(Trainer):
    def __init__(self, args):
        self.args = args
        self.single_weight_parameter = self._get_single_weight_parameter()
        super().__init__(args)

    def _get_single_weight_parameter(self):
        single_weight_parameter = torch.tensor(0.5, requires_grad=self.args.soft_average, device=device)
        print('single_weight_parameter.requires_grad:', single_weight_parameter.requires_grad)
        single_weight_parameter.data.clamp_(0, 1)
        return single_weight_parameter

    def _get_optimizer(self) -> optim.Optimizer:
        try:
            optimizer_cls = OPTIMIZERS[self.args.optimizer]
        except KeyError:
            raise Exception(f'Optimizer not found. Given: {self.args.optimizer}, Have: {OPTIMIZERS.keys()}')

        print(
            f'Optimizer {self.args.optimizer} with lr={self.args.lr}, betas={(self.args.betas[0], self.args.betas[1])}, wd={self.args.weight_decay}'
        )
        return optimizer_cls(
            params=[
                {'params': self.single_weight_parameter, 'lr': self.args.lr * self.args.single_weight__lr_multiplier},
                {'params': self.milnet.parameters()}
            ],
            lr=self.args.lr,
            betas=(self.args.betas[0], self.args.betas[1]),
            weight_decay=self.args.weight_decay
        )

    def _run_model(self, bag_feats, bag_label) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        ins_prediction, bag_prediction, attentions = self.milnet(bag_feats)
        if len(ins_prediction.shape) == 2:
            max_prediction, _ = torch.max(ins_prediction, 0)
        else:
            max_prediction, _ = torch.max(ins_prediction, 1)

        bag_loss = self.criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        max_loss = self.criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
        loss = self.single_weight_parameter * bag_loss + (1 - self.single_weight_parameter) * max_loss

        with torch.no_grad():
            bag_prediction = (
                    (1 - self.single_weight_parameter) * torch.sigmoid(max_prediction) +
                    self.single_weight_parameter * torch.sigmoid(bag_prediction)
            ).squeeze().cpu().numpy()

        return bag_prediction, loss, ins_prediction

    def train(self, data, cur_epoch):
        res = super().train(data, cur_epoch)
        return res

    def _after_run_model_in_training_mode(self, step, num_bags, batch_idx):
        super()._after_run_model_in_training_mode(step, num_bags, batch_idx)
        self.single_weight_parameter.data.clamp_(0, 1)

    def __str__(self):
        return f'Single_Weight__sa{self.args.soft_average}'
    

class Snuffy(SmallWeightTrainer):
    def _get_milnet(self) -> nn.Module:
        #feature embedding 
        # HERE 
        
        #instance classifier 
        i_classifier = snuffy.FCLayer(in_size=self.args.feats_size,
                                      out_size=self.args.num_classes).to(device)

        c = copy.deepcopy
        attn = MultiHeadedAttention(
            self.args.num_heads,
            self.args.feats_size,
        ).to(device)
        
        ff = PositionwiseFeedForward(
            self.args.feats_size,
            self.args.feats_size * self.args.mlp_multiplier,
            self.args.activation,
            self.args.encoder_dropout
        ).to(device)
        
        #bag classifier         
        b_classifier = BClassifier(
            snuffy.Encoder(
                snuffy.EncoderLayer(
                    self.args.feats_size,
                    c(attn),
                    c(ff),
                    self.args.encoder_dropout,
                    self.args.big_lambda,
                    self.args.random_patch_share
                ), self.args.depth
            ),
            self.args.num_classes,
            self.args.feats_size
        ).to(device)
        
        #milnet = i_classifier + b_classifer 
        milnet = MILNet(i_classifier, b_classifier).to(device)
        #init job
        init_funcs_registry = {
            'trunc_normal': nn.init.trunc_normal_,
            'kaiming_uniform': nn.init.kaiming_uniform_,
            'kaiming_normal': nn.init.kaiming_normal_,
            'xavier_uniform': nn.init.xavier_uniform_,
            'xavier_normal': nn.init.xavier_normal_,
            'orthogonal': nn.init.orthogonal_
        }

        modules = [(self.args.weight_init__weight_init_i__weight_init_b[1], 'i_classifier'),
                   (self.args.weight_init__weight_init_i__weight_init_b[2], 'b_classifier')]
        
        print('modules:', modules)
        for init_func_name, module_name in modules:
            init_func = init_funcs_registry.get(init_func_name)
            print('init_func:', init_func)
            for name, p in milnet.named_parameters():
                if p.dim() > 1 and name.split(".")[0] == module_name:
                    init_func(p)

        return milnet

    def _run_model(self, bag_feats, bag_label) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        '''
        This method handles the forward pass and loss calculation for training 
        '''
        bag_prediction, loss, ins_prediction = super()._run_model(bag_feats, bag_label)
        
        ins_prediction = ins_prediction.view(-1, 1)
        
        return bag_prediction, loss, torch.sigmoid(ins_prediction)

    def __str__(self):
        return f'Snuffy_k{self.args.big_lambda}_sa{self.args.soft_average}_depth{self.args.depth}'

 