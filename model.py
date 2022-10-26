import torch
import torch.nn as nn
from torchvision import models


class Gaze_Attention_Model(nn.Module):

    def __init__(self, num_classes = 91, hidden_layer_dim = [512, 256], layer_dim_rnn = 1, use_cls_head = False):
        super(Gaze_Attention_Model, self).__init__()

        self.num_classes = num_classes
        self.hidden_layer_dim = hidden_layer_dim
        self.hidden_layer_dim_rnn = layer_dim_rnn
        self.use_cls_head = use_cls_head

        backbone = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        
        # Freeze early layers
        for param in backbone.parameters():
            param.requires_grad = False

        self.feature_extractor = backbone.backbone

        num_features = 256

        self.conv_1 = nn.Conv2d(
            in_channels = num_features, 
            out_channels = num_features * 2, 
            kernel_size = 3,
            stride = 1,
            padding = 1
        )
        self.batch_norm1 = nn.BatchNorm2d(num_features * 2)
        
        self.pool_1 = nn.MaxPool2d(2)

        self.conv_2 = nn.Conv2d(
            in_channels = num_features * 2, 
            out_channels = num_features * 4, 
            kernel_size = 3,
            stride = 1,
            padding = 1
        )
        self.batch_norm2 = nn.BatchNorm2d(num_features * 4)

        self.pool_2 = nn.MaxPool2d(2)

        self.relu_fn = nn.ReLU()

        self.hidden_layer = nn.Sequential(
            nn.Flatten(), nn.Linear(num_features * 4 * 32 * 32, num_features * 2),
            nn.ReLU()
        )

        self.rnn_unit = nn.GRU(input_size = num_features * 2, hidden_size = self.hidden_layer_dim[0], num_layers = self.hidden_layer_dim_rnn)
        
        # Adding a bounding box head on top of the pretrained model
        self.bbx_head = nn.Sequential(
            nn.Linear(self.hidden_layer_dim[0], self.hidden_layer_dim[1]), 
            nn.ReLU(),
            nn.Linear(self.hidden_layer_dim[1], 4)
        )

        if use_cls_head == True:
            # Adding a classification head on top of the pretrained model
            self.classifier = nn.Sequential(
                nn.Linear(num_features * 2, self.hidden_layer_dim[0]), 
                nn.ReLU(),
                nn.Linear(self.hidden_layer_dim[0], self.hidden_layer_dim[1]), 
                nn.ReLU(),
                nn.Linear(self.hidden_layer_dim[1], self.num_classes)
            )


    def forward(self, x, gaze_map):

        # Feature extraction
        self.feature_extractor.eval()
        with torch.inference_mode():
            feature_map = self.feature_extractor(x)['0']

        # Multiplying with the fixation density map
        processed_maps = torch.mul(feature_map, gaze_map)  # 256 x 128 x 128

        processed_maps = self.relu_fn(self.batch_norm1(self.conv_1(processed_maps))) # 512 x 128 x 128
        processed_maps = self.pool_1(processed_maps) # 512 x 64 x 64
        processed_maps = self.relu_fn(self.batch_norm2(self.conv_2(processed_maps))) # 1024 x 64 x 64
        processed_maps = self.pool_2(processed_maps) # 1024 x 32 x 32

        feature_vector, _ = self.rnn_unit(self.hidden_layer(processed_maps)) # 512

        # Extracting the bounding box
        bbx = self.bbx_head(feature_vector)

        if self.use_cls_head == True:
            # Extracting the classification
            cls = self.classifier(feature_vector)
            return bbx, cls

        else:   
            return bbx


if __name__ == "__main__":

    from torchvision import transforms
    from data import MOT17_Dataset
    from torch.utils.data import DataLoader

    trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    tmp_dataset = MOT17_Dataset(
        participant_id = 1,
        videos = list(range(1,15)),
        csv_folder_path = '../Semantic Eye-Tracking/data/experiment1',
        video_folder_path = '../Semantic Eye-Tracking/data/MOT17_video_frames',
        gaze_img_path = '../Semantic Eye-Tracking/data/sequences',
        sequence_length = 5,
        sigma = 500, 
        transform = trans
    )

    tmp_dataloader = DataLoader(
        tmp_dataset, 
        batch_size = 1, 
        shuffle = False, 
        num_workers = 0
    )

    use_cls_head = False

    model = Gaze_Attention_Model(use_cls_head = use_cls_head)
    print(model)

    bbx_loss = torch.nn.L1Loss(reduction='mean')
    if use_cls_head == True:
        cls_loss = torch.nn.CrossEntropyLoss(reduction='mean')

    for idx, train_batch in enumerate(tmp_dataloader):
        
        img, gaze, gt_cls, gt_bbx, _ = train_batch

        # Forward pass
        if use_cls_head == True:
            bbx, cls = model(img.squeeze(0), gaze.squeeze(0))
            bbx_loss_val = bbx_loss(bbx, gt_bbx.squeeze(0))
            cls_loss_val = cls_loss(cls, gt_cls.squeeze(0))
            loss = bbx_loss_val + cls_loss_val

        else:     
            bbx = model(img.squeeze(0), gaze.squeeze(0))   
            bbx_loss_val = bbx_loss(bbx, gt_bbx.squeeze(0))
            loss = bbx_loss_val

        print(loss)

        break