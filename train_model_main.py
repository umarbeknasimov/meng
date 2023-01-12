import main
import torch
import models
from args import MainArgs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'using device: {DEVICE}')
args = MainArgs()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

model = models.frankleResnet20().to(DEVICE)
# model.load_state_dict(torch.load('weights_frankle_seed_1_i=2048', map_location=DEVICE))
main.main(model, args, DEVICE)