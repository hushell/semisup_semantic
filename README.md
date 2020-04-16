# Prerequisites
```
pip install torchnet tensorboardX tensorboard
conda install -c menpo opencv
```

# Usage
```
python main.py --gpu 0
```

# tensorboard visualization
```
ssh -L1234:localhost:1234 GPU_machine_IP
tensorboard --port 1234 --logdir outputs_dir
```
