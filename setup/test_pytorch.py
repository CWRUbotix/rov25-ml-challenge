try:
    import torch
except:
    print('PyTorch not installed. '
          'Install PyTorch with CUDA 12.4 from https://pytorch.org/get-started/locally/')
    exit()

if not torch.cuda.is_available():
    print('CUDA not detected! Make sure to install PyTorch '
          'compiled with CUDA 12.4 from https://pytorch.org/get-started/locally/')

    # https://stackoverflow.com/questions/57238344/i-have-a-gpu-and-cuda-installed-in-windows-10-but-pytorchs-torch-cuda-is-availa
    try:
        torch.cuda.FloatTensor()
    except Exception as e:
        print('Your PyTorch probably wasn\'t compiled with CUDA support.'
              'PyTorch recommends installing CUDA and PyTorch together.')
        print(e)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')