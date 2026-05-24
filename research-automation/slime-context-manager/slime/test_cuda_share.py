import torch
import multiprocessing as mp

def test_share_cuda_tensor():
    tensor = torch.randn(10, 10).cuda()
    shared_tensor = tensor.share_memory_()
    print("Tensor shared successfully")

if __name__ == "__main__":
    mp.set_start_method("spawn")
    p = mp.Process(target=test_share_cuda_tensor)
    p.start()
    p.join()
