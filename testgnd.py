import torch 
from itq import hamming_convert
from gnd import generate_gnd, euclidean_dis, hamming_dis


a = torch.randn(16, 4, dtype=torch.float32) * 100

b = hamming_convert(a, 4, 48, 0) 

generate_gnd(a,euclidean_dis,"eudis.txt")
generate_gnd(b,hamming_dis,"ham.txt")