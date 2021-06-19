import torch 
import torch.nn.functional as F

def generate_gnd(data, dis, filename):
    data_n = data.shape[0] 
    dis_dict = torch.zeros(data_n, data_n)
    for i in range(data_n):
        for j in range(i, data_n):
            dis_ij = dis(data[i], data[j]) 
            dis_dict[i,j]=dis_ij 
            dis_dict[j,i]=dis_ij 
    # sort 
    with open(filename, 'w') as f:
        for cur_id in range(data_n):
            id_sort = sorted(range(data_n), key=lambda x:dis_dict[cur_id,x])
            f.write(str(id_sort)[1:-1]+'\n')

def euclidean_dis(a,b):
    c = a - b 
    return c.norm()

def hamming_dis(a,b):
    assert a.shape[0] == b.shape[0] 
    cnt = 0
    for i in range(a.shape[0]):
        if a[i] != b[i]:
            cnt += 1
    return cnt 

if __name__ == '__main__':
    data0 = [ 
        [0,0],
        [1,0],
        [0,1],
        [1,1]
    ]
    data1 = torch.tensor(data0, dtype = torch.float32) 
    generate_gnd(data1, euclidean_dis, "eudis.txt")
    generate_gnd(data1, hamming_dis, "hamdis.txt")