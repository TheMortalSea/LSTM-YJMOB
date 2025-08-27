import os
import argparse
import csv
import wandb
import time
from tqdm import tqdm


import torch
from torch.utils.data import DataLoader
from dataset import *
from model import *

path_arr = [
    './dataset/city_A_challengedata.csv.gz',
    './dataset/city_B_challengedata.csv.gz',
    './dataset/city_C_challengedata.csv.gz',
    './dataset/city_D_challengedata.csv.gz'
]

def Inference(args):

    # 初始化 wandb
    # name = 'LPBERT-postembedABCD_inference_testB'
    # wandb.init(project="LPBERT", name=name, config=args)
    # wandb.run.name = name  # Set the run name
    # wandb.run.save()

    # 设置结果的存储路径
    result_path = 'inference/postembedABCD/'
    result_name = 'cityD_valid_100users.csv'
    os.makedirs(result_path, exist_ok=True)

    # 加载验证集
    # dataset_test = TestSet(path_arr[1])
    dataset_test = ValidationSet(path_arr[3], is_100val=True)
    dataloader_test = DataLoader(dataset_test, batch_size=1, num_workers=args.num_workers)

    # 通过cuda:<device_id>指定使用的GPU
    device = torch.device(f'cuda:{args.cuda}')

    # 实例化LP-BERT模型加载至GPU上，并加载预训练模型的参数
    model = LPBERT(args.layers_num, args.heads_num, args.embed_size, args.city_embed).to(device)
    model.load_state_dict(torch.load(args.pth_file, map_location=device))

    # 初始化
    results = []

    # 模型验证
    start_time = time.time()
    model.eval() # 评估模式
    with torch.no_grad():
        for data in tqdm(dataloader_test):

            # 将数据加载到GPU上
            data['uid'] = data['uid'].to(device)
            data['d'] = data['d'].to(device)
            data['t'] = data['t'].to(device)
            data['input_x'] = data['input_x'].to(device)
            data['input_y'] = data['input_y'].to(device)
            data['time_delta'] = data['time_delta'].to(device)
            data['city'] = data['city'].to(device)
            data['len'] = data['len'].to(device)

            # 获取推测
            output = model(data['d'], data['t'], data['input_x'], data['input_y'], data['time_delta'], data['len'], data['city'])

            # 处理输出
            assert torch.all((data['input_x'] == 201) == (data['input_y'] == 201))  # 捡查x和y的掩码是否一致
            pred_mask = (data['input_x'] == 201)
            output = output[pred_mask]
            pred = []
            pre_x, pre_y = -1, -1
            for step in range(len(output)):
                if step > 0:
                    output[step][0][pre_x] *= 0.9
                    output[step][1][pre_y] *= 0.9

                pred.append(torch.argmax(output[step], dim=-1))
                pre_x, pre_y = pred[-1][0].item(), pred[-1][1].item()
            
            # 
            origin_mask = (data['input_x'] >= 1) & (data['input_x'] <= 200)
            origin = torch.cat((
                data['uid'][origin_mask].unsqueeze(-1),
                data['d'][origin_mask].unsqueeze(-1)-1, 
                data['t'][origin_mask].unsqueeze(-1)-1, 
                data['input_x'][origin_mask].unsqueeze(-1),
                data['input_y'][origin_mask].unsqueeze(-1)),
                dim=-1).cpu().tolist()
            origin = [tuple(x) for x in origin]

            # 生成预测结果
            pred = torch.stack(pred)
            generated = torch.cat((
                data['uid'][pred_mask].unsqueeze(-1),
                data['d'][pred_mask].unsqueeze(-1)-1, 
                data['t'][pred_mask].unsqueeze(-1)-1, 
                pred+1), dim=-1).cpu().tolist()
            generated = [tuple(x) for x in generated]
            
            # 添加到结果列表
            results.extend(origin)
            results.extend(generated)
    
    end_time = time.time()
    time_delta = end_time - start_time
    print(f'average inference time :{time_delta / 3000}')

    # 保存结果
    csv_file_path = os.path.join(result_path, result_name)
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['uid', 'd', 't', 'x', 'y'])
        writer.writerows(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth_file', type=str, default='/content/drive/MyDrive/best_finetune_model.pth')     # 改为训练完成的模型的存储地址
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--city_embed', type=int, default=4)
    parser.add_argument('--layers_num', type=int, default=4)
    parser.add_argument('--heads_num', type=int, default=8)
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()

    Inference(args)