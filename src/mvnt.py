import csv
import copy
import numpy as np
from tqdm import tqdm
from torch.multiprocessing import Process,Manager
import torch.multiprocessing
# import pingouin as pg

from utils import average_weights
from update import federated_train_all,federated_test_idx

def MVN_Sampler(args,global_model,init_loss,local_weights,avg_idxs,test_idxs,train_dataset,user_groups,mvn_samples,count):
    avg_model=copy.deepcopy(global_model)
    for idxs in avg_idxs:
        avg_weight= average_weights(local_weights[idxs],omega=None)
        avg_model.load_state_dict(avg_weight)
        _,updated_loss = federated_test_idx(args,avg_model,test_idxs,train_dataset,user_groups)
        mvn_samples.append(np.array(updated_loss)-np.array(init_loss))
        count.value += 1
        # print("collected %d samples"%count.value)

def mp_watcher(total_num,count):
    pbar = tqdm(total=total_num,desc="Samples Collected",unit='samples')
    current_count = 0
    prev_count = 0
    while current_count<total_num:
        current_count=count.value
        if current_count>prev_count:
            pbar.update(current_count-prev_count)
            prev_count=current_count


def MVN_Test(args,global_model,train_dataset,user_groups,file_name=None):
    print("Training All Clients...")
    local_weights = federated_train_all(args,global_model,train_dataset,user_groups)
    # test_idxs = np.random.choice(range(args.num_users), args.mvn_dimensions, replace=False)
    test_idxs = list(range(args.num_users))
    _,init_loss = federated_test_idx(args,global_model,test_idxs,train_dataset,user_groups)
    m = max(int(args.frac * args.num_users), 1)

    global_model.share_memory()
    sample_per_worker = args.mvn_samples//args.mvnt_workers
    print("Collecting samples for MVN Test...")
    with Manager() as manager:
        count = manager.Value('i',0)
        mvn_samples = manager.list()
        # pool = Pool(processes = args.mvnt_workers)
        proc = []
        watcher = Process(target=mp_watcher,args=(args.mvn_samples,count))
        watcher.start()
        # proc.append(watcher)
        for i in range(args.mvnt_workers):
            avg_idxs = [np.random.choice(range(args.num_users), m, replace=False) for _ in range(sample_per_worker)]
            p = Process(target = MVN_Sampler,args = 
                        (args,global_model,
                        init_loss,local_weights,
                        avg_idxs,test_idxs,
                        train_dataset,user_groups,mvn_samples,count))
            p.start()
            proc.append(p)
        
        for p in proc:
            p.join()
        if watcher.is_alive():
            watcher.join(timeout=5)
            watcher.terminate()
        print("Collect finished!")
    
        if file_name is not None:
            with open(file_name,'w',newline='') as f:
                f_csv = csv.writer(f)
                f_csv.writerow(test_idxs)
                f_csv.writerows(mvn_samples)
                
        return np.array(mvn_samples)
