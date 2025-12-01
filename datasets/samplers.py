import numpy as np
from torch.utils.data import Sampler

class GroupBatchSampler(Sampler):
    """
    같은 group_id를 가진 샘플들이 '절대 쪼개지지 않고' 같은 배치에 들어가도록 보장하는 샘플러.
    Monotone Loss 계산을 위해 필수적임.
    """

    def __init__(self, group_ids, batch_size: int, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        # 그룹핑: {group_id: [idx1, idx2, ...]}
        self.group_indices = {}
        for idx, g in enumerate(group_ids):
            g = int(g)
            if g not in self.group_indices:
                self.group_indices[g] = []
            self.group_indices[g].append(idx)

        self.groups = list(self.group_indices.keys())
        self.num_samples = sum(len(v) for v in self.group_indices.values())

    def __iter__(self):
        # Epoch마다 그룹 순서 섞기
        groups = self.groups.copy()
        if self.shuffle:
            np.random.shuffle(groups)

        batch = []
        for g in groups:
            indices = self.group_indices[g]
            
            # [중요 수정] 
            # 이번 그룹을 넣었을 때 배치가 넘치면, 지금 배치를 먼저 yield하고 비운다.
            # 즉, 그룹이 배치 사이에서 찢어지는 것을 방지함.
            if len(batch) + len(indices) > self.batch_size and len(batch) > 0:
                yield batch
                batch = []
            
            # 그룹 통째로 추가
            batch.extend(indices)

        # 마지막 남은 배치 yield
        if len(batch) > 0:
            yield batch

    def __len__(self):
        # 정확한 배치 수는 알 수 없으므로(가변적), 근사치 반환
        return (self.num_samples + self.batch_size - 1) // self.batch_size