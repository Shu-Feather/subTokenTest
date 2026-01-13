from torch.utils.data import Dataset
import torch
from typing import Sequence, Self, Literal

from logger import logger

class ProbeDataset(Dataset):
    def __init__(self, tensor: torch.Tensor, id2token: dict[int, str], token2id: dict[str, int], char2label: dict[str, int], ids: Sequence[int] | None = None, tokens: Sequence[str] | None = None, max_num_each_char: int = 10, max_length_each_token: int = 12, label_target: Literal["exist", "number", "seq"] = "exist", shuffle_baseline: bool = False, random_rng: torch.Generator | None = None):
        """
        A dataset for probing a model with a linear layer.
        ------
        tensor: torch.Tensor
            The tensor to probe.
        id2token: dict[int, str]
            The id to token mapping of the tensor.
        """
        self.tensor = tensor.to(torch.device("cpu"))
        self.id2token = id2token
        self.token2id = token2id
        self.char2label = char2label
        if ids is not None:
            self.ids = ids
            if tokens is not None:
                logger.warning('Both ids and tokens are provided. Only ids will be used.')
        elif tokens is None:
            raise ValueError('Either ids or tokens must be provided')
        else:
            self.ids = [token2id[token] for token in tokens]
        self.max_num_each_char = max_num_each_char
        self.max_length_each_token = max_length_each_token
        self.label_target = label_target
        
        self.shuffle_baseline = shuffle_baseline
        if self.shuffle_baseline:
            rng = torch.Generator() if random_rng is None else random_rng
            self.shuffle_ids = [self.ids[idx] for idx in torch.randperm(len(self), generator=rng).tolist()]
            
    @property      
    def char_num(self) -> int:
        return len(self.char2label)
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        idx: int
            The index of the item to get.
        ------
        return: tuple[torch.Tensor, torch.Tensor]
            inputs: torch.Tensor, a 1d Tensor with shape=(hidden_dim, ). If shuffle_baseline is True, the inputs will be the shuffle_ids[idx] token, otherwise the inputs will be the ids[idx] token.
            label_exist: torch.Tensor, shape=(char_num, ), dtype=torch.float, device=self.tensor.device, 0 if the character does not appear in the token, 1 if the character appears in the token.
            label_number: torch.Tensor, shape=(char_num, ), dtype=torch.int, device=self.tensor.device, the number of times each character appears in the token.
            label_seq: torch.Tensor, shape=(max_length_each_token, ), dtype=torch.int, device=self.tensor.device, the sequence of the characters. The table is char2label. Padding with len(char2label).
        """
        token = self.id2token[self.ids[idx]]
        inputs = self.tensor[self.shuffle_ids[idx] if self.shuffle_baseline else self.ids[idx]]

        indices = torch.tensor([self.char2label[char] for char in token]) # (len(token), )
        if self.label_target == "exist":
            # label_exist = torch.zeros(self.char_num, device=self.tensor.device).scatter_(0, indices, 1).float() # (char_num, ), float
            label_exist = torch.zeros(self.char_num).scatter_(0, indices, 1).float() # (char_num, ), float
            return inputs, label_exist
        if self.label_target == "number":
            # count the number of times each character appears in the token
            numbers = torch.bincount(indices, minlength=self.char_num) # (char_num, )
            numbers = torch.minimum(numbers, torch.tensor(self.max_num_each_char, dtype=numbers.dtype)) # (char_num, )
            assert numbers.shape == (self.char_num, ), "The shape of numbers is not correct."
            label_number = numbers
            return inputs, label_number
        if self.label_target == "seq":
            padding_idx = len(self.char2label)
            if indices.shape[0] > self.max_length_each_token:
                logger.error(f"The length of the token {token} with idx {idx} is {indices.shape[0]}, which is greater than max_length_each_token {self.max_length_each_token}. Consider increasing max_length_each_token.")
                label_sequence = indices[: self.max_length_each_token]
            else:
                label_sequence = torch.concat([
                                    indices, 
                                    torch.ones(size=(self.max_length_each_token - indices.shape[0], ), dtype=torch.int) * padding_idx
                                    ], dim = 0) # (max_length_each_token, )
            return inputs, label_sequence
        raise ValueError(f"Unknown label_target: {self.label_target}")
    
    def random_split(self, nums: Sequence[int], random_rng: torch.Generator | None = None) -> list[Self]:
        if sum(nums) != len(self):
            raise ValueError('The sum of nums must be equal to the length of the dataset')
        indices = torch.randperm(len(self), generator=random_rng).tolist()
        ids: list[int] = [self.ids[i] for i in indices]
        return [self.__class__(self.tensor, self.id2token, self.token2id, self.char2label, 
                               ids=ids[start:end], 
                               max_num_each_char=self.max_num_each_char, 
                               max_length_each_token=self.max_length_each_token,
                               label_target=self.label_target, # type: ignore
                               shuffle_baseline=self.shuffle_baseline, 
                               random_rng=random_rng)
                for start, end in zip([0] + list(torch.cumsum(torch.tensor(nums), 0)), list(torch.cumsum(torch.tensor(nums), 0)))]
