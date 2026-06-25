from pathlib import Path
from typing import Any, Literal, cast

import h5py
import numpy as np
import torch
import torch.utils
import torch.utils.data

from .dataclass import RichTestData



class RichHdf5Dataset(torch.utils.data.Dataset[RichTestData]):
    def __init__(   
        self,
        hdf5_path: Path,
        file_list_path: Path,
        subseq_len: int,
    ) -> None:
        self._hdf5_path = hdf5_path
        self._subseq_len = subseq_len
        
        self.hdf5_file: h5py.File | None = None


        self._indices: list[tuple[str, int]] = []
        
        all_groups = file_list_path.read_text().splitlines()

        with h5py.File(self._hdf5_path, "r") as f:
            for group_name in all_groups:
                if group_name not in f:
                    print(f"Warning: Group {group_name} not found in HDF5 file. Skipping.")
                    continue
                
                group = f[group_name]
                total_t = cast(h5py.Dataset, group["T_world_root"]).shape[0]

                if total_t >= subseq_len:
                    for start_t in range(0, total_t - subseq_len + 1, subseq_len):
                        self._indices.append((group_name, start_t))
        
        assert len(self._indices) > 0, "No valid data sequences found!"
        print(f"Found {len(self._indices)} deterministic subsequences.")

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> RichTestData:
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self._hdf5_path, "r")

        group_name, start_t = self._indices[index]
        end_t = start_t + self._subseq_len

        npz_group = self.hdf5_file[group_name]
        
        kwargs: dict[str, Any] = {}
        for k in npz_group.keys():
            v = npz_group[k]
            
            if k == "betas":
                array = v[:]
            else:
                array = v[start_t:end_t]

            kwargs[k] = torch.from_numpy(np.array(array))
        
        return RichTestData(**kwargs)
    
    
class EgoRichHdf5Dataset(torch.utils.data.Dataset[RichTestData]):
    """
    Loads preprocessed RICH data from an HDF5 file.
    This loader is structured similarly to the EgoAmassHdf5Dataset,
    but simplified for a single dataset without train/val/test splits.
    """

    def __init__(
        self,
        hdf5_path: Path,
        file_list_path: Path,
        subseq_len: int,
        cache_files: bool,
        slice_strategy: Literal[
            "deterministic", "random_uniform_len", "random_variable_len"
        ],
        min_subseq_len: int | None = None,
        random_variable_len_proportion: float = 0.3,
        random_variable_len_min: int = 16,
    ) -> None:
        self._slice_strategy = slice_strategy
        self._random_variable_len_proportion = random_variable_len_proportion
        self._random_variable_len_min = random_variable_len_min
        self._hdf5_path = hdf5_path

        # Use a key that is confirmed to exist for checking sequence validity.
        check_key = "T_world_root"

        with h5py.File(self._hdf5_path, "r") as hdf5_file:
            all_potential_groups = file_list_path.read_text().splitlines()

            # Create a list of valid sequences from the HDF5 file
            self._groups = [
                p
                for p in all_potential_groups
                if p in hdf5_file
                and check_key in hdf5_file[p]
                and cast(
                    h5py.Dataset,
                    cast(h5py.Group, hdf5_file[p])[check_key],
                ).shape[0]
                >= (subseq_len if min_subseq_len is None else min_subseq_len)
            ]
            self._subseq_len = subseq_len

            if not self._groups:
                raise ValueError(
                    "No valid sequences found. Check if the file list and HDF5 file match, "
                    f"and if sequences are long enough (>= {subseq_len} frames)."
                )

            # Approximate the total number of subsequences in the dataset.
            self._approximated_length = (
                sum(
                    cast(
                        h5py.Dataset, cast(h5py.Group, hdf5_file[g])[check_key]
                    ).shape[0]
                    for g in self._groups
                )
                // subseq_len
            )

        self._cache: dict[str, dict[str, Any]] | None = {} if cache_files else None

    def __getitem__(self, index: int) -> RichTestData:
        group_index = index % len(self._groups)
        slice_index = index // len(self._groups)
        del index

        group = self._groups[group_index]

        hdf5_file = None
        if self._cache is not None:
            if group not in self._cache:
                hdf5_file = h5py.File(self._hdf5_path, "r")
                assert hdf5_file is not None
                self._cache[group] = {
                    k: np.array(v)
                    for k, v in cast(h5py.Group, hdf5_file[group]).items()
                }
            npz_group = self._cache[group]
        else:
            hdf5_file = h5py.File(self._hdf5_path, "r")
            npz_group = hdf5_file[group]
            assert isinstance(npz_group, h5py.Group)

        total_t = cast(h5py.Dataset, npz_group["T_world_root"]).shape[0]
        assert total_t >= self._subseq_len

        # Determine slice indexing.
        mask = torch.ones(self._subseq_len, dtype=torch.bool)
        if self._slice_strategy == "deterministic":
            valid_start_indices = total_t - self._subseq_len
            start_t = (
                (slice_index * self._subseq_len) % valid_start_indices
                if valid_start_indices > 0
                else 0
            )
            end_t = start_t + self._subseq_len
        elif self._slice_strategy == "random_uniform_len":
            start_t = np.random.randint(0, total_t - self._subseq_len + 1)
            end_t = start_t + self._subseq_len
        elif self._slice_strategy == "random_variable_len":
            random_subseq_len = min(
                (
                    np.random.randint(self._random_variable_len_min, self._subseq_len)
                    if np.random.random() < self._random_variable_len_proportion
                    else self._subseq_len
                ),
                total_t,
            )
            start_t = np.random.randint(0, total_t - random_subseq_len + 1)
            end_t = start_t + random_subseq_len
            mask[random_subseq_len:] = False
        else:
            assert_never(self._slice_strategy)

        # Read slices of the dataset.
        kwargs: dict[str, Any] = {}
        for k in npz_group.keys():
            v = npz_group[k]
            assert isinstance(k, str)
            assert isinstance(v, (h5py.Dataset, np.ndarray))

            # --- 수정된 부분 ---
            # 'betas'는 시퀀스 전체에 걸쳐 고정된 값이므로 특별히 처리합니다.
            if k == "betas":
                # betas는 (1, 16) 또는 (16,) 형태일 수 있습니다.
                # 시간 차원이 없으므로 전체를 가져옵니다.
                array = v[:]
            else:
                # 다른 모든 데이터는 시계열 데이터로 간주합니다.
                assert v.shape[0] == total_t
                array = v[start_t:end_t]
            # --- 수정 끝 ---

            # Pad to subsequence length if necessary.
            if array.shape[0] != self._subseq_len:
                array = np.concatenate(
                    [
                        array,
                        np.repeat(
                            array[-1:,], self._subseq_len - array.shape[0], axis=0
                        ),
                    ],
                    axis=0,
                )
            kwargs[k] = torch.from_numpy(array)
        kwargs["mask"] = mask

        if hdf5_file is not None:
            hdf5_file.close()

        return RichTestData(**kwargs)

    def __len__(self) -> int:
        return self._approximated_length
    

class RichHdf5Dataset_FINAL(torch.utils.data.Dataset[RichTestData]):
    """
    RICH 데이터셋을 위해 수정된 HDF5 데이터셋 클래스.
    HDF5 파일 전체를 하나의 연속된 데이터 시퀀스로 취급합니다.

    Args:
        hdf5_path: Path to the HDF5 file containing the dataset.
        subseq_len: Length of subsequences to sample from the dataset.
        cache_files: Whether to cache the entire dataset in memory.
        slice_strategy: Slicing strategy for sampling subsequences.
        ... (and other slicing-related arguments)
    """

    def __init__(
        self,
        hdf5_path: Path,
        file_list_path: Path,
        subseq_len: int,
        cache_files: bool,
        slice_strategy: Literal[
            "deterministic",
            "random_uniform_len",
            "random_variable_len",
            "full_seq",               # ✅ 추가
        ],
        min_subseq_len: int | None = None,
        random_variable_len_proportion: float = 0.3,
        random_variable_len_min: int = 16,
    ) -> None:
        self._hdf5_path = hdf5_path
        self._subseq_len = subseq_len
        self._slice_strategy = slice_strategy
        self._random_variable_len_proportion = random_variable_len_proportion
        self._random_variable_len_min = random_variable_len_min

        with h5py.File(self._hdf5_path, "r") as hdf5_file:
            # txt 파일 목록을 순서대로 읽고, 최소 길이를 만족하는 그룹만 필터링합니다.
            # split 필터링 로직을 제거하여 txt 파일의 모든 내용을 사용합니다.
            self._groups = [
                p
                for p in file_list_path.read_text().splitlines()
                if cast(h5py.Group, hdf5_file[p])["T_world_root"].shape[0]
                >= (subseq_len if min_subseq_len is None else min_subseq_len)
            ]
            
            assert len(self._groups) > 0, "No valid data groups found for the specified length in file list."
            assert len(cast(h5py.Group, hdf5_file[self._groups[0]]).keys()) > 0

            # ✅ full_seq에서는 "각 group 당 1개"가 더 자연스러움
            if self._slice_strategy == "full_seq":
                self._approximated_length = len(self._groups)
            else:
                self._approximated_length = (
                    sum(
                        cast(h5py.Dataset, cast(h5py.Group, hdf5_file[g])["T_world_root"]).shape[0]
                        for g in self._groups
                    )
                    // subseq_len
                )

        self._cache: dict[str, dict[str, Any]] | None = {} if cache_files else None

    def __getitem__(self, index: int) -> RichTestData:
        # ✅ full_seq면 group당 1개만 뽑는게 자연스러우니 index를 group index로 사용
        if self._slice_strategy == "full_seq":
            group_index = index % len(self._groups)
            slice_index = 0
        else:
            group_index = index % len(self._groups)
            slice_index = index // len(self._groups)
        del index

        group = self._groups[group_index]

        hdf5_file = None
        # 캐시를 사용하거나, 사용하지 않을 경우 파일을 직접 열어서 데이터 소스를 가져옴
        if self._cache is not None:
            if group not in self._cache:
                hdf5_file = h5py.File(self._hdf5_path, "r")
                assert hdf5_file is not None
                self._cache[group] = {
                    k: np.array(v)
                    for k, v in cast(h5py.Group, hdf5_file[group]).items()
                }
            npz_group = self._cache[group]
        else:
            hdf5_file = h5py.File(self._hdf5_path, "r")
            npz_group = hdf5_file[group]
            assert isinstance(npz_group, h5py.Group)

        total_t = cast(h5py.Dataset, npz_group["T_world_root"]).shape[0]
        assert total_t >= self._subseq_len

        # 슬라이싱 전략에 따라 start_t와 end_t를 결정
        # -------------------------
        # slice 전략
        # -------------------------
        if self._slice_strategy == "full_seq":
            start_t = 0
            end_t = total_t
            mask = torch.ones(total_t, dtype=torch.bool)  # ✅ full length mask
            target_len = total_t                          # ✅ padding 없음
        else:
            mask = torch.ones(self._subseq_len, dtype=torch.bool)
            target_len = self._subseq_len
            if self._slice_strategy == "deterministic":
                valid_start_indices = total_t - self._subseq_len
                start_t = (
                    (slice_index * self._subseq_len) % valid_start_indices
                    if valid_start_indices > 0
                    else 0
                )
                end_t = start_t + self._subseq_len
            elif self._slice_strategy == "random_uniform_len":
                start_t = np.random.randint(0, total_t - self._subseq_len + 1)
                end_t = start_t + self._subseq_len
            elif self._slice_strategy == "random_variable_len":
                random_subseq_len = min(
                    (
                        np.random.randint(self._random_variable_len_min, self._subseq_len)
                        if np.random.random() < self._random_variable_len_proportion
                        else self._subseq_len
                    ),
                    total_t,
                )
                start_t = np.random.randint(0, total_t - random_subseq_len + 1)
                end_t = start_t + random_subseq_len
                mask[random_subseq_len:] = False
            else:
                raise ValueError(f"Unknown slice_strategy: {self._slice_strategy}")

        # 결정된 슬라이스에 따라 데이터셋을 읽어옴
        kwargs: dict[str, Any] = {}
        for k in npz_group.keys():
            if k == "joints_wrt_world":
                continue

            v = npz_group[k]
            assert isinstance(k, str)
            assert isinstance(v, (h5py.Dataset, np.ndarray))

            # --- RICH 데이터셋을 위한 수정 ---
            if k == "betas":
                # RICH 데이터의 betas는 10개, AMASS는 16개일 수 있으므로 유연하게 처리
                betas_array = v[:]
                if betas_array.shape[-1] == 10:
                    # SMPL-H 모델(16개)에 맞추기 위해 6개의 0을 패딩
                    padding = np.zeros((*betas_array.shape[:-1], 6), dtype=betas_array.dtype)
                    array = np.concatenate([betas_array, padding], axis=-1)
                else:
                    array = betas_array
                
                assert array.shape[-1] == 16
            # --- 수정 끝 ---
            else:
                # 그 외의 데이터는 시간 축을 기준으로 슬라이싱
                assert v.shape[0] == total_t
                array = v[start_t:end_t]

            # ✅ full_seq에서는 padding 하지 않음
            if self._slice_strategy != "full_seq":
                if array.shape[0] != target_len:
                    array = np.concatenate(
                        [array, np.repeat(array[-1:], target_len - array.shape[0], axis=0)],
                        axis=0,
                    )

            kwargs[k] = torch.from_numpy(array)
        kwargs["mask"] = mask

        if "hand_quats" not in kwargs:
            kwargs["hand_quats"] = None

        if hdf5_file is not None:
            hdf5_file.close()

        return RichTestData(**kwargs)

    def __len__(self) -> int:
        return self._approximated_length