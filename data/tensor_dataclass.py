import dataclasses
import torch


class TensorDataclass:

    def __init_subclass__(cls) -> None:
        dataclasses.dataclass(cls)

    def to(self, device: torch.device | str):
        return self.map(lambda x: x.to(device))

    def as_nested_dict(self, numpy: bool) -> dict:
        def _to_dict(obj):
            if isinstance(obj, TensorDataclass):
                return {k: _to_dict(v) for k, v in vars(obj).items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(_to_dict(v) for v in obj)
            elif isinstance(obj, dict):
                return {k: _to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, torch.Tensor) and numpy:
                return obj.numpy(force=True)
            else:
                return obj

        return _to_dict(self)

    def map(self, fn):

        def _map_impl(
            fn,
            val,
        ):
            if isinstance(val, torch.Tensor):
                return fn(val)
            elif isinstance(val, TensorDataclass):
                return type(val)(**_map_impl(fn, vars(val)))
            elif isinstance(val, (list, tuple)):
                return type(val)(_map_impl(fn, v) for v in val)
            elif isinstance(val, dict):
                assert type(val) is dict 
                return {k: _map_impl(fn, v) for k, v in val.items()}
            else:
                return val

        return _map_impl(fn, self)