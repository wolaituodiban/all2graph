from typing import Union


class Event:
    def __init__(
        self,
        obs_timestamp: Union[int, float],
        event_timestamp: Union[int, float, None],
        censor_timestamp: Union[int, float, None],
        event_type: str,
        attributes: dict,
        foreign_keys: dict,
    ):
        """
        obs_timestamp: 观察时间
        event_timestamp: 时间时间
        censor_timestamp: 审查时间
        event_type: 事件类型
        attributes: 事件属性
        foreign_keys: 外键
        """
        self.obs_timestamp = obs_timestamp
        self.event_timestamp = event_timestamp
        self.censor_timestamp = censor_timestamp
        self.event_type = event_type
        self.attributes = attributes
        self.foreign_keys = foreign_keys

    def __repr__(self):
        output = f'{self.__class__.__name__}(\n'
        output += f'  obs_timestamp={self.obs_timestamp},\n'
        output += f'  event_timestamp={self.event_timestamp},\n'
        output += f'  censor_timestamp={self.censor_timestamp},\n'
        output += f'  event_type={self.event_type},\n'
        output += f'  attributes={self.attributes},\n'
        output += f'  foreign_keys={self.foreign_keys}\n'
        output += ')'
        return output
    