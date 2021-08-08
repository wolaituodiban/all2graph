import json
from .array_node import ArrayNode
from .category import Category
from .meta_node import MetaNode
from .object_node import ObjectNode
from .timestamp import TimeStamp, ALL_TIME_UNITS, SECOND_DIFF
from ..stats import ECDF


class JsonValue(MetaNode):
    __doc__ = """
    参照https://www.json.org/json-en.html的标准编制

    json是由collection of name/value pairs和ordered list of values组成的。一般来说list本身会对应一个name。
    为了让所有的value都能对应上一个name，我们把list中的所有value都对应到list本身的name上时，这样所有的value就都有了name上。

    JsonValue对象会记录某一个name的各种不同value的出现频率和分布
    
    JsonValue目前拓展了json的value的类型，包括一下几种：
        string：一般的value会被归为离散值
        number：一切pd.to_numeric能转换的对象
        object: collection of name/value pairs，一切isinstance(obj, dict)的对象
        array：ordered list of values，一切isinstance(obj, list)的对象
        true：对应True
        false：对应False
        null：一切pd.isna为True的对象
        timestamp：特殊的字符串，一切pd.to_datetime能转换的对象
        id：特殊的string，表示某个实体，当json重复出现相同的id时，一个新的节点会被创建，原来的节点会连接到新的节点上
        
    为了保证完备性，每一个value会按照上述所述的类型顺序，被唯一的归入其中一类。
    但是id是特殊的，因为目前的技术条件下，很难通过机器判断一个字符串是否是id，所以需要用户指定某一个name是否为id类型。
    
    JsonValue会包含以下属性：
        string_freq:    ECDF型，记录样本口径下，value为string的频率
        string_attr:    Catgory型，记录样本口径下，每一个字符串的出现频率
        number_freq:    ECDF型，记录样本口径下，value为number的频率
        number_attr:    ECDF型，记录节点口径下，number的累计分布函数
        object_freq:    ECDF型，记录样本口径下，value为object的频率
        object_attr:    ObjectNode型，记录object的属性
        array_freq:     ECDF型，记录样本口径下，value为array的频率
        array_attr:     ArrayNode型，记录array的属性
        true_freq:      ECDF型，记录样本口径下，value为true的频率
        false_freq:     ECDF型，记录样本口径下，value为false的频率
        null_freq:      ECDF型，记录样本口径下，value为null的频率
        timestamp_freq: ECDF型，记录样本口径下，value为timestamp的频率
        timestamp_attr: TimeStamp型，记录节点口径下，timestamp的属性的累计分布函数，
            属性包括{0}和{1}，其中{0}的小数点后最高精确到纳秒
    """.format(SECOND_DIFF, ALL_TIME_UNITS)

    def __init__(
            self,
            string_freq: ECDF = None,
            string_attr: Category = None,
            num_freq: ECDF = None,
            num_attr: ECDF = None,
            obj_freq: ECDF = None,
            obj_attr: ObjectNode = None,
            array_freq: ECDF = None,
            array_attr: ArrayNode = None,
            true_freq: ECDF = None,
            false_freq: ECDF = None,
            null_freq: ECDF = None,
            timestamp_freq: ECDF = None,
            timestamp_attr: TimeStamp = None,
            **kwargs
    ):
        # todo 判断freq的均值和attr的节点数是否一致
        super().__init__(**kwargs)
        freqs = {string_freq, num_freq, obj_freq, array_freq, true_freq, false_freq, num_freq, timestamp_freq}
        assert freqs != {None}, '频率不能全为None'
        assert len({freq.num_samples for freq in freqs if freq is not None}), '所有频率的样本数必须相同'

        assert (string_freq is None) == (string_attr is None), '字符串频率和字符串属性必须同时有，或者同时无'
        self.string_freq = string_freq
        self.string_attr = string_attr

        assert (num_freq is None) == (num_attr is None), '数值频率和数值属性必须同时有，或者同时无'
        self.num_freq = num_freq
        self.num_attr = num_attr

        assert (obj_freq is None) == (obj_attr is None), '键值对频率和键值对属性必须同时有，或者同时无'
        self.obj_freq = obj_freq
        self.obj_attr = obj_attr

        assert (array_freq is None) == (array_attr is None), '有序列表频率和有序列表属性必须同时有，或者同时无'
        self.array_freq = array_freq
        self.array_attr = array_attr

        self.true_freq = true_freq
        self.false_freq = false_freq
        self.null_freq = null_freq

        assert (timestamp_freq is None) == (timestamp_attr is None), '时间戳频率和时间戳属性必须同时有，或者同时无'
        self.timestamp_freq = timestamp_freq
        self.timestamp_attr = timestamp_attr
