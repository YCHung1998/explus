import enum


class LabelMap(str, enum.Enum):
    STABLE = 'Stable'
    POSITIVE_UNSTABLE = 'Positive Unstable'
    NEGATIVE_UNSTABLE_EXTERNAL_DISTURBANCES = \
        'Negative Unstable External Disturbances'
    NEGATIVE_UNSTABLE_OVEREXPOSURE = 'Negative Unstable Overexposure'
    TRIGGER = 'Trigger'

    def __str__(self) -> str:
        return self.value


class TriggerMode(str, enum.Enum):
    EMAPHASH = 'emaphash'
    YOLO = 'yolo'
    def __str__(self) -> str:
        return self.value


# signal_processor.py
class SignalState(str, enum.Enum):
    STABLE = 0
    UNSTABLE = 1

    def __str__(self) -> str:
        return self.value


LABEL_MAP = {
    1: LabelMap.STABLE,
    2: LabelMap.POSITIVE_UNSTABLE,
    3: LabelMap.NEGATIVE_UNSTABLE_EXTERNAL_DISTURBANCES,
    4: LabelMap.NEGATIVE_UNSTABLE_OVEREXPOSURE,
    5: LabelMap.TRIGGER
}
LABEL_NAME2ID = dict(zip(LABEL_MAP.values(), LABEL_MAP.keys()))

if __name__ == '__main__':

    print(LabelMap.STABLE)
    print("LABEL_NAME2ID[LabelMap.STABLE]", LABEL_NAME2ID[LabelMap.STABLE])
