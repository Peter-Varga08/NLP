from enum import Enum


class Split(str, Enum):
    TRAIN = 'train'
    TEST = 'test'


class Modality(str, Enum):
    SCRATCH = 'ht'
    GOOGLE = 'pe1'
    MBART = 'pe2'


class ConfigMode(str, Enum):
    FULL = 'full'
    MASK_SUBJECT = 'mask_subject'
    MASK_MODALITY = 'mask_modality'
    MASK_TIME = 'mask_time'


class DatasetType(str, Enum):
    LOGGING = 'logging'
    LINGUISTIC = 'linguistic'
    BOTH = 'both'
