from enum import StrEnum


class Split(StrEnum):
    TRAIN = 'train'
    TEST = ' test'


class Modality(StrEnum):
    SCRATCH = 'ht'
    GOOGLE = 'pe1'
    MBART = 'pe2'


class ConfigMode(StrEnum):
    FULL = 'full'
    MASK_SUBJECT = 'mask_subject'
    MASK_MODALITY = 'mask_modality'
    MASK_TIME = 'mask_time'
