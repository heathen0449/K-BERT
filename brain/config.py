import os


FILE_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
OLD_FILE_DIR_PATH = os.path.dirname(FILE_DIR_PATH)
KGS = {
    'HowNet': os.path.join(FILE_DIR_PATH, 'kgs/HowNet.spo'),
    'CnDbpedia': os.path.join(FILE_DIR_PATH, 'kgs/CnDbpedia.spo'),
    'Medical': os.path.join(FILE_DIR_PATH, 'kgs/Medical.spo'),
    'Industry': os.path.join(OLD_FILE_DIR_PATH,'KG_private/output/所属行业.spo'),
    'ChildComp': os.path.join(OLD_FILE_DIR_PATH, 'KG_private/output/子公司.spo'),
    'Concept': os.path.join(OLD_FILE_DIR_PATH, 'KG_private/output/所属概念.spo'),
    'CoreBus': os.path.join(OLD_FILE_DIR_PATH, 'KG_private/output/主营业务.spo')
}



MAX_ENTITIES = 2

# Special token words.
PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
MASK_TOKEN = '[MASK]'
ENT_TOKEN = '[ENT]'
SUB_TOKEN = '[SUB]'
PRE_TOKEN = '[PRE]'
OBJ_TOKEN = '[OBJ]'

NEVER_SPLIT_TAG = [
    PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN,
    ENT_TOKEN, SUB_TOKEN, PRE_TOKEN, OBJ_TOKEN
]
