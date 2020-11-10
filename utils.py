import argparse


def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_multiple_entity(tag_seq, char_seq):
    entity_type = "0"
    entity = ""
    entity_dict = {}
    entity_str = ""
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        tag = str(tag)
        if tag.split("-")[0] == "B":  # 开始了一个新实体
            save_pre_entity(entity, entity_type, entity_dict, i)
            if len(entity):
                entity_str += entity + "</{}>".format(entity_type)
            entity_type = tag.split("-")[-1]
            entity = char
            entity_str += "<{}>".format(entity_type)
        elif tag.split("-")[0] == "I":  # 实体的中间
            entity += char
        else:  # 遇到了一个O
            if len(entity):
                entity_str += entity + "</{}>".format(entity_type)
            entity, entity_type = save_pre_entity(entity, entity_type, entity_dict, i)
            entity_str += char
    if len(entity):
        entity_str += entity + "</{}>".format(entity_type)
        save_pre_entity(entity, entity_type, entity_dict, len(char_seq))
    # print(entity_str)
    return entity_dict


def save_pre_entity(entity, entity_type, entity_dict, end_idx):
    if not len(entity) or entity_type == '0' or entity_type == 'ORG' or entity == "中国":
        return "", '0'
    if entity_type not in entity_dict:
        entity_dict[entity_type] = []
    start_idx = end_idx - len(entity)
    entity_dict[entity_type].append((start_idx, end_idx))
    return "", '0'



