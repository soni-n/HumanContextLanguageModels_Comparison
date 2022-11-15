from data.utils_hart.hulm_data_utils import load_dataset as load_hulm_dataset
from data.utils_hart.ft_user_data_utils import load_dataset as load_user_ft_dataset

def load_dataset(logger, tokenizer, table, block_size, max_blocks, data_args, data_type, disable_hulm_batching):
    fields = get_fields()
    if 'pkl' in table:
        data = get_data_from_pkl(logger, table, fields, data_type)
    elif 'csv' in table:
        data = get_data_from_csv(logger, table, fields, data_type)
    else:
        data = get_data_from_db(logger, table, data_args, data_type)
    data = transform_data(logger, tokenizer, data, block_size)
    logger.info('************** Block size = {} *************'.format(block_size))
    if not disable_hulm_batching:
        return group_data(data, max_blocks, logger) 
    else:
        instances, uncut_num_blocks = group_data(data, max_blocks, logger)
        flat_list = [item for sublist in instances for item in sublist if item is not None]
        return flat_list, uncut_num_blocks

