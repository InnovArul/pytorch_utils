# pytorch utils
generic utilities for pytorch ecosystem collected from various places

# Credits

Majority of utility functions are collected from the repositories: 

* [deep_person_reid](https://github.com/KaiyangZhou/deep-person-reid/) by [KaiyangZhou](https://github.com/KaiyangZhou)


Thanks to the author(s). 

## Available methods

* generic/avgmeter
    * AverageMeter

* generic/loggers
    * Logger
    * disable_all_print_once
    * print_once

* generic/model_complexity
    * compute_model_complexity

* generic/tools
    * mkdir_if_missing
    * check_isfile 
    * read_json
    * write_json
    * set_random_seed
    * download_url 
    * read_image
    * collect_env_info
    * get_current_time
    * save_scripts
    * load_image_in_PIL

* generic/torchtools
    * save_checkpoint
    * load_checkpoint
    * resume_from_checkpoint
    * open_all_layers
    * open_specified_layers
    * count_num_param 
    * load_pretrained_weights
    * set_bn_eval
    * set_seed
    * print_cuda_mem

* optim/optimizer
    * build_optimizer

* optim/lr_scheduler
    * build_lr_scheduler
