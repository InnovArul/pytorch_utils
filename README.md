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
    * 
* generic/torchtools
    * save_checkpoint
    * load_checkpoint
    * resume_from_checkpoint
    * open_all_layers
    * open_specified_layers
    * count_num_param 
    * load_pretrained_weights
    * get_current_time
    * save_scripts
    * load_image_in_PIL
    * print_cuda_mem
    * set_bn_eval

* optim/optimizer
    * build_optimizer

* optim/lr_scheduler
    * build_lr_scheduler
