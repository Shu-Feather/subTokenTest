class Config:
    def __init__(self, model_type, model_name, task_type="both", num_samples=100, max_depth=4, 
                 output_dir="./results", verbose=False, api_key=None, base_url=None, test_file=None,
                 restricted_reasoning: bool = False):
        self.model_type = model_type
        self.model_name = model_name
        self.task_type = task_type
        self.num_samples = num_samples
        self.max_depth = max_depth
        self.output_dir = output_dir
        self.verbose = verbose
        self.api_key = api_key
        self.base_url = base_url
        self.test_file = test_file
        self.restricted_reasoning = restricted_reasoning
        
        # Task distribution based on task_type
        if task_type == "task1":
            self.task1_ratio = 1.0  # 100% task1
        elif task_type == "task2":
            self.task1_ratio = 0.0  # 100% task2
        else:  # both
            self.task1_ratio = 0.5  # 50% for task1, 50% for task2
        
        # VLLM settings
        self.vllm_tensor_parallel_size = 1
        self.vllm_gpu_memory_utilization = 0.9
        self.vllm_max_model_len = 2048
        self.vllm_enforce_eager = True
        self.vllm_batch_size = 4
        
        # Generation settings
        self.max_new_tokens = 32768
        self.temperature = 0.6
        self.top_p = 0.95
        self.reasoning_effort = "low"  # options: "low", "medium", "high"
