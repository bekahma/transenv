from dataclasses import dataclass, field



@dataclass
class GenerationConfig:
    temperature: float = field(default=0.8, metadata={"help": "temperature value for generation"})
    top_p: float = field(default=0.95, metadata={"help": "top_p value for generation"})
    batch_size: int = field(default=30, metadata={"help": "batch size for generation"})
    openai_parallelism: int = field(default=1, metadata={"help": "Maximum hosted OpenAI requests to keep in flight"})
    max_tokens: int = field(default=2000)
    max_samples: int = field(default=None, metadata={"help": "Maximum number of input rows to transform after filtering"})
    max_rules_per_chunk: int = field(default=None, metadata={"help": "Maximum accepted feature rules per transformed chunk"})
    max_rules_per_row: int = field(default=None, metadata={"help": "Maximum accepted feature-rule applications per cefr_texts row"})
    rerun: str = field(default=None)



@dataclass
class ModelConfig:
    model_name: str = field(default="google/gemma-2-27b-it", metadata={"help": "Model to Use", "choices": ['google/gemma-2-27b-it', 'gpt-4.1-mini', 'gpt-4o-mini']})
    model_provider: str = field(default="auto", metadata={"help": "Model provider: auto, local, or openai"})
    port_num: int = field(default=8000, metadata={"help": "vLLM port number for local provider"})
    tokenizer: str = field(default='google/gemma-2-27b-it')
    openai_base_url: str = field(default=None, metadata={"help": "Optional OpenAI-compatible base URL for hosted OpenAI provider"})
    semantic_model_name: str = field(default=None, metadata={"help": "Optional model for semantic checking; defaults to model_name"})



@dataclass
class DatasetConfig:
    dataset_name: str = field(metadata={"help": "Dataset name", "choices": ['mmlu', 'gsm8k', 'arc', 'hellaswag', 'truthfulqa', 'winogrande', 'cefr_texts']})
    sampling: bool = field(default=False, metadata={"action": "store_true"})
    input_path: str = field(default=None, metadata={"help": "Local CSV input path for cefr_texts"})
    text_column: str = field(default=None, metadata={"help": "Text column in a cefr_texts CSV"})
    input_cefr_levels: str = field(default=None, metadata={"help": "Comma-separated input CEFR levels to keep, e.g. A1,A2"})
    text_chunking: str = field(default="row", metadata={"help": "cefr_texts chunking mode: row, hybrid, or sentence"})
    max_chunk_words: int = field(default=80, metadata={"help": "Split sentence chunks longer than this many words"})
    sentence_chunk_min_words: int = field(default=100, metadata={"help": "Use sentence chunks only above this row word count when text_chunking=hybrid"})

    

@dataclass
class TaskConfig:
    task_name: str = field(default="english_dialect", metadata={"choices": ['english_dialect', 'cefr', 'L1']})
    dialect: str = field(default="Urban African American Vernacular English", metadata={"help": "English Dialect"})
    l1: str = field(default='Arabic', metadata={"choices": ['Arabic', 'French', 'German', 'Italian', 'Japanese', 'Mandarin', 'Portuguese', 'Russian', 'Spanish', 'Turkish']})
    cefr_level: str = field(default="A", metadata={"help": "CEFR Level"})



@dataclass
class SaveConfig:
    save_path: str = field(default="./")
    file_name: str = field(default="tmp.pk", metadata={"help": "save file name"})
    data_path: str = field(default="./")
