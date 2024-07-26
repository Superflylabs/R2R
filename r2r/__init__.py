import os
import logging

# Keep '*' imports for enhanced development velocity
# corresponding flake8 error codes are F403, F405
from .base import *
from .integrations import *
from .main import *
from .parsers import *
from .pipelines import *
from .pipes import *
from .prompts import *


# Configure the log verbosity via environment variable
log_level_env = os.getenv("R2R_LOG_LEVEL", "DEBUG").upper()
log_level = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}.get(log_level_env, logging.DEBUG)

logger = logging.getLogger("r2r")
logger.setLevel(log_level)

# Create a console handler and set the level to info
ch = logging.StreamHandler()
ch.setLevel(log_level)

# Create a formatter and set it for the handler
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
ch.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(ch)

# Optional: Prevent propagation to the root logger
logger.propagate = True

__all__ = [
    "R2RException",
    "LoggingConfig",
    "LocalKVLoggingProvider",
    "PostgresLoggingConfig",
    "PostgresKVLoggingProvider",
    "RedisLoggingConfig",
    "RedisKVLoggingProvider",
    "KVLoggingSingleton",
    "VectorEntry",
    "VectorType",
    "Vector",
    "VectorSearchRequest",
    "VectorSearchResult",
    "AsyncPipe",
    "PipeType",
    "AsyncState",
    "Prompt",
    "DataType",
    "DocumentType",
    "Document",
    "Extraction",
    "ExtractionType",
    "Fragment",
    "FragmentType",
    "SearchPipe",
    # Parsers
    "AsyncParser",
    "CSVParser",
    "DOCXParser",
    "HTMLParser",
    "JSONParser",
    "MDParser",
    "PDFParser",
    "PPTParser",
    "TextParser",
    "XLSXParser",
    "AsyncPipeline",
    # Providers
    "EmbeddingConfig",
    "EmbeddingProvider",
    "EvalConfig",
    "EvalProvider",
    "LLMEvalProvider",
    "PromptConfig",
    "PromptProvider",
    "GenerationConfig",
    "LLMChatCompletion",
    "LLMChatCompletionChunk",
    "LLMConfig",
    "LLMProvider",
    "VectorDBConfig",
    "VectorDBProvider",
    "R2RConfig",
    "TextSplitter",
    "RecursiveCharacterTextSplitter",
    "generate_run_id",
    "generate_id_from_label",
    "R2REngine",
    # Pipes
    "EmbeddingPipe",
    "EvalPipe",
    "ParsingPipe",
    "QueryTransformPipe",
    "SearchRAGPipe",
    "StreamingSearchRAGPipe",
    "VectorSearchPipe",
    "VectorStoragePipe",
    "R2RPromptProvider",
    "WebSearchPipe",
    "R2RBuilder",
    "R2R",
    "KGAgentSearchPipe",
    # Prebuilts
    "MultiSearchPipe",
    "R2RPipeFactoryWithMultiSearch",
    # Integrations
    "SerperClient",
]
